"""Run one ablation phase (phase1_base or phase2_ft) end-to-end.

Usage::

    uv run python scripts/run_evaluation_phase.py \\
        --phase phase1_base \\
        --ablation-id 2026-05-21T1430 \\
        --sample-indices data/processed/eval_sample_indices.json \\
        --llm-endpoint http://localhost:8001/v1 \\
        --expected-model llama-3.1-8b-instruct-q4

Workflow: hard preflight checks → load eval sample → build RAG pipeline →
for each config in this phase {run_config → compute_ragas_metrics → aggregate} →
log MLflow phase parent + nested children → post-run sanity warnings.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from src.config import load_config
from src.evaluation.aggregation import aggregate_config
from src.evaluation.metrics import compute_ragas_metrics
from src.evaluation.mlflow_logging import log_phase_run
from src.evaluation.runner import ConfigSpec, run_config
from src.evaluation.sample import load_eval_sample
from src.evaluation.server_check import validate_llamacpp_model
from src.llm_clients import OpenAICompatibleClient
from src.logging_config import get_logger
from src.rag.embedder import DenseEmbedder, SparseEmbedder
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import MedCPTReranker
from src.rag.retriever import Retriever
from src.rag.scorer import ClinicalScorer
from src.tracking import setup_tracking_uri

logger = get_logger(__name__)

_JUDGE_MODEL = "claude-sonnet-4-6"
_EMBEDDING_MODEL = "text-embedding-3-small"

_PHASE_CONFIGS = {
    "phase1_base": [
        ConfigSpec(name="A_base", finetuned=False, rag_enabled=False),
        ConfigSpec(name="C_base_rag", finetuned=False, rag_enabled=True),
    ],
    "phase2_ft": [
        ConfigSpec(name="B_finetuned", finetuned=True, rag_enabled=False),
        ConfigSpec(name="D_finetuned_rag", finetuned=True, rag_enabled=True),
    ],
}


def _preflight(args, rag_cfg: dict, dataset_path: Path) -> None:
    """Hard preflight: any failure aborts before any sample runs."""
    logger.info("Preflight — validating llama.cpp model")
    validate_llamacpp_model(args.llm_endpoint, args.expected_model)

    logger.info("Preflight — pinging MLflow")
    setup_tracking_uri()
    import mlflow
    mlflow.search_experiments(max_results=1)

    logger.info("Preflight — checking Qdrant collection")
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", rag_cfg["vector_store"]["host"]),
        port=int(os.getenv("QDRANT_PORT", rag_cfg["vector_store"]["port"])),
    )
    collection = rag_cfg["vector_store"]["collection_name"]
    info = qdrant.get_collection(collection_name=collection)
    points = info.points_count or 0
    if points < 1000:
        raise RuntimeError(
            f"Qdrant collection {collection!r} has only {points} points — expected >>1k"
        )
    logger.info("Qdrant collection %s has %d points", collection, points)

    logger.info("Preflight — pinging Anthropic API")
    anthropic.Anthropic().messages.create(
        model=_JUDGE_MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "ping"}],
    )

    logger.info("Preflight — pinging OpenAI embeddings")
    OpenAI().embeddings.create(model=_EMBEDDING_MODEL, input="ping")

    if not Path(args.sample_indices).exists():
        raise RuntimeError(f"Sample indices file not found: {args.sample_indices}")
    if not dataset_path.exists():
        raise RuntimeError(f"Dataset directory not found: {dataset_path}")

    logger.info("Preflight passed")


def _build_pipeline(rag_cfg: dict, llm_endpoint: str) -> RAGPipeline:
    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", rag_cfg["vector_store"]["host"]),
        port=int(os.getenv("QDRANT_PORT", rag_cfg["vector_store"]["port"])),
    )
    dense_embedder = DenseEmbedder(rag_cfg["embedding"]["model"])
    sparse_embedder = SparseEmbedder()
    retriever = Retriever(
        client=qdrant,
        collection_name=rag_cfg["vector_store"]["collection_name"],
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )
    reranker = MedCPTReranker(
        batch_size=rag_cfg["retrieval"]["reranker"]["batch_size"],
        max_length=rag_cfg["retrieval"]["reranker"]["max_length"],
    )
    scorer = ClinicalScorer(weights=rag_cfg["retrieval"]["clinical_rerank"]["weights"])
    client = OpenAICompatibleClient(base_url=llm_endpoint)
    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        scorer=scorer,
        client=client,
        candidate_k=rag_cfg["retrieval"]["candidate_k"],
        final_k=rag_cfg["retrieval"]["final_k"],
        min_year=rag_cfg["retrieval"]["metadata_filters"]["min_year"],
        min_final_score=rag_cfg["retrieval"]["clinical_rerank"]["min_final_score"],
        max_tokens=rag_cfg["generation"]["max_tokens"],
        temperature=rag_cfg["generation"]["temperature"],
    )


def _sanity_warnings(spec: ConfigSpec, metrics: dict) -> None:
    if metrics.get("error_rate", 0.0) > 0.10:
        logger.warning(
            "SANITY — %s error_rate=%.2f > 10%%", spec.name, metrics["error_rate"]
        )
    if spec.name == "D_finetuned_rag" and metrics.get("hallucinated_pmids_rate", 0.0) > 0.05:
        logger.warning(
            "SANITY — D hallucinated_pmids_rate=%.2f > 5%% (canary for wrong model loaded)",
            metrics["hallucinated_pmids_rate"],
        )
    if spec.rag_enabled and metrics.get("abstention_rate", 0.0) > 0.50:
        logger.warning(
            "SANITY — %s abstention_rate=%.2f > 50%%",
            spec.name, metrics["abstention_rate"],
        )


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=list(_PHASE_CONFIGS))
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument(
        "--sample-indices", default="data/processed/eval_sample_indices.json"
    )
    parser.add_argument("--llm-endpoint", default="http://localhost:8001/v1")
    # TODO(eval-prerequisite): pass the exact string returned by
    # `curl http://localhost:8001/v1/models | jq '.data[0].id'` for each phase.
    #   - phase1_base: base GGUF id (Meta-Llama-3.1-8B-Instruct ...)
    #   - phase2_ft:   FT-merged GGUF id
    # The preflight aborts hard on mismatch (src/evaluation/server_check.py).
    parser.add_argument("--expected-model", required=True)
    args = parser.parse_args()

    rag_cfg = load_config("rag")
    dataset_path = Path("data/processed/dataset")

    _preflight(args, rag_cfg, dataset_path)

    samples = load_eval_sample(args.sample_indices, str(dataset_path))
    logger.info("Loaded %d eval samples", len(samples))

    pipeline = _build_pipeline(rag_cfg, args.llm_endpoint)

    out_dir = Path("data/processed/eval") / args.ablation_id / args.phase
    out_dir.mkdir(parents=True, exist_ok=True)

    common_params = {
        "ablation_id": args.ablation_id,
        "phase": args.phase,
        "llm_endpoint": args.llm_endpoint,
        "llm_model_name": args.expected_model,
        "judge_model": _JUDGE_MODEL,
        "embedding_model": _EMBEDDING_MODEL,
        "sample_size": len(samples),
        "abstention_threshold": pipeline.min_final_score,
        "min_relevant_docs": pipeline.min_relevant_docs,
        "candidate_k": pipeline.candidate_k,
        "final_k": pipeline.final_k,
        "temperature": pipeline._temperature,
        "max_tokens": pipeline._max_tokens,
    }

    per_config: list[tuple[ConfigSpec, dict, str]] = []
    for spec in _PHASE_CONFIGS[args.phase]:
        jsonl_path = out_dir / f"{spec.name}.jsonl"
        logger.info("=== Running config %s → %s ===", spec.name, jsonl_path)
        records = run_config(spec, samples, pipeline, str(jsonl_path))
        ragas_scores = compute_ragas_metrics(records)
        agg = aggregate_config(records, ragas_scores)
        _sanity_warnings(spec, agg)
        per_config.append((spec, agg, str(jsonl_path)))

    child_ids = log_phase_run(
        ablation_id=args.ablation_id,
        phase=args.phase,
        configs=per_config,
        common_params=common_params,
    )

    logger.info("Phase complete — ablation_id=%s", args.ablation_id)
    for name, run_id in child_ids.items():
        logger.info("  %s → run_id=%s", name, run_id)


if __name__ == "__main__":
    main()
