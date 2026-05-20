"""Build the SFT dataset for fine-tuning the clinical RAG model.

Loads MTSamples clinical notes, retrieves PubMed evidence via the RAG pipeline,
and generates structured teacher responses using a configurable LLM. The resulting
HuggingFace DatasetDict (train / validation / test) is saved to disk.

Prerequisite: Qdrant must be running and the corpus must already be indexed
(see ``scripts/index_corpus.py``).

Usage::

    uv run python scripts/build_dataset.py [options]

Options:
    --output PATH             Dataset output directory             (default: data/processed/dataset)
    --cache PATH              JSON cache for generated examples    (default: data/processed/dataset_cache.json)
    --mtsamples PATH          MTSamples CSV path or HF Hub ID     (default: HF Hub)
    --collection NAME         Qdrant collection name              (default: from configs/rag.yaml)
    --min-class-size INT      Drop specialties below this record count (default: 10)
    --min-doc-score FLOAT     Min relevance score to include doc  (default: 0.40)
    --max-tokens INT          Max tokens for teacher responses    (default: 1024)
    --teacher anthropic|local LLM backend for teacher generation  (default: anthropic)
    --teacher-model STR       Anthropic model name               (default: claude-sonnet-4-6)
    --teacher-url STR         Base URL for local OpenAI endpoint  (default: http://localhost:8001/v1)
    --cache-flush-every INT   Flush cache every N new examples   (default: 25)
    --seed INT                Random seed                         (default: 42)
    --mlflow-experiment STR   MLflow experiment name             (default: tfm-dataset-building)
    --max-no-evidence-ratio FLOAT  Max fraction of no-evidence responses (default: 0.15)
    --use-thinking            Enable Gemma-4 thinking mode in teacher responses (default: False)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.config import load_config
from src.data.mtsamples import load_mtsamples
from src.llm_clients import AnthropicClient, OpenAICompatibleClient
from src.logging_config import get_logger
from src.rag.embedder import DenseEmbedder, SparseEmbedder
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import MedCPTReranker
from src.rag.retriever import Retriever
from src.rag.scorer import ClinicalScorer
from src.training.dataset_builder import build_dataset

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the SFT dataset for clinical fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dataset"),
        help="Directory where the HuggingFace DatasetDict will be saved",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("data/processed/dataset_cache.json"),
        help="JSON file used to cache generated examples (enables resume on failure)",
    )
    p.add_argument(
        "--mtsamples",
        type=str,
        default="hf://datasets/harishnair04/mtsamples/mtsamples.csv",
        help="MTSamples CSV path or HuggingFace Hub dataset identifier",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: vector_store.collection_name from rag.yaml)",
    )
    p.add_argument(
        "--min-class-size",
        type=int,
        default=10,
        help="Drop medical specialties with fewer than this many records before splitting",
    )
    p.add_argument(
        "--min-doc-score",
        type=float,
        default=0.40,
        help="Minimum relevance score for a retrieved document to be included",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in each teacher LLM response",
    )
    p.add_argument(
        "--teacher",
        choices=["anthropic", "local"],
        default="anthropic",
        help="LLM backend used to generate teacher responses",
    )
    p.add_argument(
        "--teacher-model",
        type=str,
        default="claude-sonnet-4-6",
        help="Anthropic model name (only used when --teacher=anthropic)",
    )
    p.add_argument(
        "--teacher-url",
        type=str,
        default="http://localhost:8001/v1",
        help="Base URL of an OpenAI-compatible local endpoint (only used when --teacher=local)",
    )
    p.add_argument(
        "--cache-flush-every",
        type=int,
        default=25,
        help="Flush the cache to disk after every N newly generated examples",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting and shuffling",
    )
    p.add_argument(
        "--mlflow-experiment",
        type=str,
        default="tfm-dataset-building",
        help="MLflow experiment name where this run will be recorded",
    )
    p.add_argument(
        "--max-no-evidence-ratio",
        type=float,
        default=0.15,
        help="Maximum fraction of examples allowed to have a no-relevant-evidence response (0–1)",
    )
    p.add_argument(
        "--use-thinking",
        action="store_true",
        default=False,
        help="Prepend <|think|> to the teacher system prompt to enable Gemma-4 thinking mode",
    )
    return p


def main() -> None:
    load_dotenv(override=True)

    args = _build_parser().parse_args()
    cfg = load_config("rag")

    collection_name: str = args.collection or cfg["vector_store"]["collection_name"]

    logger.info("Loading MTSamples from %s", args.mtsamples)
    df = load_mtsamples(args.mtsamples)
    logger.info("MTSamples loaded — %d records", len(df))

    logger.info("Initialising dense embedder: %s", cfg["embedding"]["model"])
    dense_embedder = DenseEmbedder(cfg["embedding"]["model"])

    sparse_embedder = SparseEmbedder()

    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", cfg["vector_store"]["host"]),
        port=int(os.getenv("QDRANT_PORT", cfg["vector_store"]["port"])),
    )

    retriever = Retriever(
        client=qdrant,
        collection_name=collection_name,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    reranker = MedCPTReranker(
        batch_size=cfg["retrieval"]["reranker"]["batch_size"],
        max_length=cfg["retrieval"]["reranker"]["max_length"],
    )

    scorer = ClinicalScorer(
        weights=cfg["retrieval"]["clinical_rerank"]["weights"],
    )

    if args.teacher == "anthropic":
        teacher_client = AnthropicClient(
            client=anthropic.Anthropic(),
            model=args.teacher_model,
        )
        teacher_model_id = args.teacher_model
        logger.info("Teacher LLM: Anthropic / %s", args.teacher_model)
    else:
        teacher_client = OpenAICompatibleClient(base_url=args.teacher_url)
        teacher_model_id = f"local@{args.teacher_url}"
        logger.info("Teacher LLM: local endpoint at %s", args.teacher_url)

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        scorer=scorer,
        client=teacher_client,
        system_prompt=cfg["generation"]["system_prompt"],
        candidate_k=cfg["retrieval"]["candidate_k"],
        final_k=cfg["retrieval"]["final_k"],
        min_year=cfg["retrieval"]["metadata_filters"]["min_year"],
        min_final_score=cfg["retrieval"]["clinical_rerank"]["min_final_score"],
    )
    
    dataset = build_dataset(
        df=df,
        pipeline=pipeline,
        client=teacher_client,
        min_doc_score=args.min_doc_score,
        max_tokens=args.max_tokens,
        cache_path=args.cache,
        cache_flush_every=args.cache_flush_every,
        seed=args.seed,
        min_class_size=args.min_class_size,
        max_no_evidence_ratio=args.max_no_evidence_ratio,
        use_thinking=args.use_thinking,
        output_path=args.output,
        mlflow_experiment=args.mlflow_experiment,
        teacher_model=teacher_model_id,
    )

    logger.info(
        "Done — train: %d, validation: %d, test: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )


if __name__ == "__main__":
    main()
