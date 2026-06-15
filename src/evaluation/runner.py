"""Per-config evaluation runner.

Iterates over the frozen eval sample, executes the four ablation configurations
through the shared :class:`RAGPipeline` instance, and writes one JSONL record
per sample for resumability. Per-sample exceptions are caught and recorded;
preflight checks happen upstream in the phase script.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.prompts import build_no_rag_user_message
from src.logging_config import get_logger
from src.rag.pipeline import RAGPipeline
from src.rag.scorer import verify_citations

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConfigSpec:
    """Identifies one of the four ablation configurations."""

    name: str
    finetuned: bool
    rag_enabled: bool


def _load_completed_indices(out_path: Path) -> set[int]:
    """Read existing JSONL and return the set of sample_idx values already written."""
    if not out_path.exists():
        return set()
    completed: set[int] = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", out_path)
                continue
            completed.add(int(rec["sample_idx"]))
    return completed


def _run_rag_sample(pipeline: RAGPipeline, transcription: str) -> dict:
    """Execute the full RAG pipeline and shape its output into the record schema."""
    result = pipeline.run(transcription)
    if result["abstained"]:
        return {
            "response": None,
            "abstained": True,
            "retrieved_pmids": [],
            "retrieved_contexts": [],
            "cited_pmids": [],
            "hallucinated_pmids": [],
        }
    docs = result["docs"]
    citation_check = result["citation_check"] or {"cited_pmids": [], "hallucinated_pmids": []}
    return {
        "response": result["response"],
        "abstained": False,
        "retrieved_pmids": [str(d["pmid"]) for d in docs],
        "retrieved_contexts": [str(d["abstract"]) for d in docs],
        "cited_pmids": list(citation_check["cited_pmids"]),
        "hallucinated_pmids": list(citation_check["hallucinated_pmids"]),
    }


def _run_no_rag_sample(pipeline: RAGPipeline, transcription: str) -> dict:
    """Call the LLM client directly with the no-evidence training template (configs A/B)."""
    response = pipeline._client.generate(
        system=pipeline._system,
        user=build_no_rag_user_message(transcription),
        max_tokens=pipeline._max_tokens,
        temperature=pipeline._temperature,
    )
    # No retrieved set → every cited PMID is by definition hallucinated.
    citation_check = verify_citations(response, retrieved_pmids=[])
    return {
        "response": response,
        "abstained": False,
        "retrieved_pmids": [],
        "retrieved_contexts": [],
        "cited_pmids": list(citation_check["cited_pmids"]),
        "hallucinated_pmids": list(citation_check["hallucinated_pmids"]),
    }


def _build_error_record(sample: dict, config_name: str, exc: BaseException, wall_time_s: float) -> dict:
    return {
        "sample_idx": int(sample["idx"]),
        "config": config_name,
        "medical_specialty": sample["medical_specialty"],
        "transcription": sample["transcription"],
        "ground_truth_response": sample["ground_truth_response"],
        "response": None,
        "abstained": False,
        "retrieved_pmids": [],
        "retrieved_contexts": [],
        "cited_pmids": [],
        "hallucinated_pmids": [],
        "wall_time_s": wall_time_s,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_config(
    config: ConfigSpec,
    samples: list[dict],
    pipeline: RAGPipeline,
    out_path: str,
) -> list[dict]:
    """Run one ablation configuration over the eval sample, writing JSONL per-sample.

    Resume semantics: if ``out_path`` already contains records, sample indices
    present in the file are skipped (the loaded record is included in the
    returned list so the caller sees the full set).

    Args:
        config: Which ablation configuration is being evaluated.
        samples: Eval records from :func:`src.evaluation.sample.load_eval_sample`.
        pipeline: Shared RAG pipeline; provides retrieval, LLM client, and
            generation parameters (``_temperature`` / ``_max_tokens`` are the
            single source of truth across all four configs).
        out_path: JSONL destination. Existing file is appended to.

    Returns:
        All records (both pre-existing and newly written), ordered by input
        sample iteration order.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_indices(out)
    if completed:
        logger.info(
            "Resuming %s — %d/%d samples already completed in %s",
            config.name, len(completed), len(samples), out,
        )

    # Re-read existing records so the caller has them on return.
    existing: dict[int, dict] = {}
    if out.exists():
        with open(out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    existing[int(rec["sample_idx"])] = rec
                except json.JSONDecodeError:
                    continue

    results: list[dict] = []
    t_phase_start = time.perf_counter()
    n_new = 0
    with open(out, "a") as f:
        for sample in samples:
            idx = int(sample["idx"])
            if idx in completed:
                results.append(existing[idx])
                continue

            t0 = time.perf_counter()
            try:
                if config.rag_enabled:
                    payload = _run_rag_sample(pipeline, sample["transcription"])
                else:
                    payload = _run_no_rag_sample(pipeline, sample["transcription"])
                wall = time.perf_counter() - t0
                record = {
                    "sample_idx": idx,
                    "config": config.name,
                    "medical_specialty": sample["medical_specialty"],
                    "transcription": sample["transcription"],
                    "ground_truth_response": sample["ground_truth_response"],
                    **payload,
                    "wall_time_s": round(wall, 3),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 — runner must survive per-sample failures
                wall = time.perf_counter() - t0
                logger.exception("Sample %d (%s) failed: %s", idx, config.name, exc)
                record = _build_error_record(sample, config.name, exc, round(wall, 3))

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            results.append(record)

            n_new += 1
            done_total = len(completed) + n_new
            remaining = len(samples) - done_total
            elapsed = time.perf_counter() - t_phase_start
            avg_s = elapsed / n_new
            eta_s = avg_s * remaining
            if record.get("error"):
                status = "error"
            elif record.get("abstained"):
                status = "abstained"
            else:
                status = "ok"
            logger.info(
                "[%s] %d/%d (%.0f%%) | %s | wall=%.1fs | avg=%.1fs | ETA=%s",
                config.name,
                done_total,
                len(samples),
                100.0 * done_total / len(samples),
                status,
                record["wall_time_s"],
                avg_s,
                f"{eta_s:.0f}s" if remaining > 0 else "done",
            )

    logger.info(
        "Finished %s — wrote %d new records (%d total, %d errors, %d abstained)",
        config.name,
        len(samples) - len(completed),
        len(results),
        sum(1 for r in results if r.get("error")),
        sum(1 for r in results if r.get("abstained")),
    )
    return results
