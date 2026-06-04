"""RAGAS and custom metric computation over runner JSONL records.

Locked decisions (plans/06_evaluation_implementation_plan.md §1):
- Judge LLM: Claude Sonnet 4.6 via anthropic SDK.
- Embeddings: OpenAI ``text-embedding-3-small``.
- Abstained samples excluded from RAGAS; A/B (no contexts) get only answer_relevancy.
- Judge errors retry 3× with exponential backoff, then NaN for that metric.
- ContextRecall reference = ground_truth (teacher response).
"""

from __future__ import annotations

import math
import re
import time

import anthropic
from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.logging_config import get_logger

logger = get_logger(__name__)

_FORMAT_RE = re.compile(r"^\s*1\..+?\n.*?2\..+?\n.*?3\.", re.DOTALL)
_NO_EVIDENCE_RE = re.compile(
    r"^the retrieved context is off-topic or insufficient", re.IGNORECASE
)

_FAITHFULNESS = "faithfulness"
_ANSWER_RELEVANCY = "answer_relevancy"
_CONTEXT_PRECISION = "context_precision"
_CONTEXT_RECALL = "context_recall"

_ALL_METRICS = (_FAITHFULNESS, _ANSWER_RELEVANCY, _CONTEXT_PRECISION, _CONTEXT_RECALL)
_NO_RAG_METRICS = (_ANSWER_RELEVANCY,)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0


def check_format_adherence(response: str | None) -> bool | None:
    """True if response matches the 3-section structure or the no-evidence template.

    Returns None when the response itself is None (abstained or errored).
    """
    if response is None:
        return None
    if _NO_EVIDENCE_RE.search(response.strip()):
        return True
    return bool(_FORMAT_RE.search(response))


def check_hallucinated_pmids(record: dict) -> int:
    """Number of hallucinated PMIDs in a runner record."""
    return len(record.get("hallucinated_pmids") or [])


def _build_judge() -> tuple[object, object]:
    """Construct the RAGAS LLM and embedding wrappers."""
    anthropic_client = anthropic.Anthropic()
    llm = llm_factory("claude-sonnet-4-6", provider="anthropic", client=anthropic_client)
    openai_client = OpenAI()
    embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", client=openai_client
    )
    return llm, embeddings


def _build_metric_instances(judge_llm, judge_embeddings) -> dict[str, object]:
    return {
        _FAITHFULNESS: Faithfulness(llm=judge_llm),
        _ANSWER_RELEVANCY: AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        _CONTEXT_PRECISION: ContextPrecisionWithReference(llm=judge_llm),
        _CONTEXT_RECALL: ContextRecall(llm=judge_llm),
    }


def _record_to_sample(record: dict) -> SingleTurnSample:
    return SingleTurnSample(
        user_input=record["transcription"],
        response=record["response"] or "",
        retrieved_contexts=list(record.get("retrieved_contexts") or []),
        reference=record["ground_truth_response"],
    )


def _is_eligible(record: dict) -> bool:
    return record.get("response") is not None and not record.get("error")


def _evaluate_subset(
    records: list[dict],
    metric_names: tuple[str, ...],
    metric_instances: dict[str, object],
) -> dict[int, dict[str, float]]:
    """Run RAGAS over a subset; return per-sample scores keyed by sample_idx."""
    if not records:
        return {}
    samples = [_record_to_sample(r) for r in records]
    dataset = EvaluationDataset(samples=samples)
    metrics = [metric_instances[m] for m in metric_names]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(timeout=120),
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()
    out: dict[int, dict[str, float]] = {}
    for record, (_, row) in zip(records, df.iterrows(), strict=True):
        out[int(record["sample_idx"])] = {
            m: float(row[m]) if m in row and row[m] is not None else math.nan
            for m in metric_names
        }
    return out


def _retry_nans(
    per_sample: dict[int, dict[str, float]],
    records_by_idx: dict[int, dict],
    metric_names: tuple[str, ...],
    metric_instances: dict[str, object],
) -> None:
    """Re-run RAGAS individually for samples with any NaN, up to 3 attempts each."""
    needs_retry = [
        idx for idx, scores in per_sample.items()
        if any(math.isnan(scores[m]) for m in metric_names if m in scores)
    ]
    for idx in needs_retry:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                retry_scores = _evaluate_subset(
                    [records_by_idx[idx]], metric_names, metric_instances
                )
                still_nan = any(
                    math.isnan(retry_scores[idx][m]) for m in metric_names
                )
                if not still_nan:
                    per_sample[idx] = retry_scores[idx]
                    logger.info("Retry succeeded for sample %d on attempt %d", idx, attempt)
                    break
                logger.warning(
                    "Retry %d/%d for sample %d still produced NaN", attempt, _MAX_RETRIES, idx
                )
            except Exception as exc:  # noqa: BLE001 — must survive judge transients
                logger.warning(
                    "Retry %d/%d for sample %d raised: %s", attempt, _MAX_RETRIES, idx, exc
                )
            time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))


def compute_ragas_metrics(records: list[dict]) -> dict[int, dict[str, float]]:
    """Run RAGAS over the eligible records and return per-sample metric scores.

    Eligibility:
        - ``response`` is non-null AND ``error`` is null.
        - ``retrieved_contexts`` non-empty → all four metrics.
        - ``retrieved_contexts`` empty (configs A/B) → answer_relevancy only.

    Args:
        records: Runner records as produced by :func:`src.evaluation.runner.run_config`.

    Returns:
        Mapping ``sample_idx -> {metric: score}`` where missing/failed metrics are
        :data:`math.nan`.
    """
    judge_llm, judge_embeddings = _build_judge()
    metric_instances = _build_metric_instances(judge_llm, judge_embeddings)

    rag_records = [r for r in records if _is_eligible(r) and r.get("retrieved_contexts")]
    no_rag_records = [
        r for r in records if _is_eligible(r) and not r.get("retrieved_contexts")
    ]
    logger.info(
        "RAGAS scoring — full=%d, answer_relevancy_only=%d, skipped=%d",
        len(rag_records),
        len(no_rag_records),
        len(records) - len(rag_records) - len(no_rag_records),
    )

    per_sample: dict[int, dict[str, float]] = {}
    records_by_idx = {int(r["sample_idx"]): r for r in records}

    if rag_records:
        per_sample.update(_evaluate_subset(rag_records, _ALL_METRICS, metric_instances))
        _retry_nans(per_sample, records_by_idx, _ALL_METRICS, metric_instances)
    if no_rag_records:
        per_sample.update(
            _evaluate_subset(no_rag_records, _NO_RAG_METRICS, metric_instances)
        )
        _retry_nans(per_sample, records_by_idx, _NO_RAG_METRICS, metric_instances)

    return per_sample
