"""RAGAS and custom metric computation over runner JSONL records.

Locked decisions (plans/06_evaluation_implementation_plan.md §1):
- Judge LLM: Claude Sonnet 4.6 via anthropic SDK.
- Embeddings: OpenAI ``text-embedding-3-small``.
- Abstained samples excluded from RAGAS; A/B (no contexts) get only answer_relevancy.
- Judge errors retry 3× with exponential backoff, then NaN for that metric.
- ContextRecall reference = ground_truth (teacher response).
"""

from __future__ import annotations

import asyncio
import math
import re
import time

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from src.logging_config import get_logger

logger = get_logger(__name__)

_FORMAT_RE = re.compile(r"^\s*1\..+?\n.*?2\..+?\n.*?3\.", re.DOTALL)
_NO_EVIDENCE_RE = re.compile(r"^the retrieved context is off-topic or insufficient", re.IGNORECASE)

# Deterministic (judge-free) citation-style detectors. These characterise the
# citation behaviour the fine-tuning teaches — and that RAGAS does not measure:
# the trained format is the parenthesised ``(PMID: <digits>)`` handle, base
# models tend to use ``[n]`` bracket refs, and a model with nothing real to
# cite often regurgitates the prompt's literal ``(PMID: XXXXXXXX)`` placeholder.
_PAREN_PMID_RE = re.compile(r"\(PMID:\s*(\d{4,9})\)", re.IGNORECASE)
_BRACKET_REF_RE = re.compile(r"\[(\d{1,3})\]")
_PLACEHOLDER_PMID_RE = re.compile(r"PMID:\s*[X#]{2,}", re.IGNORECASE)

_FAITHFULNESS = "faithfulness"
_ANSWER_RELEVANCY = "answer_relevancy"
_CONTEXT_PRECISION = "context_precision"
_CONTEXT_RECALL = "context_recall"

_ALL_METRICS = (_FAITHFULNESS, _ANSWER_RELEVANCY, _CONTEXT_PRECISION, _CONTEXT_RECALL)
_NO_RAG_METRICS = (_ANSWER_RELEVANCY,)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0

# Bound concurrent judge calls so 50 samples × 4 metrics don't all hit the
# Anthropic/OpenAI APIs at once (mirrors the legacy RunConfig worker cap).
_MAX_CONCURRENCY = 16


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


def extract_pmid_citations(response: str | None) -> list[str]:
    """Real ``(PMID: <digits>)`` citations in the trained parenthesised format.

    Returns the list (not set) so callers can count repeated citations.
    """
    if not response:
        return []
    return _PAREN_PMID_RE.findall(response)


def has_bracket_reference(response: str | None) -> bool:
    """True if the response uses base-model-style ``[n]`` reference handles."""
    return bool(response) and _BRACKET_REF_RE.search(response) is not None


def has_placeholder_citation(response: str | None) -> bool:
    """True if the response emits an unfilled citation placeholder, e.g. ``(PMID: XXXXXXXX)``."""
    return bool(response) and _PLACEHOLDER_PMID_RE.search(response) is not None


def _build_judge() -> tuple[object, object]:
    """Construct the RAGAS LLM and embedding wrappers.

    Async clients are required because the collections metrics score via
    ``ascore``/``agenerate``; a sync client raises "Cannot use agenerate() with
    a synchronous client".
    """
    anthropic_client = AsyncAnthropic()
    llm = llm_factory("claude-sonnet-4-6", provider="anthropic", client=anthropic_client)
    # The instructor factory always injects both temperature and top_p, but
    # Claude Sonnet 4.6 rejects requests that set both. Drop top_p and keep the
    # deterministic temperature (0.01); Anthropic params pass through unchanged.
    if hasattr(llm, "model_args") and isinstance(llm.model_args, dict):
        llm.model_args.pop("top_p", None)
        llm.model_args["max_tokens"] = 16384
    openai_client = AsyncOpenAI()
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
    return llm, embeddings


def _build_metric_instances(judge_llm, judge_embeddings) -> dict[str, object]:
    return {
        _FAITHFULNESS: Faithfulness(llm=judge_llm),
        _ANSWER_RELEVANCY: AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        _CONTEXT_PRECISION: ContextPrecisionWithReference(llm=judge_llm),
        _CONTEXT_RECALL: ContextRecall(llm=judge_llm),
    }


def _metric_inputs(metric_name: str, record: dict) -> dict[str, object]:
    """Map a runner record to the keyword args a collections metric's ascore expects."""
    user_input = record["transcription"]
    response = record["response"] or ""
    contexts = list(record.get("retrieved_contexts") or [])
    reference = record["ground_truth_response"]
    if metric_name == _FAITHFULNESS:
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": contexts,
        }
    if metric_name == _ANSWER_RELEVANCY:
        return {"user_input": user_input, "response": response}
    if metric_name == _CONTEXT_PRECISION:
        return {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": contexts,
        }
    if metric_name == _CONTEXT_RECALL:
        return {
            "user_input": user_input,
            "retrieved_contexts": contexts,
            "reference": reference,
        }
    raise ValueError(f"Unknown metric: {metric_name}")


def _is_eligible(record: dict) -> bool:
    return record.get("response") is not None and not record.get("error")


async def _score_subset_async(
    records: list[dict],
    metric_names: tuple[str, ...],
    metric_instances: dict[str, object],
) -> dict[int, dict[str, float]]:
    """Score every (record, metric) pair via the collections ascore API.

    A failing judge call yields ``math.nan`` for that single metric rather than
    aborting the whole batch, preserving the NaN-then-retry semantics.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    total_tasks = len(records) * len(metric_names)
    n_done = [0]  # mutable cell for closure counter
    t_start = [time.monotonic()]

    async def score_one(metric_name: str, record: dict) -> float:
        metric = metric_instances[metric_name]
        sample_idx = record.get("sample_idx")
        async with semaphore:
            try:
                result = await metric.ascore(**_metric_inputs(metric_name, record))
            except Exception as exc:  # noqa: BLE001 — judge transients become NaN
                logger.warning(
                    "Metric %s failed for sample %s: %s",
                    metric_name,
                    sample_idx,
                    exc,
                )
                n_done[0] += 1
                logger.info(
                    "RAGAS [%d/%d] sample=%s metric=%s → NaN (judge error)",
                    n_done[0], total_tasks, sample_idx, metric_name,
                )
                return math.nan
            value = result.value
            score = float(value) if value is not None else math.nan
            n_done[0] += 1
            elapsed = time.monotonic() - t_start[0]
            remaining = total_tasks - n_done[0]
            eta_s = (elapsed / n_done[0]) * remaining if n_done[0] else 0
            logger.info(
                "RAGAS [%d/%d] sample=%s metric=%s → %.3f | elapsed=%.0fs ETA=%.0fs",
                n_done[0], total_tasks, sample_idx, metric_name,
                score, elapsed, eta_s,
            )
            return score

    coros = [
        (int(record["sample_idx"]), metric_name, score_one(metric_name, record))
        for record in records
        for metric_name in metric_names
    ]
    scores = await asyncio.gather(*(c for _, _, c in coros))

    out: dict[int, dict[str, float]] = {}
    for (idx, metric_name, _), score in zip(coros, scores, strict=True):
        out.setdefault(idx, {})[metric_name] = score
    return out


def _evaluate_subset(
    records: list[dict],
    metric_names: tuple[str, ...],
    metric_instances: dict[str, object],
) -> dict[int, dict[str, float]]:
    """Run RAGAS over a subset; return per-sample scores keyed by sample_idx."""
    if not records:
        return {}
    return asyncio.run(_score_subset_async(records, metric_names, metric_instances))


def _retry_nans(
    per_sample: dict[int, dict[str, float]],
    records_by_idx: dict[int, dict],
    metric_names: tuple[str, ...],
    metric_instances: dict[str, object],
) -> None:
    """Re-run RAGAS individually for samples with any NaN, up to 3 attempts each."""
    needs_retry = [
        idx
        for idx, scores in per_sample.items()
        if any(math.isnan(scores[m]) for m in metric_names if m in scores)
    ]
    for idx in needs_retry:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                retry_scores = _evaluate_subset(
                    [records_by_idx[idx]], metric_names, metric_instances
                )
                still_nan = any(math.isnan(retry_scores[idx][m]) for m in metric_names)
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
    no_rag_records = [r for r in records if _is_eligible(r) and not r.get("retrieved_contexts")]
    logger.info(
        "RAGAS scoring — full=%d, answer_relevancy_only=%d, skipped=%d",
        len(rag_records),
        len(no_rag_records),
        len(records) - len(rag_records) - len(no_rag_records),
    )

    per_sample: dict[int, dict[str, float]] = {}
    records_by_idx = {int(r["sample_idx"]): r for r in records}

    t0 = time.time()
    if rag_records:
        logger.info(
            "RAGAS scoring — starting full-metric pass (%d records × %d metrics = %d tasks)",
            len(rag_records), len(_ALL_METRICS), len(rag_records) * len(_ALL_METRICS),
        )
        per_sample.update(_evaluate_subset(rag_records, _ALL_METRICS, metric_instances))
        _retry_nans(per_sample, records_by_idx, _ALL_METRICS, metric_instances)
    if no_rag_records:
        logger.info(
            "RAGAS scoring — starting answer_relevancy-only pass"
            " (%d records × 1 metric = %d tasks)",
            len(no_rag_records), len(no_rag_records),
        )
        per_sample.update(_evaluate_subset(no_rag_records, _NO_RAG_METRICS, metric_instances))
        _retry_nans(per_sample, records_by_idx, _NO_RAG_METRICS, metric_instances)

    logger.info(
        "RAGAS scoring complete — %.0fs total, %d samples scored",
        time.time() - t0, len(per_sample),
    )
    return per_sample
