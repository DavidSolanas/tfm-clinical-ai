"""Aggregate per-sample records and RAGAS scores into one config-level summary.

Bootstrap 95% CIs use 1000 resamples with a fixed seed (42) for reproducibility.
Abstained samples are excluded from RAGAS means (denominator = answered) but
count as 0 hallucinations in ``hallucinated_pmids_rate`` per locked decision #8.
"""

from __future__ import annotations

import math

import numpy as np

from src.evaluation.metrics import (
    check_format_adherence,
    check_hallucinated_pmids,
    extract_pmid_citations,
    has_bracket_reference,
    has_placeholder_citation,
)

_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 42

# Keys produced by ``compute_citation_metrics`` — re-logged by the backfill
# script so it stays in sync with native runs.
CITATION_METRIC_KEYS = (
    "pmid_citation_rate",
    "bracket_citation_rate",
    "placeholder_citation_rate",
    "no_citation_rate",
    "pmid_citations_per_answered_mean",
    "citation_grounding_precision",
    "hallucinated_pmids_rate",
)


def _answered(records: list[dict]) -> list[dict]:
    """Records that produced a real answer (not abstained, not errored)."""
    return [r for r in records if not r.get("error") and not r.get("abstained")]


def compute_citation_metrics(records: list[dict]) -> dict[str, float]:
    """Deterministic (judge-free) citation-behaviour metrics for one config.

    All rates are over *answered* records (abstained/errored excluded). These
    capture the citation axis RAGAS does not: format adherence of citations,
    placeholder regurgitation, and how well cited PMIDs are grounded in what was
    actually retrieved. Recomputable from JSONL alone, so they can be backfilled
    onto existing MLflow runs without re-paying judge calls.

    Args:
        records: Runner records for one configuration.

    Returns:
        Flat ``{metric: value}`` dict; rates are NaN when no answered records,
        and ``citation_grounding_precision`` is NaN when no real PMID is cited
        from any retrieval-bearing record (e.g. the no-RAG configs A/B).
    """
    answered = _answered(records)
    n_answered = len(answered)
    n_total = len(records)

    pmid_counts = [len(extract_pmid_citations(r.get("response"))) for r in answered]
    n_with_pmid = sum(1 for c in pmid_counts if c > 0)
    n_bracket = sum(1 for r in answered if has_bracket_reference(r.get("response")))
    n_placeholder = sum(1 for r in answered if has_placeholder_citation(r.get("response")))
    n_no_citation = sum(
        1
        for r, c in zip(answered, pmid_counts, strict=True)
        if c == 0
        and not has_bracket_reference(r.get("response"))
        and not has_placeholder_citation(r.get("response"))
    )

    # Grounding precision: of the real PMIDs cited across records that actually
    # carried retrieval, the fraction present in that record's retrieved set.
    grounded = total_cited = 0
    for r in answered:
        retrieved = {str(p) for p in (r.get("retrieved_pmids") or [])}
        if not retrieved:
            continue  # no-RAG: grounding is undefined, not zero
        for pmid in extract_pmid_citations(r.get("response")):
            total_cited += 1
            if pmid in retrieved:
                grounded += 1

    # Proper "rate": fraction of records with >=1 hallucinated PMID (the old
    # value duplicated the per-sample mean; abstained count as 0 hallucinations).
    n_with_halluc = sum(1 for r in records if check_hallucinated_pmids(r) > 0)

    return {
        "pmid_citation_rate": (n_with_pmid / n_answered) if n_answered else math.nan,
        "bracket_citation_rate": (n_bracket / n_answered) if n_answered else math.nan,
        "placeholder_citation_rate": (n_placeholder / n_answered) if n_answered else math.nan,
        "no_citation_rate": (n_no_citation / n_answered) if n_answered else math.nan,
        "pmid_citations_per_answered_mean": (
            float(np.mean(pmid_counts)) if pmid_counts else math.nan
        ),
        "citation_grounding_precision": (grounded / total_cited) if total_cited else math.nan,
        "hallucinated_pmids_rate": (n_with_halluc / n_total) if n_total else math.nan,
    }


def _bootstrap_ci(values: list[float], seed: int = _BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% CI of the mean via percentile bootstrap; NaN if no values."""
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(_BOOTSTRAP_RESAMPLES)
    n = len(arr)
    for i in range(_BOOTSTRAP_RESAMPLES):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(np.mean(sample))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _ragas_stats(
    metric: str,
    ragas_per_sample: dict[int, dict[str, float]],
) -> dict[str, float]:
    values = [
        scores[metric]
        for scores in ragas_per_sample.values()
        if metric in scores and not math.isnan(scores[metric])
    ]
    if not values:
        return {
            f"{metric}_mean": math.nan,
            f"{metric}_n": 0,
            f"{metric}_ci_low": math.nan,
            f"{metric}_ci_high": math.nan,
        }
    ci_low, ci_high = _bootstrap_ci(values)
    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_n": len(values),
        f"{metric}_ci_low": ci_low,
        f"{metric}_ci_high": ci_high,
    }


def aggregate_config(
    records: list[dict],
    ragas_per_sample: dict[int, dict[str, float]],
) -> dict:
    """Aggregate runner output for one configuration.

    Args:
        records: All runner records for this configuration (one per eval sample).
        ragas_per_sample: Output of :func:`src.evaluation.metrics.compute_ragas_metrics`.

    Returns:
        Flat dict suitable for logging as MLflow metrics. See plan §2.5 for the
        full key list.
    """
    n_total = len(records)
    n_abstained = sum(1 for r in records if r.get("abstained"))
    n_errors = sum(1 for r in records if r.get("error"))
    n_answered = n_total - n_abstained - n_errors

    out: dict[str, float] = {
        "n_total": n_total,
        "n_abstained": n_abstained,
        "n_errors": n_errors,
        "abstention_rate": (n_abstained / n_total) if n_total else math.nan,
        "error_rate": (n_errors / n_total) if n_total else math.nan,
        "wall_time_total_s": float(sum(r.get("wall_time_s") or 0.0 for r in records)),
    }

    for metric in _RAGAS_METRICS:
        out.update(_ragas_stats(metric, ragas_per_sample))

    halluc_counts = [check_hallucinated_pmids(r) for r in records]
    # hallucinated_pmids_rate (fraction of answers with >=1 hallucination) is
    # computed in compute_citation_metrics; keep the per-sample mean here.
    out["hallucinated_pmids_per_sample_mean"] = (
        float(np.mean(halluc_counts)) if halluc_counts else math.nan
    )
    out.update(compute_citation_metrics(records))

    format_results = [
        check_format_adherence(r.get("response"))
        for r in records
        if not r.get("error") and not r.get("abstained")
    ]
    out["format_adherence_among_answered"] = (
        float(sum(1 for v in format_results if v) / n_answered) if n_answered else math.nan
    )

    return out
