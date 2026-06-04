"""Aggregate per-sample records and RAGAS scores into one config-level summary.

Bootstrap 95% CIs use 1000 resamples with a fixed seed (42) for reproducibility.
Abstained samples are excluded from RAGAS means (denominator = answered) but
count as 0 hallucinations in ``hallucinated_pmids_rate`` per locked decision #8.
"""

from __future__ import annotations

import math

import numpy as np

from src.evaluation.metrics import check_format_adherence, check_hallucinated_pmids

_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 42


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
    out["hallucinated_pmids_rate"] = (
        sum(halluc_counts) / n_total if n_total else math.nan
    )
    out["hallucinated_pmids_per_sample_mean"] = (
        float(np.mean(halluc_counts)) if halluc_counts else math.nan
    )

    format_results = [
        check_format_adherence(r.get("response"))
        for r in records
        if not r.get("error") and not r.get("abstained")
    ]
    out["format_adherence_among_answered"] = (
        float(sum(1 for v in format_results if v) / n_answered) if n_answered else math.nan
    )

    return out
