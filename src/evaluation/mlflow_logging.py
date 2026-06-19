"""MLflow logging for the ablation study.

One phase parent run per phase (two phases total), with one nested child per
ablation configuration. The rollup is logged separately by
``scripts/run_evaluation_rollup.py``. All runs share ``tags.ablation_id`` so the
rollup can find them via :meth:`MlflowClient.search_runs`.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from src.evaluation.plot_style import apply_style, grouped_bar, metric_title
from src.evaluation.runner import ConfigSpec
from src.logging_config import get_logger
from src.tracking import setup_tracking_uri

logger = get_logger(__name__)

_EXPERIMENT = "tfm-evaluation"

# Configuration plot order. Keeping A→B→C→D makes A/B (no-RAG) and C/D (RAG)
# pair up visually for the FT contrast in the thesis figures.
_CONFIG_ORDER = ("A_base", "B_finetuned", "C_base_rag", "D_finetuned_rag")

_FIGURES_DIR = Path("docs/thesis_latex/figures")


def _clean_metrics(metrics: dict) -> dict[str, float]:
    """MLflow rejects NaN/Inf; convert them to 0.0 and log non-numeric keys separately."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            out[k] = float(v)
    return out


def log_phase_run(
    ablation_id: str,
    phase: str,
    configs: list[tuple[ConfigSpec, dict, str]],
    common_params: dict,
) -> dict[str, str]:
    """Log one phase parent + per-config nested children.

    Args:
        ablation_id: Shared identifier across all runs in this ablation.
        phase: ``"phase1_base"`` or ``"phase2_ft"``.
        configs: ``[(spec, aggregated_metrics, jsonl_path), ...]`` — one per
            child configuration evaluated in this phase.
        common_params: Parameters shared across children (judge_model,
            embedding_model, llm_endpoint, expected_model, sample_size,
            abstention_threshold, temperature, max_tokens, ...).

    Returns:
        Mapping ``{config_name: child_run_id}``.
    """
    setup_tracking_uri()
    mlflow.set_experiment(_EXPERIMENT)

    child_ids: dict[str, str] = {}
    parent_tags = {"ablation_id": ablation_id, "phase": phase, "kind": "phase_parent"}
    with mlflow.start_run(run_name=f"{ablation_id}_{phase}", tags=parent_tags) as parent:
        logger.info("Phase parent run started — id=%s", parent.info.run_id)
        mlflow.log_params(common_params)

        for spec, metrics, jsonl_path in configs:
            child_tags = {
                "ablation_id": ablation_id,
                "phase": phase,
                "config": spec.name,
                "kind": "config_child",
            }
            with mlflow.start_run(
                run_name=spec.name, nested=True, tags=child_tags
            ) as child:
                mlflow.log_params({
                    **common_params,
                    "config_name": spec.name,
                    "finetuned": spec.finetuned,
                    "rag_enabled": spec.rag_enabled,
                })
                mlflow.log_metrics(_clean_metrics(metrics))
                mlflow.log_artifact(jsonl_path, artifact_path="records")
                child_ids[spec.name] = child.info.run_id
                logger.info(
                    "Logged child run %s — id=%s", spec.name, child.info.run_id
                )

    return child_ids


def _bar_with_ci(
    ax,
    config_names: list[str],
    means: list[float],
    ci_lows: list[float] | None,
    ci_highs: list[float] | None,
    title: str,
    ylabel: str,
) -> None:
    """Thesis-styled per-config bar chart with optional 95% CI error bars.

    Thin wrapper over :func:`src.evaluation.plot_style.grouped_bar` so the rollup
    figures match the inspection notebook exactly.
    """
    grouped_bar(
        ax, config_names, means,
        ylabel=ylabel, title=title,
        ci_low=ci_lows, ci_high=ci_highs,
    )


def _fetch_children(client: MlflowClient, ablation_id: str) -> dict[str, dict]:
    """Return ``{config_name: {param/metric dict}}`` for all child runs of this ablation."""
    experiment = client.get_experiment_by_name(_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {_EXPERIMENT!r} not found")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.ablation_id = '{ablation_id}' and tags.kind = 'config_child'",
    )
    by_name: dict[str, dict] = {}
    for run in runs:
        name = run.data.tags.get("config")
        if name is None:
            continue
        by_name[name] = {
            "run_id": run.info.run_id,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }
    return by_name


def _save_and_log(fig, name: str) -> None:
    """Save plot to docs/thesis_latex/figures and log as MLflow artifact."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = _FIGURES_DIR / f"eval_{name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    mlflow.log_figure(fig, f"plots/eval_{name}.png")
    plt.close(fig)
    logger.info("Saved plot %s", fig_path)


def log_rollup_run(ablation_id: str) -> str:
    """Log a rollup run aggregating both phases of an ablation.

    Queries MLflow for the four child runs sharing this ``ablation_id``, computes
    headline deltas, generates grayscale-printable bar charts, and writes them
    to both MLflow artifacts and ``docs/thesis_latex/figures/eval_*.png``.

    Args:
        ablation_id: Identifier shared by the two phase parents + four children.

    Returns:
        The rollup run id.

    Raises:
        RuntimeError: If fewer than four child configs are found.
    """
    setup_tracking_uri()
    mlflow.set_experiment(_EXPERIMENT)
    apply_style()
    client = MlflowClient()

    children = _fetch_children(client, ablation_id)
    missing = [c for c in _CONFIG_ORDER if c not in children]
    if missing:
        raise RuntimeError(
            f"Rollup requires all 4 configs; missing: {missing} for ablation_id={ablation_id}"
        )

    rollup_tags = {"ablation_id": ablation_id, "kind": "rollup"}
    with mlflow.start_run(run_name=f"{ablation_id}_rollup", tags=rollup_tags) as rollup:
        m = {name: children[name]["metrics"] for name in _CONFIG_ORDER}

        deltas = {
            "D_minus_B_hallucinated_pmids_rate": (
                m["D_finetuned_rag"].get("hallucinated_pmids_rate", math.nan)
                - m["B_finetuned"].get("hallucinated_pmids_rate", math.nan)
            ),
            "D_minus_C_faithfulness_mean": (
                m["D_finetuned_rag"].get("faithfulness_mean", math.nan)
                - m["C_base_rag"].get("faithfulness_mean", math.nan)
            ),
            "D_minus_B_format_adherence": (
                m["D_finetuned_rag"].get("format_adherence_among_answered", math.nan)
                - m["B_finetuned"].get("format_adherence_among_answered", math.nan)
            ),
            "D_minus_C_abstention_rate": (
                m["D_finetuned_rag"].get("abstention_rate", math.nan)
                - m["C_base_rag"].get("abstention_rate", math.nan)
            ),
            "D_minus_C_citation_grounding_precision": (
                m["D_finetuned_rag"].get("citation_grounding_precision", math.nan)
                - m["C_base_rag"].get("citation_grounding_precision", math.nan)
            ),
        }
        mlflow.log_metrics({k: v for k, v in deltas.items() if not math.isnan(v)})
        mlflow.log_param("child_run_ids", {n: children[n]["run_id"] for n in _CONFIG_ORDER})

        # Custom metrics bar charts (no CIs — these are rates/proportions).
        for metric_key, ylabel, plot_name in [
            ("hallucinated_pmids_rate", "Fraction of answers", "hallucinated_pmids"),
            ("format_adherence_among_answered", "Fraction of answered", "format_adherence"),
            ("abstention_rate", "Fraction of samples", "abstention_rate"),
            ("error_rate", "Fraction of samples", "error_rate"),
            ("pmid_citation_rate", "Fraction of answers", "pmid_citation_rate"),
            ("placeholder_citation_rate", "Fraction of answers", "placeholder_cite"),
            ("citation_grounding_precision", "Cited PMIDs in retrieved set", "grounding_precision"),
        ]:
            means = [m[c].get(metric_key, math.nan) for c in _CONFIG_ORDER]
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            _bar_with_ci(ax, list(_CONFIG_ORDER), means, None, None,
                         title=metric_title(metric_key), ylabel=ylabel)
            _save_and_log(fig, plot_name)

        # RAGAS metrics with 95% CI error bars.
        for metric_key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            means = [m[c].get(f"{metric_key}_mean", math.nan) for c in _CONFIG_ORDER]
            lows = [m[c].get(f"{metric_key}_ci_low", math.nan) for c in _CONFIG_ORDER]
            highs = [m[c].get(f"{metric_key}_ci_high", math.nan) for c in _CONFIG_ORDER]
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            _bar_with_ci(ax, list(_CONFIG_ORDER), means, lows, highs,
                         title=f"RAGAS · {metric_title(metric_key)}",
                         ylabel="Score")
            _save_and_log(fig, metric_key)

        logger.info("Rollup run logged — id=%s", rollup.info.run_id)
        return rollup.info.run_id
