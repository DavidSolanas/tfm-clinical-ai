"""Cross-config rollup for one ablation (run after both phases complete).

Usage::

    uv run python scripts/run_evaluation_rollup.py --ablation-id 2026-05-21T1430

Queries MLflow for the 2 phase parents + 4 children sharing the given
``ablation_id``, then logs a rollup run with headline deltas and bar charts.
Plots are written to MLflow artifacts and ``docs/thesis_latex/figures/eval_*.png``.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.evaluation.mlflow_logging import log_rollup_run
from src.logging_config import get_logger
from src.tracking import setup_tracking_uri

logger = get_logger(__name__)

_EXPERIMENT = "tfm-evaluation"


def _assert_runs_present(ablation_id: str) -> None:
    setup_tracking_uri()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {_EXPERIMENT!r} not found")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.ablation_id = '{ablation_id}'",
    )
    parents = [r for r in runs if r.data.tags.get("kind") == "phase_parent"]
    children = [r for r in runs if r.data.tags.get("kind") == "config_child"]
    logger.info(
        "Found %d phase parents and %d config children for ablation_id=%s",
        len(parents), len(children), ablation_id,
    )
    if len(parents) != 2:
        raise RuntimeError(
            f"Expected 2 phase parents, found {len(parents)}. Run both phases first."
        )
    if len(children) != 4:
        raise RuntimeError(
            f"Expected 4 config children, found {len(children)}. Run both phases first."
        )


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-id", required=True)
    args = parser.parse_args()

    _assert_runs_present(args.ablation_id)
    run_id = log_rollup_run(args.ablation_id)
    logger.info("Rollup complete: ablation_id=%s run_id=%s", args.ablation_id, run_id)


if __name__ == "__main__":
    main()
