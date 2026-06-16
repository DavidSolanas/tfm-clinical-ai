from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.evaluation.aggregation import CITATION_METRIC_KEYS, compute_citation_metrics
from src.evaluation.mlflow_logging import _fetch_children
from src.logging_config import get_logger
from src.tracking import setup_tracking_uri

logger = get_logger(__name__)

_EVAL_ROOT = Path("data/processed/eval")


def _jsonl_path(ablation_id: str, phase: str, config_name: str) -> Path:
    return _EVAL_ROOT / ablation_id / phase / f"{config_name}.jsonl"


def _load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print metrics but do not write to MLflow.",
    )
    args = parser.parse_args()

    setup_tracking_uri()
    from mlflow.tracking import MlflowClient

    children = _fetch_children(MlflowClient(), args.ablation_id)
    if not children:
        raise RuntimeError(f"No child runs found for ablation_id={args.ablation_id}")

    logger.info("Found %d child runs: %s", len(children), sorted(children))

    for config_name, info in sorted(children.items()):
        phase = info["params"].get("phase")
        if phase is None:
            logger.warning("Skipping %s — no 'phase' param on run", config_name)
            continue
        path = _jsonl_path(args.ablation_id, phase, config_name)
        if not path.exists():
            logger.warning("Skipping %s — JSONL not found at %s", config_name, path)
            continue

        records = _load_records(path)
        metrics = compute_citation_metrics(records)
        # MLflow rejects NaN/Inf; drop them (e.g. grounding precision for no-RAG).
        loggable = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf"))
        }
        dropped = sorted(set(CITATION_METRIC_KEYS) - set(loggable))

        logger.info(
            "%s (%s, %d records) → %s%s",
            config_name,
            phase,
            len(records),
            {k: round(v, 4) for k, v in loggable.items()},
            f" | dropped (NaN): {dropped}" if dropped else "",
        )

        if args.dry_run:
            continue

        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metrics(loggable)
        logger.info("  ✓ logged to run_id=%s", info["run_id"])

    logger.info("Backfill complete for ablation_id=%s", args.ablation_id)


if __name__ == "__main__":
    main()
