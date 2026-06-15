#!/usr/bin/env python3
"""
Import a raw MLflow file-store run directory into the tracking server.

The pod writes runs to a local file store (``file:///workspace/vol/mlruns``).
``mlflow-export-import`` expects an ``export-run`` bundle and is incompatible
with MLflow 3.x file-store reads, so this script reads ``meta.yaml`` plus the
``params/``, ``metrics/``, and ``tags/`` subdirectories directly and replays
them via the REST tracking API.

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5001 uv run python scripts/import_filestore_run.py \\
        --run-dir mlruns_pod/<experiment_id>/<run_id> \\
        --experiment-name tfm-finetuning
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import yaml
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

_DEFAULT_TRACKING_URI = "http://localhost:5001"
_METRICS_BATCH = 1000
_PARAMS_BATCH = 100


def _read_params(run_dir: Path) -> dict[str, str]:
    params_dir = run_dir / "params"
    if not params_dir.is_dir():
        return {}
    return {p.name: p.read_text().strip() for p in params_dir.iterdir() if p.is_file()}


def _read_tags(run_dir: Path) -> dict[str, str]:
    tags_dir = run_dir / "tags"
    if not tags_dir.is_dir():
        return {}
    return {t.name: t.read_text().strip() for t in tags_dir.iterdir() if t.is_file()}


def _read_metrics(run_dir: Path) -> dict[str, list[tuple[int, float, int]]]:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return {}
    metrics: dict[str, list[tuple[int, float, int]]] = {}
    for metric_file in metrics_dir.iterdir():
        if not metric_file.is_file():
            continue
        rows: list[tuple[int, float, int]] = []
        for line in metric_file.read_text().splitlines():
            if not line.strip():
                continue
            ts, value, step = line.split()
            rows.append((int(ts), float(value), int(step)))
        metrics[metric_file.name] = rows
    return metrics


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def import_filestore_run(
    run_dir: Path,
    experiment_name: str,
    tracking_uri: str,
    run_name: str | None = None,
) -> str:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")

    with meta_path.open() as f:
        meta = yaml.safe_load(f)

    params = _read_params(run_dir)
    tags = _read_tags(run_dir)
    metrics = _read_metrics(run_dir)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    resolved_run_name = run_name or meta.get("run_name") or meta["run_id"]
    run = client.create_run(
        experiment_id,
        run_name=resolved_run_name,
        tags={"imported_from": str(run_dir), "source_run_id": meta["run_id"]},
    )
    run_id = run.info.run_id

    try:
        param_items = list(params.items())
        for batch in _batched(param_items, _PARAMS_BATCH):
            client.log_batch(
                run_id,
                params=[mlflow.entities.Param(k, v) for k, v in batch],
            )

        metric_entities = [
            mlflow.entities.Metric(key, value, ts, step)
            for key, rows in metrics.items()
            for ts, value, step in rows
        ]
        for batch in _batched(metric_entities, _METRICS_BATCH):
            client.log_batch(run_id, metrics=batch)

        tag_items = [(k, v) for k, v in tags.items() if not k.startswith("mlflow.")]
        for batch in _batched(tag_items, _PARAMS_BATCH):
            client.log_batch(
                run_id,
                tags=[mlflow.entities.RunTag(k, v) for k, v in batch],
            )

        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
    except Exception:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to <experiment_id>/<run_id> in a file-store tree",
    )
    parser.add_argument(
        "--experiment-name",
        default="tfm-finetuning",
        help="Destination experiment name on the tracking server",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI),
        help="Destination MLflow tracking URI (default: MLFLOW_TRACKING_URI env)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the imported run name",
    )
    args = parser.parse_args()

    run_id = import_filestore_run(
        run_dir=args.run_dir.resolve(),
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=args.run_name,
    )
    print(f"Imported run {run_id} into experiment '{args.experiment_name}'")
    print(f"View at {args.tracking_uri}")


if __name__ == "__main__":
    main()
