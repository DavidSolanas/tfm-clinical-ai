"""Thin MLflow helpers shared across training and evaluation modules.

Usage::

    from src.tracking import mlflow_run

    with mlflow_run("my-experiment", run_name="my-run", tags={"stage": "data"}):
        mlflow.log_params({...})
        mlflow.log_metrics({...})

The context manager starts a new run if no run is active, or joins the existing
active run (useful when calling library functions from a notebook that already
started a run). The tracking URI is resolved from the ``MLFLOW_TRACKING_URI``
environment variable, defaulting to ``http://localhost:5001`` (the host port
the MLflow container publishes in ``docker-compose.yml``: ``5001:5000``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import mlflow

from src.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_TRACKING_URI = "http://localhost:5001"


def setup_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


@contextmanager
def mlflow_run(
    experiment: str,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Start an MLflow run, or join the active one if already inside a run.

    When joined (not started), the caller does not own lifecycle — ``end_run``
    is NOT called on exit so the outer owner can close it.

    Args:
        experiment: MLflow experiment name. Created automatically if absent.
        run_name: Human-readable name shown in the MLflow UI.
        tags: Optional key-value tags attached to the run.

    Yields:
        The active ``mlflow.ActiveRun`` object.
    """
    setup_tracking_uri()
    mlflow.set_experiment(experiment)

    _started = mlflow.active_run() is None
    if _started:
        logger.debug("Starting MLflow run '%s' in experiment '%s'", run_name, experiment)
        mlflow.start_run(run_name=run_name, tags=tags or {})
    else:
        logger.debug("Joining existing MLflow run for experiment '%s'", experiment)

    try:
        yield mlflow.active_run()
    finally:
        if _started:
            mlflow.end_run()
            logger.debug("MLflow run '%s' ended", run_name)
