"""Evaluation/registration stage for time-series forecasting pipelines:
rolling-origin test scoring, champion selection, and MLflow registration.
Plan 4 implements this; this stub only pins the call signature so
src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def register_forecast_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
    config_dir: str | Path = "config",
    run_id: str = "unknown",
    reports_dir: str | Path = "reports",
    features_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate trained forecasting models via rolling-origin scoring and register.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_run_ids: Per-model MLflow run IDs from the train stage.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).
        run_id: Run identifier.
        reports_dir: Directory for the evaluation audit report.
        features_dir: Directory containing train/test parquet files.
        benchmark_dir: Fixed benchmark directory (unused until benchmark.enabled).

    Returns:
        Dictionary with registration results.

    Raises:
        NotImplementedError: Always - implemented in Plan 4.
    """
    raise NotImplementedError("register_forecast_models_to_mlflow is implemented in Plan 4")
