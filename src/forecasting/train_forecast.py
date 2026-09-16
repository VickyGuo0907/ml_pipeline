"""Training stage for time-series forecasting pipelines: trains ETS,
SARIMAX, and GBM-lagged models with rolling-origin cross-validation. Plan 3
implements this; this stub only pins the call signature so
src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def train_forecast_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Train all configured forecasting models and log metrics to MLflow.

    Args:
        features_dir: Directory containing train/test parquet files.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Dictionary with per-model MLflow run IDs and metrics.

    Raises:
        NotImplementedError: Always - implemented in Plan 3.
    """
    raise NotImplementedError("train_forecast_models is implemented in Plan 3")
