"""Training stage for finance returns/risk pipelines: trains Random Walk,
Mean, ARIMA (one fit per asset each) and one cross-sectional GBM (one fit
across the whole panel), logging each model's forecast-error standard
deviation to MLflow. A later plan implements this; this stub only pins the
call signature so src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def train_finance_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Train all configured finance models and log error-std metrics to MLflow.

    Args:
        features_dir: Directory containing train/test parquet files.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Dictionary with per-model MLflow run IDs and metrics.

    Raises:
        NotImplementedError: Always - implemented in a later plan.
    """
    raise NotImplementedError("train_finance_models is implemented in a later plan")
