"""Evaluation/registration stage for finance returns/risk pipelines:
cross-sectional rank-correlation (Information Coefficient) scoring per
model type, champion selection, and MLflow registration. A later plan
implements this; this stub only pins the call signature so
src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def register_finance_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
    config_dir: str | Path = "config",
    run_id: str = "unknown",
    reports_dir: str | Path = "reports",
    features_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score each trained finance model via cross-sectional rank correlation and register.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_run_ids: Per-(model_type, asset) MLflow run IDs from the train stage.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).
        run_id: Run identifier.
        reports_dir: Directory for the evaluation audit report.
        features_dir: Directory containing train/test parquet files.
        benchmark_dir: Fixed benchmark directory (unused - this pipeline has no benchmark design).

    Returns:
        Dictionary with registration results.

    Raises:
        NotImplementedError: Always - implemented in a later plan.
    """
    raise NotImplementedError("register_finance_models_to_mlflow is implemented in a later plan")
