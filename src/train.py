"""Model training stage with MLflow autologging."""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import r2_score

from src.utils.config import load_models_config, load_pipeline_config
from src.utils.model_registry import get_model

logger = logging.getLogger(__name__)


def _log_metrics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute and log regression metrics for both splits.

    Args:
        model: Trained estimator.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict of metric name → value.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_mse": float(((train_pred - y_train) ** 2).mean()),
        "test_mse": float(((test_pred - y_test) ** 2).mean()),
        "train_rmse": float(((train_pred - y_train) ** 2).mean() ** 0.5),
        "test_rmse": float(((test_pred - y_test) ** 2).mean() ** 0.5),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
    }
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    return metrics


def train_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Train all configured models and log metrics + artifacts to MLflow.

    Model types are resolved via the model registry — no if/elif branches.
    Adding a new model type requires only a models.yaml entry and a registry
    line in src/utils/model_registry.py.

    Args:
        features_dir: Directory containing train/test parquet files.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/biomedical_clinical).
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Dictionary with per-model MLflow run IDs and metrics.

    Raises:
        FileNotFoundError: If feature files don't exist.
    """
    features_path = Path(features_dir) / run_id
    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    pipeline_config = load_pipeline_config(config_dir)
    models_config = load_models_config(config_dir)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = pipeline_config.target.name
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    training_results: dict[str, Any] = {"run_id": run_id, "models": {}}

    for model_cfg in models_config.models:
        model = get_model(model_cfg.type, model_cfg.hyperparameters)

        with mlflow.start_run(run_name=f"{run_id}_{model_cfg.name}"):
            mlflow.set_tags({
                "model_name": model_cfg.name,
                "model_type": model_cfg.type,
                "run_id": run_id,
                "pipeline_type": pipeline_config.pipeline_type,
            })

            model.fit(X_train, y_train)
            metrics = _log_metrics(model, X_train, y_train, X_test, y_test)

            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                logger.warning("Could not log %s to MLflow: %s", model_cfg.name, e)

            mlflow_run_id = mlflow.active_run().info.run_id

        logger.info(
            "Trained %s: test_r2=%.4f test_rmse=%.4f",
            model_cfg.name, metrics["test_r2"], metrics["test_rmse"],
        )

        training_results["models"][model_cfg.name] = {
            "mlflow_run_id": mlflow_run_id,
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "feature_count": X_train.shape[1],
        }

    return training_results
