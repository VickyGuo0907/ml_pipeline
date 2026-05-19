"""Model training stage with MLflow autologging."""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.utils.config import load_models_config, load_pipeline_config

logger = logging.getLogger(__name__)


def train_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Train models (linear + LightGBM) and log metrics to MLflow.

    Args:
        features_dir: Directory containing feature matrices
        run_id: Run identifier
        config_dir: Configuration directory
        mlflow_tracking_uri: MLflow tracking server URI

    Returns:
        Dictionary with model run IDs and metrics

    Raises:
        FileNotFoundError: If feature files don't exist
    """
    features_path = Path(features_dir) / run_id
    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    # Load configs
    pipeline_config = load_pipeline_config(config_dir)
    models_config = load_models_config(config_dir)

    # Load feature data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = pipeline_config.target.name
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    training_results = {
        "run_id": run_id,
        "models": {},
    }

    # Train each model
    for model_config in models_config.models:
        model_name = model_config.name
        model_type = model_config.type

        # Create and train model
        if model_type == "linear":
            model = Ridge(**model_config.hyperparameters)
        elif model_type == "gbm":
            model = LGBMRegressor(**model_config.hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        with mlflow.start_run(run_name=f"{run_id}_{model_name}"):
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("run_id", run_id)

            # Train model
            model.fit(X_train, y_train)

            # Get predictions and log metrics
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_mse = ((train_pred - y_train) ** 2).mean()
            test_mse = ((test_pred - y_test) ** 2).mean()
            train_rmse = train_mse ** 0.5
            test_rmse = test_mse ** 0.5

            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)

            # Log model - handle gracefully if API endpoint not available
            try:
                if model_type == "linear":
                    mlflow.sklearn.log_model(model, artifact_path="model")
                elif model_type == "gbm":
                    mlflow.lightgbm.log_model(model, artifact_path="model")
            except Exception as model_log_error:
                logger.warning(f"Could not log {model_name} to MLflow: {model_log_error}")

            run_id_mlflow = mlflow.active_run().info.run_id

        training_results["models"][model_name] = {
            "mlflow_run_id": run_id_mlflow,
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "feature_count": X_train.shape[1],
        }

    return training_results
