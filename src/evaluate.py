"""Model evaluation and registration to MLflow."""
import logging
from datetime import datetime, timezone
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def _set_version_tags(
    client: mlflow.tracking.MlflowClient,
    name: str,
    version: str,
    tags: dict[str, str],
) -> None:
    """Set tags on a specific model version (not the registered model)."""
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)


def register_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Register trained models to MLflow Staging with tags and descriptions.

    Sets tags on both the registered model (top-level) and each model version.
    NO auto-promotion to Production — manual UI click only.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_run_ids: Dict mapping model names to MLflow run IDs

    Returns:
        Dictionary with model registry information

    Raises:
        ValueError: If no run IDs provided
    """
    if not mlflow_run_ids:
        raise ValueError("mlflow_run_ids required for model registration")

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    registration_results = {
        "registered_models": {},
    }

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
    timestamp = datetime.now(timezone.utc).isoformat()

    for model_name, run_id in mlflow_run_ids.items():
        try:
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            run_tags = run.data.tags

            test_rmse = metrics.get("test_rmse", 0)
            train_rmse = metrics.get("train_rmse", 0)
            test_mse = metrics.get("test_mse", 0)
            model_type = run_tags.get("model_type", "unknown")

            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

            version = registered_model.version

            # Tags on the registered model (top-level, visible in Models list)
            registered_model_tags = {
                "team": "data-eng",
                "project": "ml_pipeline",
                "model_type": model_type,
            }
            for key, value in registered_model_tags.items():
                client.set_registered_model_tag(model_name, key, value)

            # Tags on this specific version (visible in version detail)
            version_tags = {
                "deployment": "staging",
                "registered_by": "pipeline",
                "registered_at": timestamp,
                "environment": "development",
                "test_rmse": f"{test_rmse:.4f}",
                "train_rmse": f"{train_rmse:.4f}",
                "test_mse": f"{test_mse:.4f}",
                "model_type": model_type,
                "source_run_id": run_id,
                "status": "active",
            }
            _set_version_tags(client, model_name, version, version_tags)

            # Transition to Staging stage
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
            )

            # Set alias for easier lookup
            try:
                client.set_registered_model_alias(
                    model_name, "staging", version
                )
            except Exception as alias_err:
                logger.warning("Could not set alias for %s v%s: %s", model_name, version, alias_err)

            # Registered model description (top-level)
            client.update_registered_model(
                name=model_name,
                description=(
                    f"{model_name} ({model_type}) — "
                    f"Hospital readmission prediction model. "
                    f"Registered by ml_pipeline DAG."
                ),
            )

            # Version description (per-version detail)
            client.update_model_version(
                name=model_name,
                version=version,
                description=(
                    f"v{version} | Type: {model_type} | "
                    f"Test RMSE: {test_rmse:.4f} | "
                    f"Train RMSE: {train_rmse:.4f} | "
                    f"Stage: Staging | "
                    f"Registered: {timestamp}"
                ),
            )

            logger.info(
                "Registered %s v%s to Staging "
                "(test_rmse=%.4f, train_rmse=%.4f)",
                model_name,
                version,
                test_rmse,
                train_rmse,
            )

            registration_results["registered_models"][model_name] = {
                "version": version,
                "stage": "Staging",
                "source_run_id": run_id,
                "model_uri": model_uri,
                "test_rmse": test_rmse,
                "train_rmse": train_rmse,
                "version_tags": version_tags,
                "registered_model_tags": registered_model_tags,
                "registered_at": timestamp,
            }
        except Exception as e:
            logger.error("Failed to register %s: %s", model_name, e)
            registration_results["registered_models"][model_name] = {
                "error": str(e),
                "source_run_id": run_id,
            }

    return registration_results
