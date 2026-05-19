"""Model evaluation and registration to MLflow."""
import logging
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def register_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Register trained models to MLflow Staging.

    NO auto-promotion to Production—manual UI click only.

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

    # Register each model to MLflow Model Registry
    for model_name, run_id in mlflow_run_ids.items():
        try:
            # Load run
            run = mlflow.get_run(run_id)

            # Register model from run
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={"stage": "staging", "registered_by": "pipeline"},
            )

            # Move to Staging (not Production)
            client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Staging",
            )

            registration_results["registered_models"][model_name] = {
                "version": registered_model.version,
                "stage": "Staging",
                "source_run_id": run_id,
                "model_uri": model_uri,
            }
        except Exception as e:
            logger.error(f"Failed to register {model_name}: {e}")
            registration_results["registered_models"][model_name] = {
                "error": str(e),
                "source_run_id": run_id,
            }

    return registration_results
