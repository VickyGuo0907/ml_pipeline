"""Model evaluation and registration to MLflow.

Evaluation gate: each model is checked against thresholds from models.yaml
(evaluation.min_test_r2, evaluation.max_test_rmse) before registration.
Models that fail are skipped and recorded as 'rejected' in the evaluation
report. Models that pass are registered to MLflow Staging.

An evaluation YAML report is written to reports/<pipeline>/<run_id>_evaluation.yaml
regardless of outcome, providing a full audit trail of every decision made.
"""
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.utils.config import EvaluationConfig, load_models_config

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


def _check_thresholds(
    model_name: str,
    test_r2: float | None,
    test_rmse: float,
    cfg: EvaluationConfig,
) -> str | None:
    """Return a rejection reason string if any threshold is breached, else None.

    Args:
        model_name: Model identifier for log messages.
        test_r2: Test R² (may be None if not computed).
        test_rmse: Test RMSE.
        cfg: Evaluation thresholds from models.yaml.

    Returns:
        Human-readable rejection reason, or None if all thresholds pass.
    """
    if cfg.min_test_r2 is not None and test_r2 is not None:
        if test_r2 < cfg.min_test_r2:
            return (
                f"test_r2={test_r2:.4f} below min_test_r2={cfg.min_test_r2} "
                f"— model is worse than predicting the mean"
            )
    if cfg.max_test_rmse is not None:
        if test_rmse > cfg.max_test_rmse:
            return (
                f"test_rmse={test_rmse:.4f} above max_test_rmse={cfg.max_test_rmse}"
            )
    return None


def register_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
    config_dir: str | Path = "config",
    run_id: str = "unknown",
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Evaluate trained models against quality thresholds and register passing ones.

    Threshold gate: models failing min_test_r2 or max_test_rmse from models.yaml
    are skipped (not registered) and recorded as 'rejected' in the evaluation
    report. All decisions are written to reports/<pipeline>/<run_id>_evaluation.yaml.

    NO auto-promotion to Production — manual UI click only.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_run_ids: Dict mapping model names to MLflow run IDs.
        config_dir: Pipeline config directory (for evaluation thresholds).
        run_id: Airflow logical date, used for the report filename.
        reports_dir: Directory to write the evaluation YAML report.

    Returns:
        Dictionary with per-model evaluation decisions and registry info.

    Raises:
        ValueError: If no run IDs provided or a metric sanity check fails.
        RuntimeError: If any model fails to register due to an infrastructure error.
    """
    if not mlflow_run_ids:
        raise ValueError("mlflow_run_ids required for model registration")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    models_cfg = load_models_config(config_dir)
    eval_cfg = models_cfg.evaluation

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
    timestamp = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {
        "run_id": run_id,
        "evaluated_at": timestamp,
        "thresholds": {
            "min_test_r2": eval_cfg.min_test_r2,
            "max_test_rmse": eval_cfg.max_test_rmse,
        },
        "models": {},
    }
    registration_results: dict[str, Any] = {"registered_models": {}}
    infra_failures: list[str] = []

    for model_name, mlflow_run_id in mlflow_run_ids.items():
        try:
            run = mlflow.get_run(mlflow_run_id)
            metrics = run.data.metrics
            run_tags = run.data.tags

            test_rmse = metrics.get("test_rmse")
            train_rmse = metrics.get("train_rmse", 0.0)
            test_mse = metrics.get("test_mse", 0.0)
            test_r2 = metrics.get("test_r2")
            train_r2 = metrics.get("train_r2")
            model_type = run_tags.get("model_type", "unknown")
            pipeline_type = run_tags.get("pipeline_type", "unknown")
            pipeline_run_id = run_tags.get("run_id", run_id)

            # Hard sanity guards — non-finite metrics indicate a training failure.
            # NaN comparisons are always False, so an unguarded NaN test_r2 would
            # silently pass the threshold gate in _check_thresholds below.
            if test_rmse is None:
                raise ValueError(
                    f"test_rmse not found for {model_name} — did training complete?"
                )
            if not math.isfinite(test_rmse):
                raise ValueError(
                    f"test_rmse={test_rmse} is non-finite for {model_name}"
                )
            if test_r2 is not None and not math.isfinite(test_r2):
                raise ValueError(
                    f"test_r2={test_r2} is non-finite for {model_name}"
                )

            # Configurable threshold gate
            rejection_reason = _check_thresholds(model_name, test_r2, test_rmse, eval_cfg)
            if rejection_reason:
                logger.warning(
                    "REJECTED %s: %s — skipping registration", model_name, rejection_reason
                )
                report["models"][model_name] = {
                    "status": "rejected",
                    "reason": rejection_reason,
                    "test_r2": test_r2,
                    "train_r2": train_r2,
                    "test_rmse": test_rmse,
                    "train_rmse": train_rmse,
                }
                registration_results["registered_models"][model_name] = {
                    "status": "rejected",
                    "reason": rejection_reason,
                    "source_run_id": mlflow_run_id,
                }
                continue

            # Register passing model
            model_uri = f"runs:/{mlflow_run_id}/model"
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            version = registered_model.version

            # Tags on the registered model (top-level, visible in Models list)
            registered_model_tags = {
                "team": "data-eng",
                "project": "ml-pipeline",
                "pipeline_type": pipeline_type,
                "model_type": model_type,
            }
            for key, value in registered_model_tags.items():
                client.set_registered_model_tag(model_name, key, value)

            # Tags on this specific version (visible in version detail)
            version_tags: dict[str, str] = {
                "deployment": "staging",
                "registered_by": "pipeline",
                "registered_at": timestamp,
                "environment": "development",
                "pipeline_type": pipeline_type,
                "pipeline_run_id": pipeline_run_id,
                "model_type": model_type,
                "source_run_id": mlflow_run_id,
                "test_rmse": f"{test_rmse:.4f}",
                "train_rmse": f"{train_rmse:.4f}",
                "test_mse": f"{test_mse:.4f}",
            }
            if test_r2 is not None:
                version_tags["test_r2"] = f"{test_r2:.4f}"
            if train_r2 is not None:
                version_tags["train_r2"] = f"{train_r2:.4f}"

            _set_version_tags(client, model_name, version, version_tags)

            client.transition_model_version_stage(
                name=model_name, version=version, stage="Staging"
            )

            try:
                client.set_registered_model_alias(model_name, "staging", version)
            except Exception as alias_err:
                logger.warning(
                    "Could not set alias for %s v%s: %s", model_name, version, alias_err
                )

            r2_str = f"Test R²: {test_r2:.4f} | " if test_r2 is not None else ""
            client.update_registered_model(
                name=model_name,
                description=(
                    f"{model_name} ({model_type}) — "
                    f"{pipeline_type} pipeline model. "
                    f"Registered by dag_factory."
                ),
            )
            client.update_model_version(
                name=model_name,
                version=version,
                description=(
                    f"v{version} | Type: {model_type} | Pipeline: {pipeline_type} | "
                    f"Run: {pipeline_run_id} | "
                    f"{r2_str}"
                    f"Test RMSE: {test_rmse:.4f} | Train RMSE: {train_rmse:.4f} | "
                    f"Stage: Staging | Registered: {timestamp}"
                ),
            )

            logger.info(
                "REGISTERED %s v%s to Staging (test_r2=%s, test_rmse=%.4f)",
                model_name,
                version,
                f"{test_r2:.4f}" if test_r2 is not None else "n/a",
                test_rmse,
            )

            report["models"][model_name] = {
                "status": "registered",
                "version": version,
                "test_r2": test_r2,
                "train_r2": train_r2,
                "test_rmse": test_rmse,
                "train_rmse": train_rmse,
            }
            registration_results["registered_models"][model_name] = {
                "status": "registered",
                "version": version,
                "stage": "Staging",
                "source_run_id": mlflow_run_id,
                "pipeline_run_id": pipeline_run_id,
                "pipeline_type": pipeline_type,
                "model_uri": model_uri,
                "test_rmse": test_rmse,
                "train_rmse": train_rmse,
                "test_r2": test_r2,
                "train_r2": train_r2,
                "registered_at": timestamp,
            }

        except Exception as e:
            logger.error("Infrastructure error registering %s: %s", model_name, e)
            report["models"][model_name] = {"status": "error", "error": str(e)}
            registration_results["registered_models"][model_name] = {
                "status": "error",
                "error": str(e),
                "source_run_id": mlflow_run_id,
            }
            infra_failures.append(model_name)

    # Write evaluation report regardless of outcome
    _write_evaluation_report(report, run_id, reports_dir)

    registered = [k for k, v in report["models"].items() if v["status"] == "registered"]
    rejected = [k for k, v in report["models"].items() if v["status"] == "rejected"]
    logger.info(
        "Evaluation complete: %d registered, %d rejected, %d errors",
        len(registered), len(rejected), len(infra_failures),
    )

    if infra_failures:
        raise RuntimeError(
            f"Registration failed for {len(infra_failures)} model(s): "
            f"{', '.join(infra_failures)}. See logs for details."
        )

    # Zero registrations means every model failed the quality gate — fail the task
    # loudly instead of letting the DAG run go green with nothing deployable.
    if not registered:
        raise ValueError(
            f"All {len(rejected)} model(s) rejected by evaluation thresholds "
            f"({', '.join(rejected)}). Nothing registered to Staging. "
            f"See reports/<pipeline>/{run_id}_evaluation.yaml for reasons."
        )

    return registration_results


def _write_evaluation_report(
    report: dict[str, Any],
    run_id: str,
    reports_dir: str | Path,
) -> None:
    """Write evaluation YAML report to reports/<pipeline>/<run_id>_evaluation.yaml.

    Args:
        report: Evaluation results dict.
        run_id: Airflow logical date used for filename.
        reports_dir: Pipeline reports directory.
    """
    try:
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(reports_dir) / f"{run_id}_evaluation.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)
        logger.info("Evaluation report saved: %s", report_path)
    except Exception as e:
        logger.warning("Could not write evaluation report: %s", e)
