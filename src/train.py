"""Model training stage with MLflow autologging."""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import r2_score

from src.benchmark import bootstrap_metric_ci
from src.utils.config import load_models_config, load_pipeline_config
from src.utils.diagnostics import cross_validate_model, residual_diagnostics
from src.utils.io import resolve_run_path
from src.utils.model_registry import get_model

logger = logging.getLogger(__name__)

# Maps config `type` strings (models.yaml) to the MLflow flavor used to log that model.
# LightGBM's Booster/LGBMRegressor aren't on skops's default trusted-type list, so
# mlflow.sklearn.log_model() silently fails to save gbm models (caught below, logged
# as a warning, but no artifact is ever written — the model becomes unloadable even
# though registration and training otherwise succeed). Types not listed here fall
# back to mlflow.sklearn, which covers every other entry in model_registry.py.
_MLFLOW_LOG_MODEL_FNS: dict[str, Any] = {
    "gbm": mlflow.lightgbm.log_model,
}


def _log_model(model: Any, model_type: str) -> Any:
    """Log a trained model to MLflow using the flavor appropriate to its type.

    Args:
        model: Fitted estimator.
        model_type: Registry type string from models.yaml (e.g. "gbm", "ridge").

    Returns:
        The MLflow ModelInfo for the logged model.
    """
    log_fn = _MLFLOW_LOG_MODEL_FNS.get(model_type, mlflow.sklearn.log_model)
    return log_fn(model, artifact_path="model")


def _log_metrics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.Series]:
    """Compute and log regression metrics for both splits, plus bootstrapped 95%
    confidence intervals on the held-out test metrics.

    Point estimates alone (test_rmse, test_r2) don't convey how much they'd move
    on a different random train/test split — the CIs bound that. Computed only
    on the test set, since train-set CIs would just describe in-sample fit noise.

    Args:
        model: Trained estimator.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (metrics dict, test predictions Series).
        Test predictions are returned so the caller can pass them to
        mlflow.evaluate() without a second predict() call.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    def _rmse(yt, yp):
        return float(((yt - yp) ** 2).mean() ** 0.5)

    test_rmse_ci = bootstrap_metric_ci(y_test, test_pred, _rmse)
    test_r2_ci = bootstrap_metric_ci(y_test, test_pred, r2_score)

    train_mse = float(((train_pred - y_train) ** 2).mean())
    test_mse = float(((test_pred - y_test) ** 2).mean())

    metrics = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_rmse": train_mse ** 0.5,
        "test_rmse": test_mse ** 0.5,
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_rmse_ci_lower": test_rmse_ci[0],
        "test_rmse_ci_upper": test_rmse_ci[1],
        "test_r2_ci_lower": test_r2_ci[0],
        "test_r2_ci_upper": test_r2_ci[1],
    }
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    return metrics, pd.Series(test_pred, index=X_test.index)


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
    features_path = resolve_run_path(features_dir, run_id)
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

    # Read Box-Cox lambda from features manifest so serve.py can inverse-transform predictions
    boxcox_lambda: float | None = None
    manifest_path = features_path / "manifest.yaml"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        boxcox_lambda = manifest.get("transform_meta", {}).get("boxcox_lambda")

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

            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("target_col", target_col)
            # Exact trained column names + order, so serve.py can build its request
            # schema and column ordering dynamically per model — no pipeline-specific
            # schema needs to be hardcoded into the serving layer. Logged as an
            # artifact (not a param) since some pipelines' feature counts exceed a
            # param value's length limit.
            mlflow.log_dict({"columns": list(X_train.columns)}, "feature_columns.json")
            if boxcox_lambda is not None:
                mlflow.log_param("boxcox_lambda", boxcox_lambda)
            # Cross-validation runs on the training set only, before the final fit,
            # so the held-out test set is never touched during model comparison.
            cv_summary: dict[str, Any] | None = None
            cv_cfg = models_config.cross_validation
            if cv_cfg.enabled:
                try:
                    cv_summary = cross_validate_model(
                        model,
                        X_train,
                        y_train,
                        folds=cv_cfg.folds,
                        group_column=cv_cfg.group_column,
                        random_state=models_config.random_state,
                    )
                    mlflow.log_param("cv_strategy", cv_summary["strategy"])
                    mlflow.log_param("cv_folds", cv_summary["folds"])
                    for key in ("cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std"):
                        mlflow.log_metric(key, cv_summary[key])
                    # Per-fold scores as individual metrics: MLflow metrics are scalars,
                    # and having the folds means the spread behind cv_r2_std is auditable
                    # from the evaluation report rather than taken on trust.
                    for i, fold_score in enumerate(cv_summary["cv_r2_folds"], start=1):
                        mlflow.log_metric(f"cv_r2_fold_{i}", fold_score)
                    logger.info(
                        "CV (%s) for %s: R2=%.4f +/- %.4f",
                        cv_summary["strategy"], model_cfg.name,
                        cv_summary["cv_r2_mean"], cv_summary["cv_r2_std"],
                    )
                except Exception as e:
                    logger.warning("Cross-validation skipped for %s: %s", model_cfg.name, e)

            model.fit(X_train, y_train)
            metrics, test_pred = _log_metrics(model, X_train, y_train, X_test, y_test)

            # Residual assumption tests, only for linear families — tree ensembles
            # make no distributional assumptions about their errors, so these
            # statistics would not be interpretable for them.
            diag: dict[str, float] = {}
            diag_cfg = models_config.diagnostics
            if diag_cfg.enabled and model_cfg.type in diag_cfg.linear_types:
                diag = residual_diagnostics(y_test, test_pred.to_numpy())
                for key, value in diag.items():
                    mlflow.log_metric(key, value)
                if diag:
                    logger.info(
                        "Diagnostics for %s: DW=%.3f BP_p=%s",
                        model_cfg.name, diag.get("durbin_watson", float("nan")),
                        diag.get("breusch_pagan_p"),
                    )

            model_id: str | None = None
            try:
                model_info = _log_model(model, model_cfg.type)
                model_id = model_info.model_id
            except Exception as e:
                logger.warning("Could not log %s to MLflow: %s", model_cfg.name, e)

            # Populate the MLflow Evaluate tab with pre-computed predictions.
            # model_id links these metrics to the LoggedModel entity created by
            # log_model() above — without it, metrics attach only to the parent
            # Run and the model's own Evaluate tab in the UI stays empty.
            # model=None + predictions= skips model reloading and SHAP (shap not installed).
            try:
                eval_df = X_test.copy()
                eval_df["prediction"] = test_pred.values
                eval_df[target_col] = y_test.values
                mlflow.evaluate(
                    model=None,
                    data=eval_df,
                    targets=target_col,
                    predictions="prediction",
                    model_type="regressor",
                    model_id=model_id,
                )
            except Exception as e:
                logger.warning("mlflow.evaluate skipped for %s: %s", model_cfg.name, e)

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
        if cv_summary is not None:
            training_results["models"][model_cfg.name]["cross_validation"] = cv_summary
        if diag:
            training_results["models"][model_cfg.name]["residual_diagnostics"] = diag

    return training_results
