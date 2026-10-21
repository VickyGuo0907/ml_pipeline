"""Chronological validation followed by final training and MLflow logging."""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.statsmodels
import numpy as np
import pandas as pd

from src.forecasting.direct_forecast import fit_direct_forecaster
from src.forecasting.model_registry import fit_ets, fit_sarimax
from src.forecasting.rolling_origin import (
    evaluate_origins,
    select_cv_origins,
)
from src.forecasting.weather import is_weather_column
from src.utils.config import (
    load_forecast_features_config,
    load_forecast_models_config,
    load_pipeline_config,
)
from src.utils.diagnostics import feature_importance
from src.utils.io import resolve_run_path
from src.utils.model_registry import get_model

logger = logging.getLogger(__name__)


def train_forecast_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Validate chronologically, then fit final models and log metrics to MLflow.

    Args:
        features_dir: Directory containing train/test parquet files.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Dictionary with per-model MLflow run IDs and chronological validation metrics. Models that
        fail to fit or score are logged and skipped, not raised.

    Raises:
        FileNotFoundError: If feature files don't exist.
    """
    features_path = resolve_run_path(features_dir, run_id)
    train_path = features_path / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    pipeline_config = load_pipeline_config(config_dir)
    models_config = load_forecast_models_config(config_dir)
    features_config = load_forecast_features_config(config_dir)
    target_col = pipeline_config.target.name

    train_df = pd.read_parquet(train_path)
    X_train = train_df.drop(columns=[target_col])
    feature_columns = list(X_train.columns)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    eval_cfg = models_config.evaluation
    min_history_hours = max(features_config.lags + features_config.rolling_windows, default=1)
    # Reserve the last 20% of training for chronological validation. Parameters
    # are fitted once before its first cutoff, then frozen across all windows.
    validation_index = train_df.index[max(int(len(train_df) * 0.8), min_history_hours) :]
    origins = select_cv_origins(validation_index, eval_cfg.n_windows, eval_cfg.horizon_hours, 0)
    if not origins:
        raise ValueError("Training set too short for chronological daily validation")
    fit_df = train_df.loc[train_df.index < origins[0]]

    training_results: dict[str, Any] = {"run_id": run_id, "models": {}}

    for model_cfg in models_config.models:
        try:
            with mlflow.start_run(run_name=f"{run_id}_{model_cfg.name}"):
                mlflow.set_tags(
                    {
                        "model_name": model_cfg.name,
                        "model_type": model_cfg.type,
                        "run_id": run_id,
                        "pipeline_type": pipeline_config.pipeline_type,
                    }
                )
                # Per-model feature set: a gbm with use_weather: false drops the
                # wx_ columns, giving a same-run with/without-weather ablation.
                hyperparameters = dict(model_cfg.hyperparameters)
                use_weather = bool(hyperparameters.pop("use_weather", True))
                model_features = [
                    c for c in feature_columns if use_weather or not is_weather_column(c)
                ]
                mlflow.log_param("feature_count", len(model_features))
                mlflow.log_param("target_col", target_col)
                mlflow.log_param("horizon_hours", eval_cfg.horizon_hours)
                mlflow.log_param("n_windows", len(origins))
                direct_columns = (
                    model_features + ["lead_hour"]
                    if model_cfg.type == "gbm_direct"
                    else model_features
                )
                mlflow.log_dict({"columns": direct_columns}, "feature_columns.json")
                direct_stride = int(hyperparameters.pop("training_stride_hours", 6))
                direct_examples: dict[str, int] = {}

                def fit_partition(
                    partition: pd.DataFrame,
                    model_cfg: Any = model_cfg,
                    hyperparameters: dict[str, Any] = hyperparameters,
                    model_features: list[str] = model_features,
                    direct_stride: int = direct_stride,
                    direct_examples: dict[str, int] = direct_examples,
                ) -> Any:
                    if model_cfg.type == "seasonal_naive":
                        return None
                    if model_cfg.type == "gbm":
                        estimator = get_model("gbm", hyperparameters)
                        estimator.fit(partition[model_features], partition[target_col])
                        return estimator
                    if model_cfg.type == "gbm_direct":
                        estimator, _, examples = fit_direct_forecaster(
                            partition,
                            target_col,
                            model_features,
                            eval_cfg.horizon_hours,
                            direct_stride,
                            features_config.calendar_features,
                            features_config.holiday_features,
                            hyperparameters,
                            get_model,
                        )
                        direct_examples["fit"] = examples
                        return estimator
                    if model_cfg.type == "ets":
                        return fit_ets(partition[target_col], hyperparameters)
                    if model_cfg.type == "sarimax":
                        cap = hyperparameters.get("max_train_hours")
                        subset = partition.tail(cap) if cap else partition
                        exog = subset.get("is_holiday")
                        return fit_sarimax(subset[target_col], hyperparameters, exog=exog)
                    raise ValueError(f"Unknown forecasting model type '{model_cfg.type}'")

                validation_model = fit_partition(fit_df)
                validation_metrics, validation_predictions = evaluate_origins(
                    validation_model,
                    model_cfg.type,
                    train_df,
                    target_col,
                    direct_columns if model_cfg.type == "gbm_direct" else model_features,
                    origins,
                    eval_cfg.horizon_hours,
                    features_config,
                    fit_df[target_col],
                    int(hyperparameters.get("season_length", 168)),
                )
                window_scores = [
                    float(np.mean(np.abs((g.actual - g.predicted) / g.actual)) * 100)
                    for _, g in validation_predictions.groupby("origin")
                ]
                mlflow.log_dict(validation_metrics, "validation_metrics.json")
                mlflow.log_dict(
                    {
                        "cutoffs": [str(o) for o in origins],
                        "parameter_fit_end": str(fit_df.index[-1]),
                        "parameter_refit_cadence": "once before validation; final refit on full train",
                        "state_refresh": "pre-cutoff observations only",
                    },
                    "validation_protocol.json",
                )
                mlflow.set_tag("weather_information", validation_metrics["weather_information"])
                mlflow.log_param("validation_fit_end", str(fit_df.index[-1]))
                for key in ("mae", "rmse", "mape", "mase"):
                    if validation_metrics[key] is not None:
                        mlflow.log_metric(f"cv_{key}", validation_metrics[key])
                fitted = fit_partition(train_df)
                if model_cfg.type == "seasonal_naive":
                    mlflow.log_param(
                        "season_length", int(hyperparameters.get("season_length", 168))
                    )
                elif model_cfg.type in ("ets", "sarimax"):
                    mlflow.statsmodels.log_model(fitted, name="model")
                    if model_cfg.type == "sarimax" and hyperparameters.get("max_train_hours"):
                        mlflow.log_param(
                            "sarimax_train_hours",
                            min(len(train_df), hyperparameters["max_train_hours"]),
                        )
                elif model_cfg.type in ("gbm", "gbm_direct"):
                    mlflow.log_param("use_weather", use_weather)
                    mlflow.lightgbm.log_model(fitted, name="model")
                    importance = feature_importance(fitted, direct_columns)
                    if importance:
                        mlflow.log_dict(importance, "feature_importance.json")
                    if model_cfg.type == "gbm_direct":
                        mlflow.log_param("direct_training_stride_hours", direct_stride)
                        mlflow.log_param("direct_training_examples", direct_examples.get("fit", 0))

                if not window_scores:
                    raise ValueError(
                        f"No scorable rolling-origin windows for {model_cfg.name} — "
                        f"training set too short for horizon_hours={eval_cfg.horizon_hours}."
                    )
                cv_mape_mean = float(np.mean(window_scores))
                cv_mape_std = float(np.std(window_scores))
                mlflow.log_metric("cv_mape_mean", cv_mape_mean)
                mlflow.log_metric("cv_mape_std", cv_mape_std)
                for i, score in enumerate(window_scores, 1):
                    mlflow.log_metric(f"cv_mape_window_{i}", score)

                mlflow_run_id = mlflow.active_run().info.run_id

            training_results["models"][model_cfg.name] = {
                "mlflow_run_id": mlflow_run_id,
                "cv_mape_mean": cv_mape_mean,
                "cv_mape_std": cv_mape_std,
                "n_windows_scored": len(window_scores),
                "validation_metrics": validation_metrics,
            }
            logger.info(
                "Trained %s: cv_mape_mean=%.2f%% +/- %.2f%% (%d windows)",
                model_cfg.name,
                cv_mape_mean,
                cv_mape_std,
                len(window_scores),
            )
        except Exception as e:
            logger.warning("Training skipped for %s: %s", model_cfg.name, e)

    return training_results
