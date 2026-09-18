"""Training stage for time-series forecasting pipelines: trains ETS,
SARIMAX, and GBM-lagged models, each scored via rolling-origin
cross-validation (average MAPE across n_windows origins sampled from the
training set), and logs each to MLflow.

ETS/SARIMAX are fit once and scored via statsmodels' dynamic get_prediction
(no per-origin refit — verified during planning that dynamic=origin produces
genuine multi-step forecasts, not one-step-ahead using true history). GBM is
fit once (standard sklearn fit) and scored via the shared recursive_forecast()
helper at the same origins, so all three models are compared on the same
multi-step-ahead footing.
"""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.statsmodels
import numpy as np
import pandas as pd

from src.forecasting.model_registry import fit_ets, fit_sarimax
from src.forecasting.rolling_origin import score_gbm_origins, score_statsmodels_origins, select_cv_origins
from src.utils.config import load_forecast_features_config, load_forecast_models_config, load_pipeline_config
from src.utils.io import resolve_run_path
from src.utils.model_registry import get_model

logger = logging.getLogger(__name__)


def train_forecast_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Train all configured forecasting models and log rolling-origin CV metrics to MLflow.

    Args:
        features_dir: Directory containing train/test parquet files.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Dictionary with per-model MLflow run IDs and CV metrics. Models that
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
    y_train = train_df[target_col]
    feature_columns = list(X_train.columns)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    eval_cfg = models_config.evaluation
    min_history_hours = max(features_config.lags + features_config.rolling_windows, default=1)
    origins = select_cv_origins(
        train_df.index, eval_cfg.n_windows, eval_cfg.horizon_hours, min_history_hours=min_history_hours,
    )

    training_results: dict[str, Any] = {"run_id": run_id, "models": {}}

    for model_cfg in models_config.models:
        try:
            with mlflow.start_run(run_name=f"{run_id}_{model_cfg.name}"):
                mlflow.set_tags({
                    "model_name": model_cfg.name,
                    "model_type": model_cfg.type,
                    "run_id": run_id,
                    "pipeline_type": pipeline_config.pipeline_type,
                })
                mlflow.log_param("feature_count", len(feature_columns))
                mlflow.log_param("target_col", target_col)
                mlflow.log_param("horizon_hours", eval_cfg.horizon_hours)
                mlflow.log_param("n_windows", len(origins))
                mlflow.log_dict({"columns": feature_columns}, "feature_columns.json")

                if model_cfg.type == "ets":
                    fitted = fit_ets(y_train, model_cfg.hyperparameters)
                    window_scores = score_statsmodels_origins(fitted, y_train, origins, eval_cfg.horizon_hours)
                    mlflow.statsmodels.log_model(fitted, name="model")
                elif model_cfg.type == "sarimax":
                    # SARIMAX's Kalman filter state-space cost grows with series
                    # length; a seasonal_order period (e.g. 24) fit over a
                    # multi-year hourly history (100k+ rows) is memory-prohibitive
                    # on a typical local/Docker deployment. Daily/weekly
                    # seasonality doesn't need more than a couple of years of
                    # history to estimate well, so max_train_hours (optional,
                    # models.yaml hyperparameter) caps how much trailing history
                    # SARIMAX fits on — ETS and GBM above/below are unaffected
                    # and still train on the full series.
                    max_train_hours = model_cfg.hyperparameters.get("max_train_hours")
                    sarimax_df = train_df.tail(max_train_hours) if max_train_hours else train_df
                    sarimax_y = sarimax_df[target_col]
                    exog = sarimax_df["is_holiday"] if "is_holiday" in sarimax_df.columns else None
                    if max_train_hours:
                        sarimax_origins = select_cv_origins(
                            sarimax_y.index, eval_cfg.n_windows, eval_cfg.horizon_hours,
                            min_history_hours=min_history_hours,
                        )
                        mlflow.log_param("sarimax_train_hours", len(sarimax_y))
                    else:
                        sarimax_origins = origins
                    fitted = fit_sarimax(sarimax_y, model_cfg.hyperparameters, exog=exog)
                    if not fitted.mle_retvals.get("converged", True):
                        logger.warning(
                            "SARIMAX fit for %s did not converge (order/seasonal_order are "
                            "fixed, not searched, so this can happen on some data spans) — "
                            "proceeding with the unconverged fit's parameters.", model_cfg.name,
                        )
                    mlflow.set_tag("sarimax_converged", fitted.mle_retvals.get("converged", True))
                    window_scores = score_statsmodels_origins(fitted, sarimax_y, sarimax_origins, eval_cfg.horizon_hours)
                    mlflow.statsmodels.log_model(fitted, name="model")
                elif model_cfg.type == "gbm":
                    model = get_model("gbm", model_cfg.hyperparameters)
                    model.fit(X_train, y_train)
                    window_scores = score_gbm_origins(
                        model, train_df, target_col, feature_columns, origins, eval_cfg.horizon_hours,
                        features_config.lags, features_config.rolling_windows,
                        features_config.calendar_features, features_config.holiday_features,
                    )
                    # LightGBM's Booster/LGBMRegressor aren't on skops's default
                    # trusted-type list, so mlflow.sklearn.log_model() silently
                    # refuses to save gbm models here (same issue documented in
                    # src/train.py's _MLFLOW_LOG_MODEL_FNS) — use the lightgbm
                    # flavor instead, which serializes natively.
                    mlflow.lightgbm.log_model(model, name="model")
                else:
                    raise ValueError(
                        f"Unknown forecasting model type '{model_cfg.type}'. "
                        f"Expected one of: ets, sarimax, gbm."
                    )

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
            }
            logger.info(
                "Trained %s: cv_mape_mean=%.2f%% +/- %.2f%% (%d windows)",
                model_cfg.name, cv_mape_mean, cv_mape_std, len(window_scores),
            )
        except Exception as e:
            logger.warning("Training skipped for %s: %s", model_cfg.name, e)

    return training_results
