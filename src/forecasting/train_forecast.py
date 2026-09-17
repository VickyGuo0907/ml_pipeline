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
from src.forecasting.recursive import recursive_forecast
from src.utils.config import load_forecast_features_config, load_forecast_models_config, load_pipeline_config
from src.utils.io import resolve_run_path
from src.utils.model_registry import get_model

logger = logging.getLogger(__name__)


def _mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error, as a percentage (0-100+ scale).

    Safe for PJM load values (always well above zero); no zero-guard needed
    for this pipeline's target column.
    """
    actual_arr = actual.to_numpy(dtype=float)
    predicted_arr = predicted.to_numpy(dtype=float)
    return float(np.mean(np.abs((actual_arr - predicted_arr) / actual_arr)) * 100)


def _select_cv_origins(
    index: pd.DatetimeIndex,
    n_windows: int,
    horizon_hours: int,
    min_history_hours: int = 1,
) -> list[pd.Timestamp]:
    """Pick up to n_windows evenly-spaced rolling-origin timestamps from index.

    Each origin leaves at least horizon_hours of real data after it (inclusive
    of the origin itself), so the true values needed to score that window
    actually exist in index. Each origin also leaves at least min_history_hours
    of real data before it — never index[0] itself — so GBM scoring
    (_score_gbm_origins) has enough real history before the origin for its
    longest configured lag/rolling window to be fully populated (a naive
    1-hour margin leaves e.g. lag_168h/rolling_mean_168h as NaN at the first
    origin, which LightGBM still predicts on, producing a garbage outlier
    score). statsmodels scoring doesn't need this margin (get_prediction
    relies on the already-fitted model, not rebuilt lag features), but
    sharing one origin set keeps both model families scored on identical
    cutoffs for a fair comparison, so the margin must satisfy GBM's stricter
    requirement.

    Args:
        index: Training set's DatetimeIndex (sorted ascending).
        n_windows: Number of origins to pick.
        horizon_hours: Forecast horizon — origins within horizon_hours - 1
            hours of the end of index are excluded.
        min_history_hours: Minimum real history required before an origin
            (e.g. max of configured lags/rolling_windows). Defaults to 1,
            the previous hardcoded minimum.

    Returns:
        List of up to n_windows Timestamps (fewer if the series is too short
        to support that many distinct positions), or an empty list if the
        series can't support even one full horizon plus min_history_hours of
        history.
    """
    usable_start = max(min_history_hours, 1)
    usable_count = len(index) - horizon_hours + 1
    if usable_count <= usable_start:
        return []
    usable = index[usable_start:usable_count]
    n = min(n_windows, len(usable))
    positions = np.linspace(0, len(usable) - 1, n).astype(int)
    return [usable[p] for p in sorted(set(positions))]


def _score_statsmodels_origins(
    fitted: Any, y: pd.Series, origins: list[pd.Timestamp], horizon_hours: int,
) -> list[float]:
    """Score a fitted ETS/SARIMAX result at each origin via dynamic get_prediction.

    Args:
        fitted: Fitted ETSResultsWrapper or SARIMAXResultsWrapper.
        y: The full training target series (same series the model was fit on).
        origins: Rolling-origin timestamps from _select_cv_origins.
        horizon_hours: Forecast horizon per origin.

    Returns:
        List of MAPE scores, one per origin that had enough trailing data.
    """
    scores = []
    for origin in origins:
        end = origin + pd.Timedelta(hours=horizon_hours - 1)
        if end > y.index[-1]:
            continue
        pred = fitted.get_prediction(start=origin, end=end, dynamic=origin).predicted_mean
        actual = y.loc[origin:end]
        scores.append(_mape(actual, pred))
    return scores


def _score_gbm_origins(
    model: Any,
    train_df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    origins: list[pd.Timestamp],
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
) -> list[float]:
    """Score a fitted GBM model at each origin via recursive_forecast.

    Args:
        model: Fitted sklearn-compatible estimator.
        train_df: Full training feature matrix (DatetimeIndex), including target_col.
        target_col: Target column name.
        feature_columns: Exact column order the model expects.
        origins: Rolling-origin timestamps from _select_cv_origins.
        horizon_hours: Forecast horizon per origin.
        lags, rolling_windows, calendar_features, holiday_features: Feature
            config, passed through to recursive_forecast.

    Returns:
        List of MAPE scores, one per origin that had enough trailing data
        and at least one hour of history before it.
    """
    scores = []
    y = train_df[target_col]
    for origin in origins:
        end = origin + pd.Timedelta(hours=horizon_hours - 1)
        if end > y.index[-1]:
            continue
        history = y.loc[: origin - pd.Timedelta(hours=1)]
        if history.empty:
            continue
        pred = recursive_forecast(
            model, history, horizon_hours, feature_columns,
            lags, rolling_windows, calendar_features, holiday_features,
        )
        actual = y.loc[origin:end]
        scores.append(_mape(actual, pred))
    return scores


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
    origins = _select_cv_origins(
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
                    window_scores = _score_statsmodels_origins(fitted, y_train, origins, eval_cfg.horizon_hours)
                    mlflow.statsmodels.log_model(fitted, name="model")
                elif model_cfg.type == "sarimax":
                    exog = train_df["is_holiday"] if "is_holiday" in train_df.columns else None
                    fitted = fit_sarimax(y_train, model_cfg.hyperparameters, exog=exog)
                    if not fitted.mle_retvals.get("converged", True):
                        logger.warning(
                            "SARIMAX fit for %s did not converge (order/seasonal_order are "
                            "fixed, not searched, so this can happen on some data spans) — "
                            "proceeding with the unconverged fit's parameters.", model_cfg.name,
                        )
                    mlflow.set_tag("sarimax_converged", fitted.mle_retvals.get("converged", True))
                    window_scores = _score_statsmodels_origins(fitted, y_train, origins, eval_cfg.horizon_hours)
                    mlflow.statsmodels.log_model(fitted, name="model")
                elif model_cfg.type == "gbm":
                    model = get_model("gbm", model_cfg.hyperparameters)
                    model.fit(X_train, y_train)
                    window_scores = _score_gbm_origins(
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
