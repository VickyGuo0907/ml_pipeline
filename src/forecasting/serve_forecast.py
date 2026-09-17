"""Live forecast serving: given a model already loaded from MLflow by
src/serve.py (which handles the MLflow flavor dispatch), produce a
timestamped multi-step forecast.

ETS/SARIMAX forecast natively from their own persisted fitted state —
trustworthiness degrades the longer it's been since training (the fit's
own state grows stale relative to "now"), addressed operationally via
retraining cadence, not architecturally here. A model fit with an
exogenous regressor (the real config/pjm_load_forecast/models.yaml sets
use_holiday_exog: true for SARIMAX) needs future exog values supplied
explicitly — is_holiday is fully computable in advance from the calendar,
so this never depends on unknown future data.

GBM forecasts via the same recursive_forecast() helper Plans 3-4 already
use for scoring, seeded from the latest run's last_window.parquet snapshot
(written by engineer_forecast_features, Plan 2) so live predictions are
grounded in real, recent data rather than whatever was in the training set.
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.forecasting.recursive import recursive_forecast
from src.utils.io import find_latest_run_id, resolve_run_path

logger = logging.getLogger(__name__)


def load_latest_snapshot(features_dir: str | Path) -> pd.DataFrame | None:
    """Load the most recent run's last_window.parquet snapshot.

    Args:
        features_dir: Pipeline features directory (e.g. data/pjm_load_forecast/features).

    Returns:
        The snapshot DataFrame (DatetimeIndex), or None if no run has
        produced a snapshot yet (pipeline never run, or ran before Plan 2).
    """
    latest_run_id = find_latest_run_id(features_dir)
    if latest_run_id is None:
        return None
    snapshot_path = resolve_run_path(features_dir, latest_run_id) / "last_window.parquet"
    if not snapshot_path.exists():
        return None
    return pd.read_parquet(snapshot_path)


def forecast_with_statsmodels(model: Any, horizon_hours: int) -> pd.Series:
    """Forecast horizon_hours ahead from a fitted ETS/SARIMAX model's own state.

    Args:
        model: Loaded ETSResultsWrapper or SARIMAXResultsWrapper (via
            mlflow.statsmodels.load_model).
        horizon_hours: Number of future hourly steps to forecast.

    Returns:
        Series of horizon_hours predicted values, DatetimeIndex continuing
        from the model's own last fitted timestamp (not necessarily "now" —
        see module docstring on staleness).
    """
    if getattr(model.model, "k_exog", 0) > 0:
        last_fitted_ts = pd.Timestamp(model.model.data.dates[-1])
        future_index = pd.date_range(
            last_fitted_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h",
        )
        calendar = USFederalHolidayCalendar()
        padding = pd.Timedelta(days=200)
        holidays = calendar.holidays(
            start=future_index.min() - padding, end=future_index.max() + padding,
        )
        # A named Series, matching exactly how train_forecast.py fits SARIMAX
        # (exog = train_df["is_holiday"], a Series — not a DataFrame). Verified
        # empirically during planning that a Series works identically to a
        # DataFrame here, and matching the fit-time shape is the safer choice.
        future_exog = pd.Series(
            future_index.normalize().isin(holidays).astype(int), index=future_index, name="is_holiday",
        )
        return model.forecast(horizon_hours, exog=future_exog)
    return model.forecast(horizon_hours)


def forecast_with_gbm(
    model: Any,
    snapshot: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
) -> pd.Series:
    """Forecast horizon_hours ahead from a fitted GBM model, seeded by the
    latest real snapshot.

    Args:
        model: Loaded sklearn-compatible estimator (via mlflow.lightgbm.load_model).
        snapshot: Latest known real actuals (last_window.parquet), DatetimeIndex.
        target_col: Target column name.
        feature_columns: Exact column order the model expects.
        horizon_hours: Number of future hourly steps to forecast.
        lags, rolling_windows, calendar_features, holiday_features: Feature
            config, passed through to recursive_forecast — read from the
            pipeline's own config/<pipeline_type>/features.yaml by the
            caller (src/serve.py), not duplicated here.

    Returns:
        Series of horizon_hours predicted values, DatetimeIndex continuing
        from the snapshot's last timestamp.
    """
    history = snapshot[target_col]
    return recursive_forecast(
        model, history, horizon_hours, feature_columns,
        lags, rolling_windows, calendar_features, holiday_features,
    )
