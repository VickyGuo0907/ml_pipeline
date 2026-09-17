"""Recursive multi-step forecasting for sklearn-style forecasting models
(GBM). Statsmodels models (ETS/SARIMAX) forecast natively via
get_prediction()/forecast() and don't need this — recursive stepping exists
because GBM's lag/rolling features must be rebuilt from scratch at each
future step, since real historical values run out beyond the training data.

Used in three places: rolling-origin CV during training (train_forecast.py,
Plan 3), final test-set evaluation (evaluate_forecast.py, Plan 4), and live
serving (serve_forecast.py, Plan 5) — written once here so all three share
identical step-by-step feature logic.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

logger = logging.getLogger(__name__)


def _build_feature_row(
    buffer: pd.Series,
    ts: pd.Timestamp,
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
    holidays: pd.DatetimeIndex | None,
) -> dict[str, Any]:
    """Build one feature row for timestamp ts from a growing value buffer.

    Mirrors engineer_forecast_features' bulk feature computation (Plan 2)
    row-by-row: lag_Nh and rolling_*_Wh are read directly from buffer (real
    history for the first step, the model's own prior predictions for later
    steps), rather than via pandas .shift()/.rolling() over a whole
    DataFrame. A lag/window reaching before buffer's start is NaN, matching
    pd.Series.shift()'s own behavior in the bulk pipeline.

    Args:
        buffer: Known + previously-predicted target values, DatetimeIndex,
            hourly, covering at least the longest lag/rolling window before ts.
        ts: Timestamp to build the feature row for.
        lags: Lag hours (matches ForecastFeaturesConfig.lags).
        rolling_windows: Rolling window hours (matches .rolling_windows).
        calendar_features: Whether to add hour/day_of_week/month/is_weekend.
        holiday_features: Whether to add is_holiday/days_to_nearest_holiday.
        holidays: Precomputed holiday DatetimeIndex spanning well beyond
            buffer's and the forecast horizon's range (padded, like
            engineer_forecast_features' ±200-day window) — computed once per
            recursive_forecast() call, not recomputed per step, since it's
            identical for every step.

    Returns:
        Dict of feature name -> value for this single timestamp.
    """
    row: dict[str, Any] = {}
    for lag in lags:
        row[f"lag_{lag}h"] = buffer.get(ts - pd.Timedelta(hours=lag), np.nan)
    for window in rolling_windows:
        window_vals = pd.Series([
            buffer.get(ts - pd.Timedelta(hours=h), np.nan) for h in range(1, window + 1)
        ])
        # Match pd.Series.rolling(window).mean()/std() default: min_periods=window,
        # so NaN if any position in the window is missing (not skipna average).
        if window_vals.isna().any():
            row[f"rolling_mean_{window}h"] = np.nan
            row[f"rolling_std_{window}h"] = np.nan
        else:
            row[f"rolling_mean_{window}h"] = window_vals.mean()
            row[f"rolling_std_{window}h"] = window_vals.std()
    if calendar_features:
        row["hour"] = ts.hour
        row["day_of_week"] = ts.dayofweek
        row["month"] = ts.month
        row["is_weekend"] = int(ts.dayofweek >= 5)
    if holiday_features:
        normalized_ts = ts.normalize()
        is_holiday = holidays is not None and normalized_ts in holidays
        row["is_holiday"] = int(is_holiday)
        if holidays is not None and len(holidays):
            row["days_to_nearest_holiday"] = int(
                np.abs(holidays.values - np.datetime64(normalized_ts)).min() / np.timedelta64(1, "D")
            )
        else:
            row["days_to_nearest_holiday"] = np.nan
    return row


def recursive_forecast(
    model: Any,
    history: pd.Series,
    horizon: int,
    feature_columns: list[str],
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
) -> pd.Series:
    """Recursively forecast `horizon` future hourly steps from a fitted model.

    At each step: build the feature row from real + previously-predicted
    history, predict one step, append the prediction to the buffer, repeat.
    This lets a lag-feature model (GBM) forecast a multi-step horizon on the
    same footing as ETS/SARIMAX's native multi-step get_prediction(), for a
    fair three-model comparison.

    Args:
        model: Fitted sklearn-compatible estimator (.predict(X) -> array),
            trained on exactly the feature set this function reconstructs.
        history: Real (or, for chained calls, previously-forecasted) target
            values, DatetimeIndex, hourly, sorted ascending. The forecast
            starts one hour after history's last timestamp.
        horizon: Number of future hourly steps to forecast.
        feature_columns: Exact column order the model expects — must match
            X_train.columns from training, so column order/set is identical
            (sklearn estimators are positional, not name-aware).
        lags, rolling_windows, calendar_features, holiday_features: Same
            feature-engineering config as engineer_forecast_features, so
            per-step features match what the model was trained on.

    Returns:
        Series of `horizon` predicted values, DatetimeIndex continuing
        directly from history.
    """
    buffer = history.copy()
    holidays = None
    if holiday_features:
        calendar = USFederalHolidayCalendar()
        padding = pd.Timedelta(days=200)
        holidays = calendar.holidays(
            start=buffer.index.min() - padding,
            end=buffer.index.max() + pd.Timedelta(hours=horizon) + padding,
        )

    predictions = []
    next_ts = buffer.index[-1] + pd.Timedelta(hours=1)
    for _ in range(horizon):
        row = _build_feature_row(
            buffer, next_ts, lags, rolling_windows, calendar_features, holiday_features, holidays,
        )
        X_row = pd.DataFrame([row], index=[next_ts])[feature_columns]
        pred = float(model.predict(X_row)[0])
        predictions.append(pred)
        buffer.loc[next_ts] = pred
        next_ts = next_ts + pd.Timedelta(hours=1)

    forecast_index = buffer.index[-horizon:]
    return pd.Series(predictions, index=forecast_index)
