"""Direct multi-step LightGBM regression with one model across forecast leads.

Each training row represents an issue cutoff and lead hour. Load lag/rolling
features end before the cutoff; target-time calendar/weather features align to
that lead. At prediction time, load features stay anchored at the cutoff instead
of feeding earlier predictions back into later leads.
"""
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.forecasting.recursive import _build_feature_row
from src.forecasting.weather import weather_columns


def _calendar_features(index: pd.DatetimeIndex, holiday_features: bool) -> pd.DataFrame:
    """Build target-valid-time calendar features for a batch of leads."""
    frame = pd.DataFrame(index=np.arange(len(index)))
    frame["hour"] = index.hour
    frame["day_of_week"] = index.dayofweek
    frame["month"] = index.month
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["year"] = index.year
    if holiday_features:
        normalized = index.normalize().values
        holiday_index = USFederalHolidayCalendar().holidays(
            index.min().normalize() - pd.Timedelta(days=200),
            index.max().normalize() + pd.Timedelta(days=200),
        )
        holiday_values = holiday_index.values
        positions = np.searchsorted(holiday_values, normalized)
        left = np.maximum(positions - 1, 0)
        right = np.minimum(positions, len(holiday_values) - 1)
        distance = np.minimum(
            np.abs(normalized - holiday_values[left]),
            np.abs(normalized - holiday_values[right]),
        ) // np.timedelta64(1, "D")
        frame["is_holiday"] = index.normalize().isin(holiday_index).astype(int)
        frame["days_to_nearest_holiday"] = distance.astype(int)
    return frame


def _training_matrix(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    horizon_hours: int,
    stride_hours: int,
    calendar_features: bool,
    holiday_features: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    if stride_hours < 1 or horizon_hours < 1:
        raise ValueError("stride_hours and horizon_hours must be positive")
    feature_names = list(feature_columns) + ["lead_hour"]
    if "lead_hour" in feature_columns:
        raise ValueError("lead_hour is reserved for direct forecasting")
    positions = np.arange(0, len(frame) - horizon_hours, stride_hours)
    if not len(positions):
        raise ValueError("Insufficient history to build direct training examples")
    base = frame[feature_columns].iloc[positions].reset_index(drop=True)
    x_parts, y_parts = [], []
    timestamps = frame.index
    wx_cols = weather_columns(feature_columns)
    for lead in range(1, horizon_hours + 1):
        targets = positions + lead
        X = base.copy()
        valid_times = timestamps.take(targets)
        target_calendar = _calendar_features(valid_times, holiday_features)
        if calendar_features:
            for column in ("hour", "day_of_week", "month", "is_weekend", "year"):
                X[column] = target_calendar[column].to_numpy()
        if holiday_features:
            X["is_holiday"] = target_calendar["is_holiday"].to_numpy()
            X["days_to_nearest_holiday"] = target_calendar["days_to_nearest_holiday"].to_numpy()
        if wx_cols:
            X.loc[:, wx_cols] = frame[wx_cols].iloc[targets].to_numpy()
        X["lead_hour"] = lead
        x_parts.append(X[feature_names])
        y_parts.append(frame[target_col].iloc[targets].reset_index(drop=True))
    return pd.concat(x_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)


def fit_direct_forecaster(
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    horizon_hours: int,
    stride_hours: int,
    calendar_features: bool,
    holiday_features: bool,
    hyperparameters: dict[str, Any],
    estimator_factory: Any,
) -> tuple[Any, list[str], int]:
    """Fit one lead-conditioned regression model using labels before frame end."""
    X, y = _training_matrix(
        frame,
        target_col,
        feature_columns,
        horizon_hours,
        stride_hours,
        calendar_features,
        holiday_features,
    )
    model = estimator_factory("gbm", hyperparameters)
    model.fit(X, y)
    return model, list(X.columns), len(X)


def direct_forecast(
    model: Any,
    frame: pd.DataFrame,
    target_col: str,
    model_columns: list[str],
    feature_columns: list[str],
    origin: pd.Timestamp,
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
) -> pd.Series:
    """Forecast each lead from the same pre-cutoff load history."""
    history = frame[target_col].loc[: origin - pd.Timedelta(hours=1)]
    if history.empty:
        raise ValueError("Direct forecast requires target history before cutoff")
    holidays = None
    if holiday_features:
        calendar = USFederalHolidayCalendar()
        pad = pd.Timedelta(days=200)
        holidays = calendar.holidays(
            history.index.min() - pad, origin + pd.Timedelta(hours=horizon_hours) + pad
        )
    wx_cols = weather_columns(feature_columns)
    rows = []
    index = pd.date_range(origin, periods=horizon_hours, freq="h")
    for lead, valid_time in enumerate(index, start=1):
        row = _build_feature_row(
            history, origin, lags, rolling_windows, calendar_features, holiday_features, holidays
        )
        target_calendar = _calendar_features(pd.DatetimeIndex([valid_time]), holiday_features)
        row.update(target_calendar.iloc[0].to_dict())
        if wx_cols:
            if valid_time not in frame.index:
                raise ValueError(f"Missing future weather at {valid_time}")
            row.update(frame.loc[valid_time, wx_cols].to_dict())
        row["lead_hour"] = lead
        rows.append(row)
    X = pd.DataFrame(rows, index=index)[model_columns]
    return pd.Series(model.predict(X), index=index, name="forecast")
