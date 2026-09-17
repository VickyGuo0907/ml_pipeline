"""Statsmodels model wrappers for forecasting pipelines: ETS and SARIMAX.

Complements src/utils/model_registry.py — "gbm" is resolved there and reused
as-is (trained via the normal sklearn fit(X, y)/predict(X) interface on the
engineered lag/rolling/calendar/holiday feature matrix). ETS and SARIMAX are
univariate statsmodels models with a fundamentally different
fit(endog)/get_prediction() interface, so they get their own fit functions
here rather than a class-based registry.
"""
from typing import Any

import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_ets(y: pd.Series, hyperparameters: dict[str, Any]) -> Any:
    """Fit a state-space ETS (error-trend-seasonal) model.

    Uses statsmodels' modern state-space ETSModel — not the legacy
    tsa.holtwinters.ExponentialSmoothing — because only the state-space
    implementation supports get_prediction(dynamic=...), the mechanism the
    rolling-origin CV in train_forecast.py depends on to score multi-step
    forecasts from many origins without refitting.

    Args:
        y: Training target series, DatetimeIndex, hourly.
        hyperparameters: seasonal_periods (int), trend (str), seasonal (str) —
            from models.yaml. Defaults to seasonal_periods=24 (daily), both
            trend/seasonal="add" if unspecified.

    Returns:
        Fitted ETSResultsWrapper.
    """
    model = ETSModel(
        y,
        trend=hyperparameters.get("trend", "add"),
        seasonal=hyperparameters.get("seasonal", "add"),
        seasonal_periods=hyperparameters.get("seasonal_periods", 24),
    )
    return model.fit(disp=False)


def fit_sarimax(y: pd.Series, hyperparameters: dict[str, Any], exog: pd.Series | None = None) -> Any:
    """Fit a SARIMAX model, optionally with an exogenous regressor.

    order/seasonal_order are fixed via config, never searched — consistent
    with the project's ban on automated hyperparameter search.

    Args:
        y: Training target series, DatetimeIndex, hourly.
        hyperparameters: order (list[int], default [1,0,0]), seasonal_order
            (list[int], default [0,0,0,0]), use_holiday_exog (bool, default
            False) — from models.yaml.
        exog: Exogenous regressor (e.g. the is_holiday column), used only if
            hyperparameters["use_holiday_exog"] is true AND exog is not None;
            ignored otherwise (never raises for a missing/disabled exog).

    Returns:
        Fitted SARIMAXResultsWrapper.
    """
    order = tuple(hyperparameters.get("order", (1, 0, 0)))
    seasonal_order = tuple(hyperparameters.get("seasonal_order", (0, 0, 0, 0)))
    use_exog = hyperparameters.get("use_holiday_exog", False) and exog is not None
    model = SARIMAX(
        y,
        exog=exog if use_exog else None,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)
