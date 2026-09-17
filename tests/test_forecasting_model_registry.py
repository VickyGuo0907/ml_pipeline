"""Tests for the forecasting model registry: ETS and SARIMAX fit wrappers."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.model_registry import fit_ets, fit_sarimax


def _synthetic_series(n_hours: int = 150, seasonal_period: int = 24) -> pd.Series:
    """A daily-seasonal hourly series with light noise, long enough for ETS/SARIMAX
    to fit without convergence failures (>=6 full seasonal cycles)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period) + rng.normal(0, 1, n_hours)
    return pd.Series(values, index=idx)


class TestFitEts:
    def test_returns_usable_results_supporting_dynamic_prediction(self):
        """The fitted object must support get_prediction(dynamic=...) — Task 3's
        rolling-origin CV depends on this, and the legacy ExponentialSmoothing
        class does NOT support it (verified during planning)."""
        y = _synthetic_series()
        fitted = fit_ets(y, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        origin = y.index[100]
        end = y.index[105]
        pred = fitted.get_prediction(start=origin, end=end, dynamic=origin).predicted_mean

        assert len(pred) == 6

    def test_uses_configured_seasonal_periods(self):
        y = _synthetic_series(n_hours=150, seasonal_period=12)
        fitted = fit_ets(y, {"seasonal_periods": 12, "trend": "add", "seasonal": "add"})
        assert fitted.model.seasonal_periods == 12

    def test_defaults_when_hyperparameters_sparse(self):
        """Missing keys fall back to sane defaults rather than raising."""
        y = _synthetic_series()
        fitted = fit_ets(y, {})
        assert fitted.model.seasonal_periods == 24


class TestFitSarimax:
    def test_returns_usable_results_supporting_dynamic_prediction(self):
        y = _synthetic_series()
        fitted = fit_sarimax(y, {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 24]})

        origin = y.index[100]
        end = y.index[105]
        pred = fitted.get_prediction(start=origin, end=end, dynamic=origin).predicted_mean

        assert len(pred) == 6

    def test_ignores_exog_when_use_holiday_exog_false(self):
        y = _synthetic_series()
        exog = pd.Series((y.index.dayofweek >= 5).astype(int), index=y.index)
        fitted = fit_sarimax(
            y, {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "use_holiday_exog": False}, exog=exog,
        )
        assert fitted.model.k_exog == 0

    def test_uses_exog_when_use_holiday_exog_true(self):
        y = _synthetic_series()
        exog = pd.Series((y.index.dayofweek >= 5).astype(int), index=y.index)
        fitted = fit_sarimax(
            y, {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "use_holiday_exog": True}, exog=exog,
        )
        assert fitted.model.k_exog == 1

    def test_ignores_exog_when_none_provided_even_if_flag_true(self):
        """use_holiday_exog=True with exog=None (e.g. holiday_features disabled
        upstream) must not raise — it just trains without an exogenous regressor."""
        y = _synthetic_series()
        fitted = fit_sarimax(
            y, {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "use_holiday_exog": True}, exog=None,
        )
        assert fitted.model.k_exog == 0

    def test_defaults_when_hyperparameters_sparse(self):
        y = _synthetic_series()
        fitted = fit_sarimax(y, {})
        assert fitted.model.k_exog == 0
