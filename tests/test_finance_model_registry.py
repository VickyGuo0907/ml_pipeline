"""Tests for the finance model registry: Random Walk, Mean, ARIMA, and
cross-sectional GBM fit functions."""
import numpy as np
import pandas as pd
import pytest

from src.finance.model_registry import fit_arima, fit_cross_sectional_gbm, fit_mean, fit_random_walk


def _synthetic_returns(n: int = 36, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.01, 0.05, n))


class TestFitRandomWalk:
    def test_forecast_is_always_zero(self):
        y = _synthetic_returns()
        fitted = fit_random_walk(y)
        forecast = fitted.forecast(5)
        assert list(forecast) == pytest.approx([0.0] * 5)

    def test_fittedvalues_are_all_zero_and_indexed_like_input(self):
        y = _synthetic_returns()
        fitted = fit_random_walk(y)
        assert list(fitted.fittedvalues.index) == list(y.index)
        assert (fitted.fittedvalues == 0.0).all()

    def test_predict_matches_pyfunc_contract(self):
        """mlflow.pyfunc.PythonModel.predict(context, model_input) -> array
        of len(model_input), zeros regardless of content."""
        y = _synthetic_returns()
        fitted = fit_random_walk(y)
        model_input = pd.DataFrame({"anything": [1, 2, 3]})
        result = fitted.predict(None, model_input)
        assert list(result) == pytest.approx([0.0, 0.0, 0.0])


class TestFitMean:
    def test_full_history_forecast_equals_hand_computed_mean(self):
        y = pd.Series([0.01, 0.02, 0.03, 0.04])
        fitted = fit_mean(y, {"window": None})
        assert list(fitted.forecast(3)) == pytest.approx([0.025, 0.025, 0.025])

    def test_full_history_fittedvalues_are_constant_mean(self):
        y = pd.Series([0.01, 0.02, 0.03, 0.04])
        fitted = fit_mean(y, {"window": None})
        assert fitted.fittedvalues.tolist() == pytest.approx([0.025] * len(fitted.fittedvalues))

    def test_default_window_is_none_when_key_missing(self):
        y = pd.Series([0.01, 0.02, 0.03])
        fitted = fit_mean(y, {})
        assert list(fitted.forecast(1)) == pytest.approx([y.mean()])

    def test_trailing_window_forecast_uses_only_recent_values(self):
        y = pd.Series([0.10, 0.10, 0.10, 0.01, 0.02, 0.03])  # last 3: 0.01, 0.02, 0.03 -> mean 0.02
        fitted = fit_mean(y, {"window": 3})
        assert list(fitted.forecast(1)) == pytest.approx([0.02])

    def test_trailing_window_fittedvalues_exclude_current_row(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        fitted = fit_mean(y, {"window": 2})
        # Row at index 2 (value 3.0): trailing mean of the 2 PRIOR values (1.0, 2.0), excluding itself.
        assert fitted.fittedvalues.iloc[2] == pytest.approx(1.5)
        # The first 2 rows have no full trailing window yet -> NaN.
        assert fitted.fittedvalues.iloc[:2].isna().all()


class TestFitArima:
    def test_returns_fitted_result_with_fittedvalues_and_forecast(self):
        y = _synthetic_returns(n=36)
        fitted = fit_arima(y, {"order": [1, 0, 0]})
        assert len(fitted.fittedvalues) == len(y)
        forecast = fitted.forecast(steps=4)
        assert len(forecast) == 4

    def test_default_order_when_hyperparameters_sparse(self):
        y = _synthetic_returns(n=36)
        fitted = fit_arima(y, {})
        assert fitted.model.order == (1, 0, 0)


class TestFitCrossSectionalGbm:
    def test_fits_and_predicts_on_stacked_panel(self):
        panel_df = pd.DataFrame({
            "log_return": [0.01, 0.02, -0.01, 0.03, 0.00, 0.015] * 5,
            "lag_1_return": [0.01, 0.00, 0.02, -0.01, 0.01, 0.02] * 5,
            "trailing_12m_vol": [0.02] * 30,
            "ticker_encoded": ([0] * 6 + [1] * 6 + [0] * 6 + [1] * 6 + [0] * 6),
        })
        feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
        hyperparameters = {"n_estimators": 5, "max_depth": 2, "random_state": 42}

        model = fit_cross_sectional_gbm(panel_df, "log_return", feature_columns, hyperparameters)
        predictions = model.predict(panel_df[feature_columns])

        assert len(predictions) == len(panel_df)
