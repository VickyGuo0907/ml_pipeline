"""Tests for shared rolling-origin scoring utilities: MAPE, origin selection,
and per-model-family scoring (statsmodels dynamic prediction, GBM recursive
forecasting). Used by both the training stage (Plan 3, in-sample CV) and the
evaluation stage (Plan 4, out-of-sample test scoring)."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.model_registry import fit_ets, fit_sarimax
from src.forecasting.rolling_origin import (
    mape,
    score_gbm_origins,
    score_statsmodels_origins,
    select_cv_origins,
)


def _synthetic_series(n_hours: int = 250, seasonal_period: int = 24) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period) + rng.normal(0, 1, n_hours)
    return pd.Series(values, index=idx)


def _synthetic_train_df(n_hours: int = 200, seasonal_period: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h", name="Datetime")
    target = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period) + rng.normal(0, 1, n_hours)
    target = pd.Series(target, index=idx)
    df = pd.DataFrame({"PJME_MW": target})
    df["lag_1h"] = df["PJME_MW"].shift(1)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df.dropna()


class TestMape:
    def test_zero_error_is_zero_percent(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        assert mape(actual, actual) == pytest.approx(0.0)

    def test_known_percentage_error(self):
        actual = pd.Series([100.0, 200.0])
        predicted = pd.Series([110.0, 180.0])  # +10%, -10%
        assert mape(actual, predicted) == pytest.approx(10.0)


class TestSelectCvOrigins:
    def test_origins_leave_room_for_full_horizon(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        origins = select_cv_origins(idx, n_windows=3, horizon_hours=10)
        for origin in origins:
            end = origin + pd.Timedelta(hours=9)
            assert end <= idx[-1]

    def test_min_history_hours_pushes_first_origin_past_naive_margin(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        origins = select_cv_origins(idx, n_windows=5, horizon_hours=10, min_history_hours=5)
        assert origins[0] >= idx[5]


class TestScoreStatsmodelsOrigins:
    def test_in_sample_dynamic_scores_each_valid_origin(self):
        """dynamic=True (the default) — Plan 3's in-sample CV usage, unchanged."""
        df = _synthetic_train_df()
        fitted = fit_ets(df["PJME_MW"], {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})
        origins = select_cv_origins(df.index, n_windows=2, horizon_hours=5)

        scores = score_statsmodels_origins(fitted, df["PJME_MW"], origins, horizon_hours=5)

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)

    def test_out_of_sample_dynamic_false_works_for_ets(self):
        """The empirically-verified fix: dynamic=origin RAISES for an
        out-of-sample origin with ETSModel ('Cannot anchor simulation outside
        of the sample'). dynamic=False must be used instead, and must work."""
        full = _synthetic_series(250)
        y_train = full.iloc[:200]
        y_test = full.iloc[200:]
        fitted = fit_ets(y_train, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        origins = select_cv_origins(y_test.index, n_windows=2, horizon_hours=5)
        full_y = pd.concat([y_train, y_test])

        scores = score_statsmodels_origins(fitted, full_y, origins, horizon_hours=5, dynamic=False)

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)

    def test_out_of_sample_dynamic_true_raises_for_ets(self):
        """Pin the exact failure mode this task's dynamic parameter exists to
        avoid — if this stops raising, statsmodels' behavior changed and the
        dynamic=False branch may no longer be necessary (or something else broke)."""
        full = _synthetic_series(250)
        y_train = full.iloc[:200]
        y_test = full.iloc[200:]
        fitted = fit_ets(y_train, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        origins = select_cv_origins(y_test.index, n_windows=1, horizon_hours=5)
        full_y = pd.concat([y_train, y_test])

        with pytest.raises(ValueError, match="outside of the sample"):
            score_statsmodels_origins(fitted, full_y, origins, horizon_hours=5, dynamic=True)

    def test_out_of_sample_dynamic_false_works_for_sarimax(self):
        full = _synthetic_series(250)
        y_train = full.iloc[:200]
        y_test = full.iloc[200:]
        fitted = fit_sarimax(y_train, {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 24]})

        origins = select_cv_origins(y_test.index, n_windows=2, horizon_hours=5)
        full_y = pd.concat([y_train, y_test])

        scores = score_statsmodels_origins(fitted, full_y, origins, horizon_hours=5, dynamic=False)

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)


class TestScoreGbmOrigins:
    def test_scores_each_valid_origin(self):
        df = _synthetic_train_df()

        class _MeanModel:
            def predict(self, X):
                return np.full(len(X), df["PJME_MW"].mean())

        origins = select_cv_origins(df.index, n_windows=2, horizon_hours=5)
        scores = score_gbm_origins(
            _MeanModel(), df, "PJME_MW", ["lag_1h", "hour"], origins, horizon_hours=5,
            lags=[1], rolling_windows=[], calendar_features=True, holiday_features=False,
        )

        assert len(scores) == len(origins)
        assert all(s >= 0 for s in scores)
