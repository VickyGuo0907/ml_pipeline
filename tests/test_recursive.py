"""Tests for recursive multi-step forecasting: the row-by-row feature
builder used to score/serve GBM on the same multi-step footing as
ETS/SARIMAX's native forecast()/get_prediction()."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.recursive import _build_feature_row, recursive_forecast


class _IdentityLag1Model:
    """A fake model whose predict() returns lag_1h verbatim — makes recursive
    stepping fully hand-verifiable: given a history ending at value V, every
    subsequent prediction should also be V (each step's lag_1h is the
    previous step's prediction, which was V, forever)."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_1h"].to_numpy()


class TestBuildFeatureRow:
    """Exact parity checks against the same lag/rolling/calendar/holiday
    formulas Plan 2's engineer_forecast_features computes in bulk."""

    def test_lag_and_rolling_match_bulk_formulas(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="h")
        buffer = pd.Series(range(20), index=idx, dtype=float)  # values 0..19
        ts = idx[10] + pd.Timedelta(hours=1)  # predicting the step right after hour 10 (value 10)

        row = _build_feature_row(
            buffer, ts, lags=[1, 2], rolling_windows=[3],
            calendar_features=False, holiday_features=False, holidays=None,
        )

        # Bulk formula (Plan 2): lag_1h = value 1h before ts = buffer at hour10 = 10.0
        assert row["lag_1h"] == pytest.approx(10.0)
        assert row["lag_2h"] == pytest.approx(9.0)
        # rolling_mean_3h = mean of the 3 hours immediately before ts = hours 8,9,10 = mean(8,9,10) = 9.0
        assert row["rolling_mean_3h"] == pytest.approx(9.0)

    def test_calendar_features_match_timestamp(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="h")  # 2020-01-01 is a Wednesday
        buffer = pd.Series(range(5), index=idx, dtype=float)
        ts = pd.Timestamp("2020-01-04 10:00:00")  # a Saturday

        row = _build_feature_row(
            buffer, ts, lags=[], rolling_windows=[],
            calendar_features=True, holiday_features=False, holidays=None,
        )

        assert row["hour"] == 10
        assert row["day_of_week"] == 5  # Saturday
        assert row["month"] == 1
        assert row["is_weekend"] == 1

    def test_holiday_features_use_precomputed_holidays(self):
        idx = pd.date_range("2019-07-01", periods=5, freq="h")
        buffer = pd.Series(range(5), index=idx, dtype=float)
        holidays = pd.DatetimeIndex(["2019-07-04"])

        row_on_holiday = _build_feature_row(
            buffer, pd.Timestamp("2019-07-04 12:00:00"), lags=[], rolling_windows=[],
            calendar_features=False, holiday_features=True, holidays=holidays,
        )
        row_off_holiday = _build_feature_row(
            buffer, pd.Timestamp("2019-07-03 12:00:00"), lags=[], rolling_windows=[],
            calendar_features=False, holiday_features=True, holidays=holidays,
        )

        assert row_on_holiday["is_holiday"] == 1
        assert row_on_holiday["days_to_nearest_holiday"] == 0
        assert row_off_holiday["is_holiday"] == 0
        assert row_off_holiday["days_to_nearest_holiday"] == 1

    def test_missing_history_produces_nan_lag(self):
        """A lag reaching before the buffer's start is NaN, not a KeyError —
        matches pd.Series.shift()'s behavior in the bulk pipeline."""
        idx = pd.date_range("2020-01-01", periods=3, freq="h")
        buffer = pd.Series(range(3), index=idx, dtype=float)
        ts = idx[0] + pd.Timedelta(hours=1)

        row = _build_feature_row(
            buffer, ts, lags=[10], rolling_windows=[],
            calendar_features=False, holiday_features=False, holidays=None,
        )
        assert np.isnan(row["lag_10h"])


class TestRecursiveForecast:
    def test_output_length_and_index_continue_from_history(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        history = pd.Series(range(10), index=idx, dtype=float)

        result = recursive_forecast(
            _IdentityLag1Model(), history, horizon=5, feature_columns=["lag_1h"],
            lags=[1], rolling_windows=[], calendar_features=False, holiday_features=False,
        )

        assert len(result) == 5
        assert result.index[0] == idx[-1] + pd.Timedelta(hours=1)
        assert list(result.index) == list(pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=5, freq="h"))

    def test_identity_model_holds_the_last_value_constant(self):
        """With a model that just echoes lag_1h, every future prediction equals
        the last real value forever — a fully hand-verifiable recursive chain:
        step 1 predicts history[-1]; step 2's lag_1h is step 1's prediction
        (still history[-1]); and so on."""
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        history = pd.Series(range(10), index=idx, dtype=float)  # ends at value 9.0

        result = recursive_forecast(
            _IdentityLag1Model(), history, horizon=4, feature_columns=["lag_1h"],
            lags=[1], rolling_windows=[], calendar_features=False, holiday_features=False,
        )

        assert list(result.values) == pytest.approx([9.0, 9.0, 9.0, 9.0])

    def test_feature_columns_order_is_respected(self):
        """The model must receive columns in exactly feature_columns order —
        a model fit on [lag_1h, hour] would silently mispredict if fed
        [hour, lag_1h] instead, since sklearn estimators are positional."""
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        history = pd.Series(range(10), index=idx, dtype=float)

        class _ColumnOrderCheckingModel:
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                assert list(X.columns) == ["lag_1h", "hour"], f"got {list(X.columns)}"
                return np.array([0.0])

        recursive_forecast(
            _ColumnOrderCheckingModel(), history, horizon=1, feature_columns=["lag_1h", "hour"],
            lags=[1], rolling_windows=[], calendar_features=True, holiday_features=False,
        )
