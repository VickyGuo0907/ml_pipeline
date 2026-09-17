"""Tests for live forecast serving: snapshot loading and per-model-family
multi-step forecasting (statsmodels native forecast, GBM recursive)."""
from pathlib import Path

import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pytest

from src.forecasting.model_registry import fit_sarimax
from src.forecasting.serve_forecast import (
    forecast_with_gbm,
    forecast_with_statsmodels,
    load_latest_snapshot,
)


def _synthetic_series(n_hours: int = 200, seasonal_period: int = 24) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period) + rng.normal(0, 1, n_hours)
    return pd.Series(values, index=idx)


class TestLoadLatestSnapshot:
    def test_returns_none_when_no_runs_exist(self, tmp_path):
        assert load_latest_snapshot(tmp_path / "features") is None

    def test_returns_none_when_latest_run_has_no_snapshot(self, tmp_path):
        features_dir = tmp_path / "features"
        (features_dir / "2026-01-01").mkdir(parents=True)
        assert load_latest_snapshot(features_dir) is None

    def test_loads_the_latest_runs_snapshot(self, tmp_path):
        features_dir = tmp_path / "features"
        idx = pd.date_range("2020-01-01", periods=5, freq="h", name="Datetime")
        old_snapshot = pd.DataFrame({"PJME_MW": [1.0] * 5}, index=idx)
        new_snapshot = pd.DataFrame({"PJME_MW": [2.0] * 5}, index=idx)

        old_run = features_dir / "2026-01-01"
        old_run.mkdir(parents=True)
        old_snapshot.to_parquet(old_run / "last_window.parquet")

        new_run = features_dir / "2026-02-01"
        new_run.mkdir(parents=True)
        new_snapshot.to_parquet(new_run / "last_window.parquet")

        result = load_latest_snapshot(features_dir)

        assert result is not None
        assert (result["PJME_MW"] == 2.0).all()


class TestForecastWithStatsmodels:
    def test_forecasts_horizon_hours_ahead_without_exog(self):
        y = _synthetic_series()
        from src.forecasting.model_registry import fit_ets
        fitted = fit_ets(y, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        result = forecast_with_statsmodels(fitted, horizon_hours=12)

        assert len(result) == 12
        assert result.index[0] == y.index[-1] + pd.Timedelta(hours=1)

    def test_forecasts_with_exog_when_model_requires_it(self):
        """The real config/pjm_load_forecast/models.yaml fits SARIMAX with
        use_holiday_exog: true — this is the exact scenario that broke
        Plan 4's evaluation stage until fixed. Serving must handle it too."""
        y = _synthetic_series(n_hours=300)
        # A named Series, matching fit_sarimax's real signature (exog: pd.Series | None)
        # and exactly how src/forecasting/train_forecast.py passes it
        # (exog = train_df["is_holiday"]) — not a DataFrame.
        exog = pd.Series((y.index.dayofweek >= 5).astype(int), index=y.index, name="is_holiday")
        fitted = fit_sarimax(
            y, {"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 24], "use_holiday_exog": True}, exog=exog,
        )

        result = forecast_with_statsmodels(fitted, horizon_hours=12)

        assert len(result) == 12
        assert result.index[0] == y.index[-1] + pd.Timedelta(hours=1)
        assert result.notna().all()

    def test_works_on_a_model_round_tripped_through_mlflow(self, tmp_path):
        """Real regression protection: forecast_with_statsmodels must work on
        the object mlflow.statsmodels.load_model() returns, not just a
        freshly-fitted one — these are not always identical in capability."""
        y = _synthetic_series()
        from src.forecasting.model_registry import fit_ets
        fitted = fit_ets(y, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.set_experiment("test_serve_forecast_roundtrip")
        with mlflow.start_run() as run:
            mlflow.statsmodels.log_model(fitted, name="model")
            run_id = run.info.run_id
        loaded = mlflow.statsmodels.load_model(f"runs:/{run_id}/model")

        result = forecast_with_statsmodels(loaded, horizon_hours=6)
        assert len(result) == 6


class TestForecastWithGbm:
    def test_forecasts_horizon_hours_ahead_from_snapshot(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="h", name="Datetime")
        snapshot = pd.DataFrame({"PJME_MW": range(20)}, index=idx, dtype=float)

        class _IdentityLag1Model:
            def predict(self, X):
                return X["lag_1h"].to_numpy()

        result = forecast_with_gbm(
            _IdentityLag1Model(), snapshot, "PJME_MW", ["lag_1h"], horizon_hours=4,
            lags=[1], rolling_windows=[], calendar_features=False, holiday_features=False,
        )

        assert len(result) == 4
        assert result.index[0] == idx[-1] + pd.Timedelta(hours=1)
        # Identity model on constant lag_1h => every prediction equals the last real value (19.0)
        assert list(result.values) == pytest.approx([19.0, 19.0, 19.0, 19.0])
