"""Tests for live finance model serving: per-asset multi-month forecasting
(reusing Plan 4's reload contract) and the latest-known-feature-row lookup
for cross_sectional_gbm serving."""
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pytest
import yaml

from src.finance.model_registry import fit_arima, fit_mean, fit_random_walk
from src.finance.rank_ic import reload_model
from src.finance.serve_finance import forecast_per_asset_model, load_latest_asset_features


def _log_and_reload(mlflow_uri: str, experiment: str, model_type: str, fitted) -> object:
    """Round-trips a fitted model through a real MLflow store and reloads it
    via Plan 4's reload_model — forecast_per_asset_model must work on
    exactly what reload_model returns, not a freshly-fitted object (these
    are not always identical in capability, per Plan 3/4's own precedent)."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        if model_type in ("random_walk", "mean"):
            mlflow.pyfunc.log_model(python_model=fitted, name="model")
        else:
            mlflow.statsmodels.log_model(fitted, name="model")
        run_id = run.info.run_id
    return reload_model(model_type, run_id, mlflow_uri)


class TestForecastPerAssetModel:
    def test_random_walk_forecasts_all_zeros(self, tmp_path):
        y = pd.Series([0.01, 0.02, 0.03, 0.04])
        fitted = fit_random_walk(y)
        loaded = _log_and_reload(f"sqlite:///{tmp_path / 'mlflow.db'}", "test_serve_finance_rw", "random_walk", fitted)

        result = forecast_per_asset_model(loaded, "random_walk", horizon_months=3)

        assert len(result) == 3
        assert list(result) == pytest.approx([0.0, 0.0, 0.0])

    def test_mean_forecasts_the_historical_average(self, tmp_path):
        y = pd.Series([0.01, 0.02, 0.03, 0.04])
        fitted = fit_mean(y, {"window": None})
        loaded = _log_and_reload(f"sqlite:///{tmp_path / 'mlflow.db'}", "test_serve_finance_mean", "mean", fitted)

        result = forecast_per_asset_model(loaded, "mean", horizon_months=2)

        assert len(result) == 2
        assert list(result) == pytest.approx([0.025, 0.025])

    def test_arima_forecasts_horizon_months_ahead(self, tmp_path):
        y = pd.Series([0.01, 0.02, 0.015, 0.03, 0.025, 0.035, 0.03, 0.04])
        fitted = fit_arima(y, {"order": [1, 0, 0]})
        loaded = _log_and_reload(f"sqlite:///{tmp_path / 'mlflow.db'}", "test_serve_finance_arima", "arima", fitted)

        result = forecast_per_asset_model(loaded, "arima", horizon_months=4)

        assert len(result) == 4


class TestLoadLatestAssetFeatures:
    def test_returns_none_when_no_runs_exist(self, tmp_path):
        assert load_latest_asset_features(tmp_path / "features", "AAPL", ["lag_1_return"]) is None

    def test_returns_none_when_ticker_not_in_latest_manifest(self, tmp_path):
        features_dir = tmp_path / "features"
        run_path = features_dir / "2026-09-23"
        run_path.mkdir(parents=True)
        pd.DataFrame({"log_return": [0.01], "ticker_encoded": [0]}).to_parquet(run_path / "train.parquet", index=False)
        with open(run_path / "manifest.yaml", "w") as f:
            yaml.dump({"ticker_mapping": {"AAPL": 0}}, f)

        assert load_latest_asset_features(features_dir, "MSFT", ["log_return"]) is None

    def test_constructs_the_next_unrealized_period_row_for_the_requested_ticker(self, tmp_path):
        """AAPL's realized returns (across train+test, in chronological
        order) are 0.01 through 0.12 (11 train rows + 1 test row of 0.12 —
        see fixture below; 12 values total so trailing_12m_vol's window is
        satisfiable). The NEXT (unrealized) month's lag_1_return must equal
        the most recent REALIZED return (0.12, the test row's own TARGET
        value), not that row's own lag_1_return column (0.11, a stale
        value pre-computed for THAT row's own month)."""
        features_dir = tmp_path / "features"
        run_path = features_dir / "2026-09-23"
        run_path.mkdir(parents=True)
        aapl_train_returns = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
        n_aapl_train = len(aapl_train_returns)
        train_df = pd.DataFrame({
            "log_return": aapl_train_returns + [0.20, 0.21],
            "lag_1_return": [0.0] + aapl_train_returns[:-1] + [0.0, 0.20],
            "trailing_12m_vol": [0.05] * n_aapl_train + [0.06, 0.06],
            "ticker_encoded": [0] * n_aapl_train + [1, 1],
            "date_ordinal": list(range(700, 700 + n_aapl_train)) + [700, 701],
        })
        test_df = pd.DataFrame({
            "log_return": [0.12],
            "lag_1_return": [0.11],
            "trailing_12m_vol": [0.05],
            "ticker_encoded": [0],
            "date_ordinal": [700 + n_aapl_train],
        })
        train_df.to_parquet(run_path / "train.parquet", index=False)
        test_df.to_parquet(run_path / "test.parquet", index=False)
        with open(run_path / "manifest.yaml", "w") as f:
            yaml.dump({"ticker_mapping": {"AAPL": 0, "MSFT": 1}}, f)

        feature_columns = ["lag_1_return", "trailing_12m_vol"]
        result = load_latest_asset_features(features_dir, "AAPL", feature_columns, target_col="log_return")

        assert result is not None
        assert list(result.columns) == feature_columns
        # AAPL's realized returns in chronological order: 0.01 ... 0.11
        # (train rows) then 0.12 (the test row). The next unrealized
        # month's lag_1_return is the MOST RECENT realized value: 0.12,
        # not 0.11 (the old, buggy "last row's own lag" value).
        assert result["lag_1_return"].iloc[0] == pytest.approx(0.12)

    def test_returns_none_when_not_enough_history_for_a_requested_lag(self, tmp_path):
        features_dir = tmp_path / "features"
        run_path = features_dir / "2026-09-23"
        run_path.mkdir(parents=True)
        train_df = pd.DataFrame({
            "log_return": [0.01],
            "ticker_encoded": [0],
            "date_ordinal": [700],
        })
        train_df.to_parquet(run_path / "train.parquet", index=False)
        with open(run_path / "manifest.yaml", "w") as f:
            yaml.dump({"ticker_mapping": {"AAPL": 0}}, f)

        # lag_3_return needs 3 realized values; only 1 exists.
        result = load_latest_asset_features(features_dir, "AAPL", ["lag_3_return"], target_col="log_return")
        assert result is None
