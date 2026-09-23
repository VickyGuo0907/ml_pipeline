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

    def test_loads_the_most_recent_row_for_the_requested_ticker(self, tmp_path):
        features_dir = tmp_path / "features"
        run_path = features_dir / "2026-09-23"
        run_path.mkdir(parents=True)
        train_df = pd.DataFrame({
            "log_return": [0.01, 0.02, 0.10, 0.11],
            "lag_1_return": [0.0, 0.01, 0.0, 0.10],
            "trailing_12m_vol": [0.05, 0.05, 0.06, 0.06],
            "ticker_encoded": [0, 0, 1, 1],
            "date_ordinal": [700, 701, 700, 701],
        })
        test_df = pd.DataFrame({
            "log_return": [0.03],
            "lag_1_return": [0.02],
            "trailing_12m_vol": [0.05],
            "ticker_encoded": [0],
            "date_ordinal": [702],
        })
        train_df.to_parquet(run_path / "train.parquet", index=False)
        test_df.to_parquet(run_path / "test.parquet", index=False)
        with open(run_path / "manifest.yaml", "w") as f:
            yaml.dump({"ticker_mapping": {"AAPL": 0, "MSFT": 1}}, f)

        feature_columns = ["lag_1_return", "trailing_12m_vol"]
        result = load_latest_asset_features(features_dir, "AAPL", feature_columns)

        assert result is not None
        assert list(result.columns) == feature_columns
        # AAPL's most recent row is the test-set one (date_ordinal=702, lag_1_return=0.02)
        assert result["lag_1_return"].iloc[0] == pytest.approx(0.02)
