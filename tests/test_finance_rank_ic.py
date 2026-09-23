"""Tests for shared finance evaluation utilities: cross-sectional rank
correlation (Spearman IC) and the per-model-type MLflow reload contract
Plan 3 locked in (pyfunc for random_walk/mean, statsmodels for arima,
lightgbm for cross_sectional_gbm)."""
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pytest

from src.finance.model_registry import fit_arima, fit_cross_sectional_gbm, fit_mean, fit_random_walk
from src.finance.rank_ic import reload_model, spearman_ic


class TestSpearmanIc:
    def test_perfect_rank_correlation_gives_ic_near_one(self):
        long_df = pd.DataFrame({
            "month": [1, 1, 1, 1],
            "predicted": [0.01, 0.02, 0.03, 0.04],
            "actual": [0.10, 0.20, 0.30, 0.40],  # same rank order as predicted
        })
        assert spearman_ic(long_df) == pytest.approx(1.0)

    def test_inverse_rank_correlation_gives_ic_near_negative_one(self):
        long_df = pd.DataFrame({
            "month": [1, 1, 1, 1],
            "predicted": [0.01, 0.02, 0.03, 0.04],
            "actual": [0.40, 0.30, 0.20, 0.10],  # exactly reversed rank order
        })
        assert spearman_ic(long_df) == pytest.approx(-1.0)

    def test_averages_across_multiple_months(self):
        long_df = pd.DataFrame({
            "month": [1, 1, 1, 2, 2, 2],
            "predicted": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
            "actual": [0.10, 0.20, 0.30, 0.30, 0.20, 0.10],  # month 1: +1.0, month 2: -1.0
        })
        assert spearman_ic(long_df) == pytest.approx(0.0)

    def test_degenerate_month_with_constant_predictions_is_excluded(self):
        """A month where every predicted value is identical (e.g. Random Walk's
        constant-0 forecast) has an undefined correlation (NaN) - it must be
        excluded from the average, not treated as 0 or crash."""
        long_df = pd.DataFrame({
            "month": [1, 1, 1, 2, 2, 2],
            "predicted": [0.0, 0.0, 0.0, 0.01, 0.02, 0.03],  # month 1 degenerate
            "actual": [0.10, 0.20, 0.30, 0.10, 0.20, 0.30],
        })
        # Only month 2 (perfect correlation) counts -> IC == 1.0, not averaged with a 0 for month 1.
        assert spearman_ic(long_df) == pytest.approx(1.0)

    def test_returns_nan_when_no_valid_months(self):
        long_df = pd.DataFrame({
            "month": [1, 1, 1],
            "predicted": [0.0, 0.0, 0.0],
            "actual": [0.10, 0.20, 0.30],
        })
        assert np.isnan(spearman_ic(long_df))


class TestReloadModel:
    """Round-trips a real fitted model through a real MLflow log/load cycle -
    each reload path has a genuinely different call convention (Plan 3's
    final review found and documented this), so a hand-built fixture would
    not exercise the real risk here."""

    def test_reloads_random_walk_as_working_pyfunc(self, tmp_path):
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_reload_random_walk")
        y = pd.Series([0.01, 0.02, -0.01, 0.03])
        fitted = fit_random_walk(y)
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(python_model=fitted, name="model")
            run_id = run.info.run_id

        loaded = reload_model("random_walk", run_id, mlflow_uri)
        predictions = loaded.predict(pd.DataFrame({"_dummy": range(3)}))
        assert list(predictions) == pytest.approx([0.0, 0.0, 0.0])

    def test_reloads_mean_as_working_pyfunc(self, tmp_path):
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_reload_mean")
        y = pd.Series([0.01, 0.02, 0.03, 0.04])
        fitted = fit_mean(y, {"window": None})
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(python_model=fitted, name="model")
            run_id = run.info.run_id

        loaded = reload_model("mean", run_id, mlflow_uri)
        predictions = loaded.predict(pd.DataFrame({"_dummy": range(2)}))
        assert list(predictions) == pytest.approx([0.025, 0.025])

    def test_reloads_arima_as_working_statsmodels_forecast(self, tmp_path):
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_reload_arima")
        rng = np.random.default_rng(42)
        y = pd.Series(rng.normal(0.01, 0.05, 36))
        fitted = fit_arima(y, {"order": [1, 0, 0]})
        with mlflow.start_run() as run:
            mlflow.statsmodels.log_model(fitted, name="model")
            run_id = run.info.run_id

        loaded = reload_model("arima", run_id, mlflow_uri)
        forecast = loaded.forecast(steps=4)
        assert len(forecast) == 4

    def test_reloads_cross_sectional_gbm_as_working_lightgbm(self, tmp_path):
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_reload_gbm")
        panel_df = pd.DataFrame({
            "log_return": [0.01, 0.02, -0.01, 0.03, 0.00, 0.015] * 5,
            "lag_1_return": [0.01, 0.00, 0.02, -0.01, 0.01, 0.02] * 5,
            "trailing_12m_vol": [0.02] * 30,
            "ticker_encoded": ([0] * 6 + [1] * 6 + [0] * 6 + [1] * 6 + [0] * 6),
        })
        feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
        model = fit_cross_sectional_gbm(panel_df, "log_return", feature_columns, {"n_estimators": 5, "max_depth": 2, "random_state": 42})
        with mlflow.start_run() as run:
            mlflow.lightgbm.log_model(model, name="model")
            run_id = run.info.run_id

        loaded = reload_model("cross_sectional_gbm", run_id, mlflow_uri)
        predictions = loaded.predict(panel_df[feature_columns])
        assert len(predictions) == len(panel_df)

    def test_raises_for_unknown_model_type(self, tmp_path):
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        with pytest.raises(ValueError, match="Unknown finance model type"):
            reload_model("not_a_real_type", "fake_run_id", mlflow_uri)
