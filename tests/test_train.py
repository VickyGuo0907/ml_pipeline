"""Tests for model training metrics, including bootstrapped test-metric CIs."""
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.train import _log_metrics


@pytest.fixture(autouse=True)
def local_mlflow_tracking(tmp_path):
    """Point mlflow at a local sqlite store so _log_metrics' mlflow.log_metric calls
    have somewhere to write, without needing a real tracking server. (The file-store
    backend is in maintenance mode and blocked by default in this mlflow version.)"""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test_train_metrics")
    with mlflow.start_run():
        yield


def _fitted_model_and_splits():
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    y_train = X_train["x1"] * 2 + X_train["x2"] * 0.5 + rng.normal(scale=0.1, size=200)
    X_test = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60)})
    y_test = X_test["x1"] * 2 + X_test["x2"] * 0.5 + rng.normal(scale=0.1, size=60)

    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


class TestLogMetricsBootstrapCI:
    """Tests for the bootstrapped 95% CIs added to _log_metrics' return value."""

    def test_returns_rmse_and_r2_ci_bounds(self):
        model, X_train, y_train, X_test, y_test = _fitted_model_and_splits()

        metrics, _ = _log_metrics(model, X_train, y_train, X_test, y_test)

        for key in ("test_rmse_ci_lower", "test_rmse_ci_upper", "test_r2_ci_lower", "test_r2_ci_upper"):
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_rmse_ci_bounds_are_ordered_and_bracket_the_point_estimate_reasonably(self):
        model, X_train, y_train, X_test, y_test = _fitted_model_and_splits()

        metrics, _ = _log_metrics(model, X_train, y_train, X_test, y_test)

        assert metrics["test_rmse_ci_lower"] <= metrics["test_rmse_ci_upper"]
        assert metrics["test_rmse_ci_lower"] >= 0  # RMSE is non-negative
        # A 95% bootstrap CI over the same test set should contain the point estimate.
        assert metrics["test_rmse_ci_lower"] <= metrics["test_rmse"] <= metrics["test_rmse_ci_upper"]

    def test_r2_ci_bounds_are_ordered_and_bracket_the_point_estimate_reasonably(self):
        model, X_train, y_train, X_test, y_test = _fitted_model_and_splits()

        metrics, _ = _log_metrics(model, X_train, y_train, X_test, y_test)

        assert metrics["test_r2_ci_lower"] <= metrics["test_r2_ci_upper"]
        assert metrics["test_r2_ci_lower"] <= metrics["test_r2"] <= metrics["test_r2_ci_upper"]

    def test_ci_metrics_are_logged_to_mlflow(self):
        model, X_train, y_train, X_test, y_test = _fitted_model_and_splits()

        _log_metrics(model, X_train, y_train, X_test, y_test)

        run = mlflow.active_run()
        logged = mlflow.get_run(run.info.run_id).data.metrics
        assert "test_rmse_ci_lower" in logged
        assert "test_r2_ci_upper" in logged
