"""Tests for model training metrics, including bootstrapped test-metric CIs."""
import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.train import _log_metrics, train_models


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

    @pytest.fixture(autouse=True)
    def local_mlflow_tracking(self, tmp_path):
        """Point mlflow at a local sqlite store and hold an active run open, since
        _log_metrics assumes a run is already active (as train_models normally
        provides). Scoped to this class only — TestTrainModelsLogsServingMetadata
        below calls train_models() directly, which opens its own run via
        `with mlflow.start_run()` and would collide with an already-active one."""
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.set_experiment("test_train_metrics")
        with mlflow.start_run():
            yield

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


class TestTrainModelsLogsServingMetadata:
    """Tests that train_models logs the exact trained schema src/serve.py needs to
    build a dynamic, pipeline-agnostic request schema (no hardcoded feature list)."""

    def test_logs_feature_columns_artifact_and_target_col_param(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-01"
        run_dir.mkdir(parents=True)

        train_df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        test_df = pd.DataFrame({
            "f1": [7.0, 8.0],
            "f2": [0.7, 0.8],
            "target": [7.0, 8.0],
        })
        train_df.to_parquet(run_dir / "train.parquet")
        test_df.to_parquet(run_dir / "test.parquet")

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text("""
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: target
  type: continuous
problem_type: regression
pipeline_type: test_pipeline
""")
        (config_dir / "models.yaml").write_text("""
models:
  - name: test_pipeline_ridge
    type: ridge
    hyperparameters: {}
random_state: 42
""")

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        # Explicitly bind the experiment against this test's own fresh sqlite db —
        # otherwise mlflow's fluent API can carry over a stale active-experiment-id
        # cached from a previous test's tracking URI in the same pytest process.
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_train_models_serving_metadata")
        result = train_models(features_dir, "2026-07-01", config_dir, mlflow_tracking_uri=mlflow_uri)

        run_id = result["models"]["test_pipeline_ridge"]["mlflow_run_id"]
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        run = client.get_run(run_id)

        assert run.data.params.get("target_col") == "target"

        local_path = client.download_artifacts(run_id, "feature_columns.json")
        with open(local_path) as f:
            artifact = json.load(f)
        assert artifact["columns"] == ["f1", "f2"]
