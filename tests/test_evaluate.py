"""Tests for the model evaluation gate and MLflow registration."""
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluate import _check_thresholds, register_models_to_mlflow
from src.utils.config import EvaluationConfig


@pytest.fixture
def config_dir(tmp_path):
    """Minimal models.yaml with evaluation thresholds for gate tests."""
    cfg = {
        "evaluation": {"min_test_r2": 0.5, "max_test_rmse": 1.0},
        "models": [{"name": "ridge_baseline", "type": "ridge"}],
    }
    (tmp_path / "models.yaml").write_text(yaml.dump(cfg))
    return tmp_path


@pytest.fixture
def config_dir_with_pipeline(tmp_path):
    """models.yaml + pipeline.yaml, needed for tests that exercise the benchmark path."""
    models_cfg = {
        "evaluation": {"min_test_r2": 0.5, "max_test_rmse": 1.0},
        "models": [{"name": "ridge_baseline", "type": "ridge"}],
    }
    pipeline_cfg = {
        "sources": [{"name": "src", "path": "data.csv"}],
        "target": {"name": "target", "type": "continuous"},
        "problem_type": "regression",
        "benchmark": {"enabled": True},
    }
    (tmp_path / "models.yaml").write_text(yaml.dump(models_cfg))
    (tmp_path / "pipeline.yaml").write_text(yaml.dump(pipeline_cfg))
    return tmp_path


def _make_benchmark_dir(tmp_path, rows: int = 50) -> Path:
    """A benchmark dir with one already-created version, ready for load_current_benchmark()."""
    from src.benchmark import create_benchmark_snapshot

    features_dir = tmp_path / "features"
    run_dir = features_dir / "bench-run"
    run_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "f1": list(range(rows)),
        "target": [float(i) for i in range(rows)],
    })
    df.to_parquet(run_dir / "train.parquet")

    benchmark_dir = tmp_path / "benchmark"
    create_benchmark_snapshot(features_dir, "bench-run", benchmark_dir)
    return benchmark_dir


def _mock_run(metrics: dict) -> MagicMock:
    run = MagicMock()
    run.data.metrics = metrics
    run.data.tags = {
        "model_type": "ridge",
        "pipeline_type": "test_pipeline",
        "run_id": "2026-07-02",
    }
    return run


class TestCheckThresholds:
    """Unit tests for the pure threshold-comparison function."""

    def test_passes_when_no_thresholds_configured(self):
        cfg = EvaluationConfig(min_test_r2=None, max_test_rmse=None)
        assert _check_thresholds("m", test_r2=-5.0, test_rmse=999.0, cfg=cfg) is None

    def test_rejects_below_min_r2(self):
        cfg = EvaluationConfig(min_test_r2=0.5, max_test_rmse=None)
        reason = _check_thresholds("m", test_r2=0.2, test_rmse=0.1, cfg=cfg)
        assert reason is not None and "test_r2" in reason

    def test_rejects_above_max_rmse(self):
        cfg = EvaluationConfig(min_test_r2=None, max_test_rmse=0.5)
        reason = _check_thresholds("m", test_r2=0.9, test_rmse=0.6, cfg=cfg)
        assert reason is not None and "test_rmse" in reason

    def test_passes_within_both_thresholds(self):
        cfg = EvaluationConfig(min_test_r2=0.5, max_test_rmse=1.0)
        assert _check_thresholds("m", test_r2=0.8, test_rmse=0.3, cfg=cfg) is None

    def test_missing_r2_skips_r2_check(self):
        """A None test_r2 should not raise or auto-reject — only the RMSE check applies."""
        cfg = EvaluationConfig(min_test_r2=0.5, max_test_rmse=None)
        assert _check_thresholds("m", test_r2=None, test_rmse=0.3, cfg=cfg) is None


class TestRegisterModelsToMlflow:
    """Tests for the registration gate, using a mocked mlflow module."""

    def test_raises_without_run_ids(self, config_dir):
        with pytest.raises(ValueError, match="mlflow_run_ids required"):
            register_models_to_mlflow(mlflow_run_ids=None, config_dir=config_dir)

    @patch("src.evaluate.mlflow")
    def test_rejects_model_below_threshold_and_writes_report(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.1, "train_rmse": 0.1, "train_r2": 0.3}
        )
        reports_dir = tmp_path / "reports"

        with pytest.raises(ValueError, match="rejected by evaluation thresholds"):
            register_models_to_mlflow(
                mlflow_run_ids={"ridge_baseline": "run123"},
                config_dir=config_dir,
                run_id="2026-07-02",
                reports_dir=reports_dir,
            )

        mock_mlflow.register_model.assert_not_called()
        report = yaml.safe_load((reports_dir / "2026-07-02_evaluation.yaml").read_text())
        assert report["models"]["ridge_baseline"]["status"] == "rejected"

    @patch("src.evaluate.mlflow")
    def test_registers_model_passing_thresholds(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = client

        result = register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-02",
            reports_dir=tmp_path / "reports",
        )

        assert result["registered_models"]["ridge_baseline"]["status"] == "registered"
        client.transition_model_version_stage.assert_called_once_with(
            name="ridge_baseline", version="1", stage="Staging"
        )

    @patch("src.evaluate.mlflow")
    def test_raises_when_all_models_rejected(self, mock_mlflow, config_dir, tmp_path):
        """A run where nothing passes the gate must fail the task, not go green silently."""
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 5.0, "test_r2": -1.0, "train_rmse": 0.1, "train_r2": 0.3}
        )

        with pytest.raises(ValueError, match=r"All 1 model\(s\) rejected"):
            register_models_to_mlflow(
                mlflow_run_ids={"ridge_baseline": "run123"},
                config_dir=config_dir,
                run_id="2026-07-02",
                reports_dir=tmp_path / "reports",
            )
        mock_mlflow.register_model.assert_not_called()

    @patch("src.evaluate.mlflow")
    def test_raises_on_missing_test_rmse(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run({"test_r2": 0.8})

        with pytest.raises(RuntimeError, match="Registration failed"):
            register_models_to_mlflow(
                mlflow_run_ids={"ridge_baseline": "run123"},
                config_dir=config_dir,
                run_id="2026-07-02",
                reports_dir=tmp_path / "reports",
            )
        mock_mlflow.register_model.assert_not_called()

    @patch("src.evaluate.mlflow")
    def test_nan_test_r2_fails_loudly_instead_of_bypassing_gate(self, mock_mlflow, config_dir, tmp_path):
        """A NaN test_r2 must not silently pass the < threshold comparison."""
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": math.nan, "train_rmse": 0.1, "train_r2": 0.3}
        )

        with pytest.raises(RuntimeError, match="Registration failed"):
            register_models_to_mlflow(
                mlflow_run_ids={"ridge_baseline": "run123"},
                config_dir=config_dir,
                run_id="2026-07-02",
                reports_dir=tmp_path / "reports",
            )
        mock_mlflow.register_model.assert_not_called()


class TestRunChampion:
    """Tests for tagging the best-performing model of a run."""

    @patch("src.evaluate.mlflow")
    def test_tags_the_lowest_rmse_model_as_champion(self, mock_mlflow, config_dir, tmp_path):
        def fake_get_run(run_id):
            metrics_by_run = {
                "run_a": {"test_rmse": 0.5, "test_r2": 0.6, "train_rmse": 0.4, "train_r2": 0.65},
                "run_b": {"test_rmse": 0.2, "test_r2": 0.9, "train_rmse": 0.15, "train_r2": 0.92},
            }
            return _mock_run(metrics_by_run[run_id])

        mock_mlflow.get_run.side_effect = fake_get_run
        mock_mlflow.register_model.side_effect = lambda model_uri, name: MagicMock(version="1")
        client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = client

        # Two distinct model names so each takes its own path through the loop
        cfg = {
            "evaluation": {"min_test_r2": 0.0, "max_test_rmse": None},
            "models": [{"name": "model_a", "type": "ridge"}, {"name": "model_b", "type": "ridge"}],
        }
        multi_config_dir = tmp_path / "multi_config"
        multi_config_dir.mkdir()
        (multi_config_dir / "models.yaml").write_text(yaml.dump(cfg))

        register_models_to_mlflow(
            mlflow_run_ids={"model_a": "run_a", "model_b": "run_b"},
            config_dir=multi_config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert report["run_champion"] == "model_b"  # lower test_rmse (0.2 < 0.5)


class TestRegressionVsProduction:
    """Tests for the benchmark-based statistical regression check."""

    @patch("src.evaluate.mlflow")
    def test_skipped_when_benchmark_dir_not_provided(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "regression_vs_production" not in report["models"]["ridge_baseline"]

    @patch("src.evaluate.mlflow")
    def test_skipped_when_no_production_version_exists(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        client = MagicMock()
        client.get_latest_versions.return_value = []  # nothing in Production yet
        mock_mlflow.tracking.MlflowClient.return_value = client

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=_make_benchmark_dir(tmp_path),
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "regression_vs_production" not in report["models"]["ridge_baseline"]

    @patch("src.evaluate.mlflow")
    def test_flags_regression_when_candidate_is_significantly_worse(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="2")
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="1")]
        mock_mlflow.tracking.MlflowClient.return_value = client

        benchmark_dir = _make_benchmark_dir(tmp_path, rows=50)

        production_model = MagicMock()
        production_model.predict.return_value = list(range(50))  # perfect on the benchmark target
        candidate_model = MagicMock()
        candidate_model.predict.return_value = [i + 10 for i in range(50)]  # way off
        mock_mlflow.pyfunc.load_model.side_effect = [production_model, candidate_model]

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=benchmark_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        model_report = report["models"]["ridge_baseline"]
        assert model_report["regression_vs_production"] is True
        assert "production_rmse_ci" in model_report
        assert "candidate_rmse_ci" in model_report

    @patch("src.evaluate.mlflow")
    def test_no_regression_flag_when_cis_overlap(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="2")
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="1")]
        mock_mlflow.tracking.MlflowClient.return_value = client

        benchmark_dir = _make_benchmark_dir(tmp_path, rows=50)

        # Both models get the SAME magnitude of random noise applied independently —
        # genuinely similar performance, not a constant offset (a constant offset would
        # give both a near-zero-width CI that trivially wouldn't overlap, which would
        # defeat the point of this test — real variance is required for a fair overlap check).
        rng = np.random.default_rng(0)
        production_model = MagicMock()
        production_model.predict.return_value = np.arange(50, dtype=float) + rng.normal(0, 1.0, 50)
        candidate_model = MagicMock()
        candidate_model.predict.return_value = np.arange(50, dtype=float) + rng.normal(0, 1.0, 50)
        mock_mlflow.pyfunc.load_model.side_effect = [production_model, candidate_model]

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=benchmark_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert report["models"]["ridge_baseline"]["regression_vs_production"] is False

    @patch("src.evaluate.mlflow")
    def test_production_load_failure_does_not_block_registration(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="2")
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="1")]
        mock_mlflow.tracking.MlflowClient.return_value = client
        mock_mlflow.pyfunc.load_model.side_effect = RuntimeError("registry unavailable")

        result = register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=_make_benchmark_dir(tmp_path),
        )

        assert result["registered_models"]["ridge_baseline"]["status"] == "registered"
        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "regression_vs_production" not in report["models"]["ridge_baseline"]


class TestDriftContext:
    """Tests for attaching drift_detected context to the evaluation report."""

    @patch("src.evaluate.compute_drift_detected")
    @patch("src.evaluate.mlflow")
    def test_attaches_drift_detected_when_features_dir_provided(self, mock_mlflow, mock_compute_drift, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()
        mock_compute_drift.return_value = True

        features_dir = tmp_path / "features"
        for run_id in ["2026-07-02", "2026-07-03"]:
            run_dir = features_dir / run_id
            run_dir.mkdir(parents=True)
            pd.DataFrame({"a": [1.0, 2.0], "target": [0.1, 0.2]}).to_parquet(run_dir / "train.parquet")

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            features_dir=features_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert report["models"]["ridge_baseline"]["drift_detected"] is True

    @patch("src.evaluate.mlflow")
    def test_no_drift_field_when_features_dir_not_provided(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "drift_detected" not in report["models"]["ridge_baseline"]

    @patch("src.evaluate.mlflow")
    def test_no_drift_field_when_no_previous_run_exists(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-03"
        run_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1.0], "target": [0.1]}).to_parquet(run_dir / "train.parquet")

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            features_dir=features_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "drift_detected" not in report["models"]["ridge_baseline"]
