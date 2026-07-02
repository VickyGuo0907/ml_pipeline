"""Tests for the model evaluation gate and MLflow registration."""
import math
from unittest.mock import MagicMock, patch

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
