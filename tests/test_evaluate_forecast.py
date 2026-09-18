"""Tests for the forecasting evaluation/registration stage: rolling-origin
test-set scoring, champion selection, and MLflow Staging registration."""
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.statsmodels
import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecasting.evaluate_forecast import _select_forecast_champion, register_forecast_models_to_mlflow
from src.forecasting.model_registry import fit_ets, fit_sarimax
from src.utils.model_registry import get_model


def _synthetic_series(n_hours: int, seasonal_period: int = 24, start: str = "2020-01-01") -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=n_hours, freq="h")
    values = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period) + rng.normal(0, 1, n_hours)
    return pd.Series(values, index=idx)


def _write_feature_parquets(features_dir: Path, run_id: str, n_train: int = 200, n_test: int = 40) -> None:
    run_path = features_dir / run_id
    run_path.mkdir(parents=True)
    full = _synthetic_series(n_train + n_test)
    df = pd.DataFrame({"PJME_MW": full})
    df["lag_1h"] = df["PJME_MW"].shift(1)
    df["hour"] = df.index.hour
    df = df.dropna()
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    train_df.to_parquet(run_path / "train.parquet")
    test_df.to_parquet(run_path / "test.parquet")


def _write_forecast_config(config_dir: Path, horizon_hours: int = 4, n_windows: int = 2) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: PJME_MW\n  type: continuous\nproblem_type: forecasting\n"
        "pipeline_type: test_forecast\n"
    )
    (config_dir / "features.yaml").write_text(
        "lags: [1]\nrolling_windows: []\ncalendar_features: true\nholiday_features: false\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n  - name: test_ets\n    type: ets\n    hyperparameters: {}\n"
        f"evaluation:\n  horizon_hours: {horizon_hours}\n  n_windows: {n_windows}\n"
    )


def _train_and_log_one_ets_run(mlflow_uri: str, experiment: str, train_df: pd.DataFrame) -> str:
    """Mimics what Plan 3's train_forecast_models does for one ETS model,
    minimally — this test file scores/registers, it doesn't re-test training."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="test_run_test_ets") as run:
        mlflow.set_tags({"model_name": "test_ets", "model_type": "ets", "run_id": "2026-09-17", "pipeline_type": "test_forecast"})
        fitted = fit_ets(train_df["PJME_MW"], {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})
        mlflow.log_metric("cv_mape_mean", 5.0)
        mlflow.log_metric("cv_mape_std", 1.0)
        mlflow.statsmodels.log_model(fitted, name="model")
        return run.info.run_id


def _write_feature_parquets_with_holiday(
    features_dir: Path, run_id: str, n_train: int = 200, n_test: int = 40,
) -> None:
    """Same as _write_feature_parquets, but adds an is_holiday column — mirrors
    the real config/pjm_load_forecast/features.yaml's holiday_features: true,
    which is what SARIMAX's use_holiday_exog actually consumes (see
    src/forecasting/train_forecast.py's sarimax branch)."""
    run_path = features_dir / run_id
    run_path.mkdir(parents=True)
    full = _synthetic_series(n_train + n_test)
    df = pd.DataFrame({"PJME_MW": full})
    df["lag_1h"] = df["PJME_MW"].shift(1)
    df["hour"] = df.index.hour
    df["is_holiday"] = (df.index.dayofweek >= 5).astype(int)
    df = df.dropna()
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    train_df.to_parquet(run_path / "train.parquet")
    test_df.to_parquet(run_path / "test.parquet")


def _write_forecast_config_sarimax(config_dir: Path, horizon_hours: int = 4, n_windows: int = 2) -> None:
    """Mirrors the real config/pjm_load_forecast/models.yaml + features.yaml
    combination that triggered Finding 1: a sarimax model with
    use_holiday_exog: true, scored against a feature set with
    holiday_features: true."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: PJME_MW\n  type: continuous\nproblem_type: forecasting\n"
        "pipeline_type: test_forecast\n"
    )
    (config_dir / "features.yaml").write_text(
        "lags: [1]\nrolling_windows: []\ncalendar_features: true\nholiday_features: true\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n  - name: test_sarimax\n    type: sarimax\n    hyperparameters:\n"
        "      order: [1, 0, 0]\n      seasonal_order: [0, 0, 0, 0]\n      use_holiday_exog: true\n"
        f"evaluation:\n  horizon_hours: {horizon_hours}\n  n_windows: {n_windows}\n"
    )


def _train_and_log_one_sarimax_run(mlflow_uri: str, experiment: str, train_df: pd.DataFrame) -> str:
    """Mimics what Plan 3's train_forecast_models does for the sarimax branch
    with use_holiday_exog: true — fits with exog=train_df["is_holiday"], the
    exact path that left the registered model unable to be scored
    out-of-sample without also supplying exog (Finding 1)."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="test_run_test_sarimax") as run:
        mlflow.set_tags({"model_name": "test_sarimax", "model_type": "sarimax", "run_id": "2026-09-17", "pipeline_type": "test_forecast"})
        fitted = fit_sarimax(
            train_df["PJME_MW"],
            {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "use_holiday_exog": True},
            exog=train_df["is_holiday"],
        )
        mlflow.log_metric("cv_mape_mean", 5.0)
        mlflow.log_metric("cv_mape_std", 1.0)
        mlflow.statsmodels.log_model(fitted, name="model")
        return run.info.run_id


class TestSelectForecastChampion:
    def test_picks_lowest_metric(self):
        model_metrics = {
            "a": {"cv_mape": 5.0, "test_mape": 6.0},
            "b": {"cv_mape": 3.0, "test_mape": 4.0},
        }
        assert _select_forecast_champion(model_metrics, "cv_mape") == "b"

    def test_falls_back_to_test_mape_when_cv_mape_missing(self):
        model_metrics = {
            "a": {"cv_mape": None, "test_mape": 6.0},
            "b": {"cv_mape": 3.0, "test_mape": 4.0},
        }
        assert _select_forecast_champion(model_metrics, "cv_mape") == "b"  # test_mape: 6.0 vs 4.0, b still wins


class TestRegisterForecastModelsToMlflow:
    def test_scores_and_registers_to_staging(self, tmp_path):
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-17"

        _write_feature_parquets(features_dir, run_id)
        _write_forecast_config(config_dir)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_evaluate_forecast_models")
        mlflow_run_id = _train_and_log_one_ets_run(mlflow_uri, "test_evaluate_forecast_models", train_df)

        result = register_forecast_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids={"test_ets": mlflow_run_id},
            config_dir=config_dir,
            run_id=run_id,
            reports_dir=reports_dir,
            features_dir=features_dir,
        )

        assert "test_ets" in result["registered_models"]
        entry = result["registered_models"]["test_ets"]
        assert entry["status"] == "registered"
        assert entry["stage"] == "Staging"
        assert entry["test_mape_mean"] >= 0

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        versions = client.get_latest_versions("test_ets", stages=["Staging"])
        assert len(versions) == 1
        # Never auto-promoted to Production.
        prod_versions = client.get_latest_versions("test_ets", stages=["Production"])
        assert len(prod_versions) == 0

        report_path = reports_dir / f"{run_id}_evaluation.yaml"
        assert report_path.exists()
        with open(report_path) as f:
            report = yaml.safe_load(f)
        assert report["run_champion"] == "test_ets"
        assert report["models"]["test_ets"]["status"] == "registered"

    def test_raises_when_features_dir_missing(self, tmp_path):
        config_dir = tmp_path / "config"
        _write_forecast_config(config_dir)

        with pytest.raises(ValueError, match="features_dir"):
            register_forecast_models_to_mlflow(
                mlflow_run_ids={"x": "fake_run_id"}, config_dir=config_dir, run_id="2026-09-17",
                reports_dir=tmp_path / "reports", features_dir=None,
            )

    def test_raises_when_no_run_ids_provided(self, tmp_path):
        with pytest.raises(ValueError, match="mlflow_run_ids"):
            register_forecast_models_to_mlflow(mlflow_run_ids=None, features_dir=tmp_path / "features")


class TestRegisterSarimaxWithHolidayExog:
    """Regression test for Finding 1: config/pjm_load_forecast/models.yaml sets
    use_holiday_exog: true and features.yaml sets holiday_features: true, so
    the real sarimax model is fit WITH exog. Scoring it out-of-sample without
    re-supplying exog raised ValueError from statsmodels and the whole model
    registered as status="error", not "registered". The prior test file only
    exercised one exog-free ETS model, which could never have caught this."""

    def test_scores_and_registers_sarimax_fit_with_exog(self, tmp_path):
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-17"

        _write_feature_parquets_with_holiday(features_dir, run_id)
        _write_forecast_config_sarimax(config_dir)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_evaluate_forecast_sarimax_exog")
        mlflow_run_id = _train_and_log_one_sarimax_run(mlflow_uri, "test_evaluate_forecast_sarimax_exog", train_df)

        result = register_forecast_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids={"test_sarimax": mlflow_run_id},
            config_dir=config_dir,
            run_id=run_id,
            reports_dir=reports_dir,
            features_dir=features_dir,
        )

        entry = result["registered_models"]["test_sarimax"]
        assert entry["status"] == "registered"
        assert entry["stage"] == "Staging"
        assert entry["test_mape_mean"] >= 0

        report_path = reports_dir / f"{run_id}_evaluation.yaml"
        with open(report_path) as f:
            report = yaml.safe_load(f)
        assert report["models"]["test_sarimax"]["status"] == "registered"

    def test_sarimax_converged_tag_copied_to_registered_model_version(self, tmp_path):
        """A sarimax_converged tag set on the training run must also land on
        the registered model version, so an operator checking the MLflow
        registry UI for a SARIMAX model doesn't have to separately look up
        its source run to see whether the fit actually converged."""
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-17"

        _write_feature_parquets_with_holiday(features_dir, run_id)
        _write_forecast_config_sarimax(config_dir)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_evaluate_forecast_sarimax_converged_tag")
        with mlflow.start_run(run_name="test_run_test_sarimax") as run:
            mlflow.set_tags({
                "model_name": "test_sarimax", "model_type": "sarimax",
                "run_id": run_id, "pipeline_type": "test_forecast",
                "sarimax_converged": "True",
            })
            fitted = fit_sarimax(
                train_df["PJME_MW"],
                {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "use_holiday_exog": True},
                exog=train_df["is_holiday"],
            )
            mlflow.log_metric("cv_mape_mean", 5.0)
            mlflow.statsmodels.log_model(fitted, name="model")
            mlflow_run_id = run.info.run_id

        register_forecast_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids={"test_sarimax": mlflow_run_id},
            config_dir=config_dir,
            run_id=run_id,
            reports_dir=reports_dir,
            features_dir=features_dir,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        version = client.get_latest_versions("test_sarimax", stages=["Staging"])[0]
        assert version.tags.get("sarimax_converged") == "True"
