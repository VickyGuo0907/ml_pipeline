"""Tests for the forecasting training stage: rolling-origin CV scoring and
MLflow logging for ETS, SARIMAX, and GBM."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import mlflow

from src.forecasting.train_forecast import train_forecast_models


def _synthetic_train_df(n_hours: int = 200, seasonal_period: int = 24) -> pd.DataFrame:
    """A daily-seasonal hourly frame with the exact feature set
    _build_feature_row(calendar_features=True, holiday_features=False, ...)
    would produce, long enough for ETS/SARIMAX to fit and for several
    rolling-origin windows.

    Deliberately mirrors engineer_forecast_features' calendar block exactly
    (hour, day_of_week, month, is_weekend together, since calendar_features
    is one on/off flag, not four independent ones) — a partial column set
    here would desync from what recursive_forecast's per-step feature
    builder produces during GBM's rolling-origin scoring, raising a KeyError
    the moment feature_columns includes a column _build_feature_row doesn't
    generate for the configured flags.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h", name="Datetime")
    target = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / seasonal_period) + rng.normal(0, 1, n_hours)
    target = pd.Series(target, index=idx)
    df = pd.DataFrame({"PJME_MW": target})
    df["lag_1h"] = df["PJME_MW"].shift(1)
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df.dropna()


def _write_forecast_config(
    config_dir: Path,
    models: list[dict],
    horizon_hours: int = 4,
    n_windows: int = 2,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: PJME_MW\n  type: continuous\nproblem_type: forecasting\n"
        "pipeline_type: test_forecast\n"
    )
    lags = lags if lags is not None else [1]
    rolling_windows = rolling_windows if rolling_windows is not None else []
    (config_dir / "features.yaml").write_text(
        f"lags: {lags}\nrolling_windows: {rolling_windows}\n"
        "calendar_features: true\nholiday_features: false\n"
    )
    models_yaml = "models:\n"
    for m in models:
        models_yaml += f"  - name: {m['name']}\n    type: {m['type']}\n    hyperparameters: {m['hyperparameters']}\n"
    models_yaml += f"evaluation:\n  horizon_hours: {horizon_hours}\n  n_windows: {n_windows}\n"
    (config_dir / "models.yaml").write_text(models_yaml)


class TestTrainForecastModels:
    def test_trains_all_three_model_types_and_logs_cv_mape(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-09-16"
        run_dir.mkdir(parents=True)

        train_df = _synthetic_train_df(n_hours=200)
        train_df.to_parquet(run_dir / "train.parquet")
        train_df.tail(20).to_parquet(run_dir / "test.parquet")  # unused by train stage, but present

        config_dir = tmp_path / "config"
        _write_forecast_config(
            config_dir,
            models=[
                {"name": "test_ets", "type": "ets", "hyperparameters": "{seasonal_periods: 24, trend: add, seasonal: add}"},
                {"name": "test_sarimax", "type": "sarimax", "hyperparameters": "{order: [1, 0, 0], seasonal_order: [1, 0, 0, 24]}"},
                {"name": "test_gbm", "type": "gbm", "hyperparameters": "{n_estimators: 10, max_depth: 3, random_state: 42}"},
            ],
            horizon_hours=4, n_windows=2,
            lags=[1], rolling_windows=[],
        )

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_train_forecast_models")

        result = train_forecast_models(features_dir, "2026-09-16", config_dir, mlflow_tracking_uri=mlflow_uri)

        assert set(result["models"]) == {"test_ets", "test_sarimax", "test_gbm"}
        for name, info in result["models"].items():
            assert "mlflow_run_id" in info
            assert info["cv_mape_mean"] >= 0
            assert info["n_windows_scored"] >= 1

    def test_one_bad_model_does_not_block_the_others(self, tmp_path):
        """A model type that fails to fit (here: an unknown type) is logged
        and skipped, not raised — the other two still train successfully."""
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-09-16"
        run_dir.mkdir(parents=True)

        train_df = _synthetic_train_df(n_hours=200)
        train_df.to_parquet(run_dir / "train.parquet")
        train_df.tail(20).to_parquet(run_dir / "test.parquet")

        config_dir = tmp_path / "config"
        _write_forecast_config(
            config_dir,
            models=[
                {"name": "test_bad", "type": "not_a_real_type", "hyperparameters": "{}"},
                {"name": "test_ets", "type": "ets", "hyperparameters": "{seasonal_periods: 24, trend: add, seasonal: add}"},
            ],
            horizon_hours=4, n_windows=2,
            lags=[1], rolling_windows=[],
        )

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_train_forecast_models_partial_failure")

        result = train_forecast_models(features_dir, "2026-09-16", config_dir, mlflow_tracking_uri=mlflow_uri)

        assert "test_bad" not in result["models"]
        assert "test_ets" in result["models"]

    def test_raises_when_train_parquet_missing(self, tmp_path):
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        _write_forecast_config(config_dir, models=[{"name": "x", "type": "ets", "hyperparameters": "{}"}])

        with pytest.raises(FileNotFoundError):
            train_forecast_models(features_dir, "2026-09-16", config_dir)
