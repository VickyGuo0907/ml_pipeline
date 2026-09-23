"""Tests for the finance training stage: per-asset Random Walk/Mean/ARIMA
runs and one cross-sectional GBM run, each logging a training-period error
std-dev to MLflow."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.finance.train_finance import _training_error_std, train_finance_models


def _synthetic_train_panel(n_per_asset: int = 24) -> tuple[pd.DataFrame, dict[str, int]]:
    """A 2-asset stacked panel with the exact column shape
    engineer_finance_features (Plan 2) produces: log_return, lag_1_return,
    trailing_12m_vol, ticker_encoded, date_ordinal."""
    rng = np.random.default_rng(42)
    ticker_mapping = {"AAPL": 0, "MSFT": 1}
    rows = []
    for ticker, code in ticker_mapping.items():
        returns = rng.normal(0.01, 0.05, n_per_asset)
        for i, r in enumerate(returns):
            rows.append({
                "log_return": r,
                "lag_1_return": returns[i - 1] if i > 0 else 0.0,
                "trailing_12m_vol": 0.05,
                "ticker_encoded": code,
                "date_ordinal": 600 + i,
            })
    return pd.DataFrame(rows), ticker_mapping


def _write_features_fixture(features_dir: Path, run_id: str, train_df: pd.DataFrame, ticker_mapping: dict[str, int]) -> None:
    run_path = features_dir / run_id
    run_path.mkdir(parents=True)
    train_df.to_parquet(run_path / "train.parquet", index=False)
    feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
    manifest = {
        "run_id": run_id,
        "stage": "feature_engineer_finance",
        "feature_columns": feature_columns,
        "index_columns": ["date_ordinal"],
        "ticker_mapping": ticker_mapping,
    }
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)


def _write_finance_config(config_dir: Path, models: list[dict]) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: log_return\n  type: continuous\nproblem_type: finance\n"
        "pipeline_type: test_m6_returns_risk\n"
    )
    models_yaml = "models:\n"
    for m in models:
        models_yaml += f"  - name: {m['name']}\n    type: {m['type']}\n    hyperparameters: {m['hyperparameters']}\n"
    models_yaml += "evaluation:\n  champion_metric: rank_ic\n"
    (config_dir / "models.yaml").write_text(models_yaml)


class TestTrainingErrorStd:
    def test_matches_hand_computed_std_of_residuals(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        fittedvalues = pd.Series([1.0, 1.0, 1.0, 1.0])  # constant prediction
        expected = pd.Series([0.0, 1.0, 2.0, 3.0]).std()  # residuals, pandas default ddof=1
        assert _training_error_std(y, fittedvalues) == pytest.approx(expected)

    def test_nan_fitted_rows_are_excluded(self):
        y = pd.Series([1.0, 2.0, 3.0])
        fittedvalues = pd.Series([np.nan, 2.0, 2.0])
        expected = pd.Series([0.0, 1.0]).std()  # only rows 1,2 have non-NaN fittedvalues
        assert _training_error_std(y, fittedvalues) == pytest.approx(expected)


class TestTrainFinanceModels:
    def test_trains_per_asset_models_and_one_gbm_run(self, tmp_path):
        features_dir = tmp_path / "features"
        run_id = "2026-09-23"
        train_df, ticker_mapping = _synthetic_train_panel()
        _write_features_fixture(features_dir, run_id, train_df, ticker_mapping)

        config_dir = tmp_path / "config"
        _write_finance_config(config_dir, models=[
            {"name": "m6_returns_risk_random_walk", "type": "random_walk", "hyperparameters": "{}"},
            {"name": "m6_returns_risk_mean", "type": "mean", "hyperparameters": "{window: null}"},
            {"name": "m6_returns_risk_arima", "type": "arima", "hyperparameters": "{order: [1, 0, 0]}"},
            {"name": "m6_returns_risk_cross_sectional_gbm", "type": "cross_sectional_gbm",
             "hyperparameters": "{n_estimators: 5, max_depth: 2, random_state: 42}"},
        ])

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_train_finance_models")

        result = train_finance_models(features_dir, run_id, config_dir, mlflow_tracking_uri=mlflow_uri)

        expected_keys = {
            "m6_returns_risk_random_walk_AAPL", "m6_returns_risk_random_walk_MSFT",
            "m6_returns_risk_mean_AAPL", "m6_returns_risk_mean_MSFT",
            "m6_returns_risk_arima_AAPL", "m6_returns_risk_arima_MSFT",
            "m6_returns_risk_cross_sectional_gbm",
        }
        assert set(result["models"]) == expected_keys
        for key, info in result["models"].items():
            assert "mlflow_run_id" in info
            assert "train_error_std" in info
            assert info["train_error_std"] >= 0

    def test_one_bad_asset_does_not_block_the_others(self, tmp_path):
        """An asset with too few rows to fit is logged and skipped, not raised."""
        features_dir = tmp_path / "features"
        run_id = "2026-09-23"
        train_df, ticker_mapping = _synthetic_train_panel()
        # Truncate MSFT down to a single row -> too few to fit anything meaningfully.
        msft_first_idx = train_df[train_df["ticker_encoded"] == 1].index[0]
        keep_mask = (train_df["ticker_encoded"] != 1) | (train_df.index == msft_first_idx)
        train_df = train_df[keep_mask].reset_index(drop=True)
        _write_features_fixture(features_dir, run_id, train_df, ticker_mapping)

        config_dir = tmp_path / "config"
        _write_finance_config(config_dir, models=[
            {"name": "m6_returns_risk_random_walk", "type": "random_walk", "hyperparameters": "{}"},
        ])

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_train_finance_models_partial_failure")

        result = train_finance_models(features_dir, run_id, config_dir, mlflow_tracking_uri=mlflow_uri)

        assert "m6_returns_risk_random_walk_AAPL" in result["models"]
        assert "m6_returns_risk_random_walk_MSFT" not in result["models"]

    def test_raises_when_train_parquet_missing(self, tmp_path):
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        _write_finance_config(config_dir, models=[{"name": "x", "type": "random_walk", "hyperparameters": "{}"}])

        with pytest.raises(FileNotFoundError):
            train_finance_models(features_dir, "2026-09-23", config_dir)
