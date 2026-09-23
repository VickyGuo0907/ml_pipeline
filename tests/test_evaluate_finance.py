"""Tests for the finance evaluation/registration stage: per-model-type
cross-sectional rank-IC scoring, champion selection, and MLflow Staging
registration."""
from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.statsmodels
import pandas as pd
import pytest
import yaml

from src.finance.evaluate_finance import register_finance_models_to_mlflow
from src.finance.model_registry import fit_cross_sectional_gbm, fit_mean, fit_random_walk


def _write_test_features_fixture(features_dir: Path, run_id: str, test_df: pd.DataFrame, ticker_mapping: dict[str, int], feature_columns: list[str]) -> None:
    run_path = features_dir / run_id
    run_path.mkdir(parents=True)
    test_df.to_parquet(run_path / "test.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "stage": "feature_engineer_finance",
        "feature_columns": feature_columns,
        "index_columns": ["date_ordinal"],
        "ticker_mapping": ticker_mapping,
    }
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)


def _write_finance_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: log_return\n  type: continuous\nproblem_type: finance\n"
        "pipeline_type: test_m6_returns_risk\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: m6_returns_risk_random_walk\n    type: random_walk\n    hyperparameters: {}\n"
        "  - name: m6_returns_risk_mean\n    type: mean\n    hyperparameters: {window: null}\n"
        "evaluation:\n  champion_metric: rank_ic\n"
    )


def _log_random_walk_run(mlflow_uri: str, experiment: str, ticker: str, y_train: pd.Series) -> str:
    """Mimics what Plan 3's train_finance_models does for one random_walk asset."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    fitted = fit_random_walk(y_train)
    with mlflow.start_run(run_name=f"test_run_random_walk_{ticker}") as run:
        mlflow.set_tags({"model_name": "m6_returns_risk_random_walk", "model_type": "random_walk", "ticker": ticker, "run_id": "2026-09-23", "pipeline_type": "test_m6_returns_risk"})
        mlflow.log_metric("train_error_std", 0.05)
        mlflow.pyfunc.log_model(python_model=fitted, name="model")
        return run.info.run_id


def _log_mean_run(mlflow_uri: str, experiment: str, ticker: str, y_train: pd.Series) -> str:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    fitted = fit_mean(y_train, {"window": None})
    with mlflow.start_run(run_name=f"test_run_mean_{ticker}") as run:
        mlflow.set_tags({"model_name": "m6_returns_risk_mean", "model_type": "mean", "ticker": ticker, "run_id": "2026-09-23", "pipeline_type": "test_m6_returns_risk"})
        mlflow.log_metric("train_error_std", 0.05)
        mlflow.pyfunc.log_model(python_model=fitted, name="model")
        return run.info.run_id


def _log_gbm_run(mlflow_uri: str, experiment: str, train_df: pd.DataFrame) -> str:
    """Mimics what Plan 3's train_finance_models does for the cross_sectional_gbm run."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
    model = fit_cross_sectional_gbm(
        train_df, "log_return", feature_columns,
        # min_child_samples defaults to 20 in LGBMRegressor; with only 8
        # training rows here no split would ever pass that threshold, so the
        # model would always predict the constant train-set mean, making
        # rank-IC undefined (NaN) and the champion-selection assertion below
        # untestable. Lowered only for this small fixture.
        {"n_estimators": 5, "max_depth": 2, "random_state": 42, "min_child_samples": 1},
    )
    with mlflow.start_run(run_name="test_run_cross_sectional_gbm") as run:
        mlflow.set_tags({
            "model_name": "m6_returns_risk_cross_sectional_gbm", "model_type": "cross_sectional_gbm",
            "run_id": "2026-09-23", "pipeline_type": "test_m6_returns_risk",
        })
        mlflow.log_metric("train_error_std", 0.05)
        mlflow.lightgbm.log_model(model, name="model")
        return run.info.run_id


class TestRegisterFinanceModelsToMlflow:
    def test_scores_and_registers_per_asset_models_to_staging(self, tmp_path):
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"
        ticker_mapping = {"AAPL": 0, "MSFT": 1}

        test_df = pd.DataFrame({
            "log_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "lag_1_return": [0.0] * 6,
            "trailing_12m_vol": [0.05] * 6,
            "ticker_encoded": [0, 0, 0, 1, 1, 1],
            "date_ordinal": [700, 701, 702, 700, 701, 702],
        })
        feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
        _write_test_features_fixture(features_dir, run_id, test_df, ticker_mapping, feature_columns)
        _write_finance_config(config_dir)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        experiment = "test_evaluate_finance_models"
        # AAPL and MSFT need distinguishable training histories: identical
        # y_train for both tickers makes random_walk's (last value) and
        # mean's (average) predictions identical across tickers at every
        # test month, which is a constant cross-sectional input -> Spearman
        # rank-IC is undefined (NaN) every month, no model type ever has a
        # scorable IC, and champion selection can never pick a winner.
        y_train_aapl = pd.Series([0.01, 0.02, 0.03, 0.04])
        y_train_msft = pd.Series([0.03, 0.04, 0.05, 0.06])

        run_ids = {
            "m6_returns_risk_random_walk_AAPL": _log_random_walk_run(mlflow_uri, experiment, "AAPL", y_train_aapl),
            "m6_returns_risk_random_walk_MSFT": _log_random_walk_run(mlflow_uri, experiment, "MSFT", y_train_msft),
            "m6_returns_risk_mean_AAPL": _log_mean_run(mlflow_uri, experiment, "AAPL", y_train_aapl),
            "m6_returns_risk_mean_MSFT": _log_mean_run(mlflow_uri, experiment, "MSFT", y_train_msft),
        }

        result = register_finance_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids=run_ids,
            config_dir=config_dir,
            run_id=run_id,
            reports_dir=reports_dir,
            features_dir=features_dir,
        )

        assert set(result["registered_models"]) == set(run_ids)
        for name, info in result["registered_models"].items():
            assert info["status"] == "registered"
            assert info["stage"] == "Staging"
            assert "test_error_std" in info

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        versions = client.get_latest_versions("m6_returns_risk_mean_AAPL", stages=["Staging"])
        assert len(versions) == 1
        prod_versions = client.get_latest_versions("m6_returns_risk_mean_AAPL", stages=["Production"])
        assert len(prod_versions) == 0  # never auto-promoted

        report_path = reports_dir / f"{run_id}_evaluation.yaml"
        assert report_path.exists()
        with open(report_path) as f:
            report = yaml.safe_load(f)
        assert "random_walk" in report["ic_leaderboard"]
        assert "mean" in report["ic_leaderboard"]
        assert report["run_champion_type"] in {"random_walk", "mean"}
        assert "stationarity_note" in report

    def test_champion_type_tag_applied_to_every_run_of_the_winning_type(self, tmp_path):
        """mean's forecast (the historical average, 0.025) ranks perfectly against
        a monotonically increasing actual series; random_walk's forecast (always 0)
        produces a degenerate (NaN) cross-sectional correlation every month since
        every asset gets an identical prediction -> mean must win, and EVERY
        m6_returns_risk_mean_* version (both AAPL and MSFT) must be tagged."""
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"
        ticker_mapping = {"AAPL": 0, "MSFT": 1}

        # AAPL's actual returns rank higher than MSFT's at every test month.
        test_df = pd.DataFrame({
            "log_return": [0.05, 0.06, 0.07, 0.01, 0.02, 0.03],
            "lag_1_return": [0.0] * 6,
            "trailing_12m_vol": [0.05] * 6,
            "ticker_encoded": [0, 0, 0, 1, 1, 1],
            "date_ordinal": [700, 701, 702, 700, 701, 702],
        })
        feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
        _write_test_features_fixture(features_dir, run_id, test_df, ticker_mapping, feature_columns)
        _write_finance_config(config_dir)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        experiment = "test_evaluate_finance_champion"
        # mean's forecast is higher for AAPL (trained on higher returns) than MSFT -> correct rank.
        y_train_aapl = pd.Series([0.05, 0.06, 0.07])
        y_train_msft = pd.Series([0.01, 0.02, 0.03])

        run_ids = {
            "m6_returns_risk_random_walk_AAPL": _log_random_walk_run(mlflow_uri, experiment, "AAPL", y_train_aapl),
            "m6_returns_risk_random_walk_MSFT": _log_random_walk_run(mlflow_uri, experiment, "MSFT", y_train_msft),
            "m6_returns_risk_mean_AAPL": _log_mean_run(mlflow_uri, experiment, "AAPL", y_train_aapl),
            "m6_returns_risk_mean_MSFT": _log_mean_run(mlflow_uri, experiment, "MSFT", y_train_msft),
        }

        register_finance_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri, mlflow_run_ids=run_ids, config_dir=config_dir,
            run_id=run_id, reports_dir=reports_dir, features_dir=features_dir,
        )

        report_path = reports_dir / f"{run_id}_evaluation.yaml"
        with open(report_path) as f:
            report = yaml.safe_load(f)
        assert report["run_champion_type"] == "mean"

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        for ticker in ("AAPL", "MSFT"):
            version = client.get_latest_versions(f"m6_returns_risk_mean_{ticker}", stages=["Staging"])[0]
            tags = client.get_model_version(f"m6_returns_risk_mean_{ticker}", version.version).tags
            assert tags.get("run_champion") == "true"

    def test_one_bad_run_does_not_block_the_others(self, tmp_path):
        """A run_id that doesn't exist in MLflow is logged and skipped, not raised."""
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"
        ticker_mapping = {"AAPL": 0}

        test_df = pd.DataFrame({
            "log_return": [0.01, 0.02, 0.03],
            "lag_1_return": [0.0] * 3,
            "trailing_12m_vol": [0.05] * 3,
            "ticker_encoded": [0, 0, 0],
            "date_ordinal": [700, 701, 702],
        })
        feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
        _write_test_features_fixture(features_dir, run_id, test_df, ticker_mapping, feature_columns)
        _write_finance_config(config_dir)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        experiment = "test_evaluate_finance_partial_failure"
        y_train = pd.Series([0.01, 0.02, 0.03])
        good_run_id = _log_random_walk_run(mlflow_uri, experiment, "AAPL", y_train)

        run_ids = {
            "m6_returns_risk_random_walk_AAPL": good_run_id,
            "m6_returns_risk_mean_AAPL": "not_a_real_run_id",
        }

        result = register_finance_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri, mlflow_run_ids=run_ids, config_dir=config_dir,
            run_id=run_id, reports_dir=reports_dir, features_dir=features_dir,
        )

        assert result["registered_models"]["m6_returns_risk_random_walk_AAPL"]["status"] == "registered"
        assert result["registered_models"]["m6_returns_risk_mean_AAPL"]["status"] == "error"

    def test_cross_sectional_gbm_branch_scores_and_registers(self, tmp_path):
        """Locks in the cross_sectional_gbm orchestration branch - untested
        elsewhere in the suite, previously verified only by manual review."""
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"
        ticker_mapping = {"AAPL": 0, "MSFT": 1}

        test_df = pd.DataFrame({
            "log_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "lag_1_return": [0.01, 0.02, 0.03, 0.03, 0.04, 0.05],
            "trailing_12m_vol": [0.05] * 6,
            "ticker_encoded": [0, 0, 0, 1, 1, 1],
            "date_ordinal": [700, 701, 702, 700, 701, 702],
        })
        feature_columns = ["lag_1_return", "trailing_12m_vol", "ticker_encoded"]
        _write_test_features_fixture(features_dir, run_id, test_df, ticker_mapping, feature_columns)
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "pipeline.yaml").write_text(
            "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
            "target:\n  name: log_return\n  type: continuous\nproblem_type: finance\n"
            "pipeline_type: test_m6_returns_risk\n"
        )
        (config_dir / "models.yaml").write_text(
            "models:\n"
            "  - name: m6_returns_risk_cross_sectional_gbm\n    type: cross_sectional_gbm\n"
            "    hyperparameters: {n_estimators: 5, max_depth: 2, random_state: 42}\n"
            "evaluation:\n  champion_metric: rank_ic\n"
        )

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        experiment = "test_evaluate_finance_gbm"
        train_df = pd.DataFrame({
            "log_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.02, 0.03],
            "lag_1_return": [0.0, 0.01, 0.02, 0.03, 0.0, 0.04, 0.05, 0.02],
            "trailing_12m_vol": [0.05] * 8,
            "ticker_encoded": [0, 0, 0, 0, 1, 1, 1, 1],
        })
        gbm_run_id = _log_gbm_run(mlflow_uri, experiment, train_df)

        result = register_finance_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids={"m6_returns_risk_cross_sectional_gbm": gbm_run_id},
            config_dir=config_dir, run_id=run_id, reports_dir=reports_dir, features_dir=features_dir,
        )

        assert result["registered_models"]["m6_returns_risk_cross_sectional_gbm"]["status"] == "registered"
        assert result["registered_models"]["m6_returns_risk_cross_sectional_gbm"]["stage"] == "Staging"

        report_path = reports_dir / f"{run_id}_evaluation.yaml"
        with open(report_path) as f:
            report = yaml.safe_load(f)
        assert "cross_sectional_gbm" in report["ic_leaderboard"]
        assert report["run_champion_type"] == "cross_sectional_gbm"

    def test_raises_when_features_dir_missing(self, tmp_path):
        config_dir = tmp_path / "config"
        _write_finance_config(config_dir)

        with pytest.raises(ValueError, match="features_dir"):
            register_finance_models_to_mlflow(
                mlflow_run_ids={"x": "fake_run_id"}, config_dir=config_dir, run_id="2026-09-23",
                reports_dir=Path("reports"), features_dir=None,
            )

    def test_raises_when_no_run_ids_provided(self, tmp_path):
        with pytest.raises(ValueError, match="mlflow_run_ids"):
            register_finance_models_to_mlflow(mlflow_run_ids=None, features_dir=tmp_path / "features")
