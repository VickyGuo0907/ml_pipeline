"""End-to-end integration test for the m6_returns_risk pipeline: calls every
finance stage function directly in sequence (ingest through register), plus
a serving smoke test — mirrors tests/test_forecasting_integration.py's
pattern for pjm_load_forecast (this repo has no separate
tests/test_integration.py). Deferred from Plan 2 per this pipeline's own
established phasing, since it needs Plans 3-5's training/evaluation/serving
stages to produce something to validate against.

Uses a small synthetic 4-asset x 60-month panel so the full run completes
in seconds. drift_report and unsupervised_explore are skipped:
config/m6_returns_risk/orchestration.yaml sets both to false (no finance-
specific drift comparison design; PCA/k-means don't apply to per-asset
return series) — there is no finance stage function for either.
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from src.finance.clean_finance import clean_finance_data
from src.finance.evaluate_finance import register_finance_models_to_mlflow
from src.finance.features_finance import engineer_finance_features
from src.finance.train_finance import train_finance_models
from src.ingest import ingest_files
from src.profile import generate_adf_report, profile_raw_files
from src.schemas.features import build_finance_features_schema
from src.validate import validate_raw_files

PIPELINE_TYPE = "test_m6_returns_risk"
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _write_landing_csv(landing_dir: Path, n_months: int = 60) -> None:
    """A 5-year synthetic monthly panel, 4 assets, shaped like the real
    long-format m6_returns_panel.csv (Date, Ticker, log_return)."""
    landing_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    dates = pd.period_range("2019-01", periods=n_months, freq="M").to_timestamp()
    rows = []
    for ticker in TICKERS:
        returns = rng.normal(0.01, 0.05, n_months)
        for date, r in zip(dates, returns):
            rows.append({"Date": date.strftime("%Y-%m-%d"), "Ticker": ticker, "log_return": float(r)})
    pd.DataFrame(rows).to_csv(landing_dir / "m6_returns_panel_test.csv", index=False)


def _write_config(config_dir: Path) -> None:
    """A reduced-scale m6_returns_risk-shaped config: same shape as
    config/m6_returns_risk/*.yaml, small hyperparameters so the run
    completes in seconds, plus an orchestration.yaml (read by
    src/serve.py's predict_finance_return during the serving smoke test)."""
    config_dir.mkdir(parents=True)
    (config_dir / "pipeline.yaml").write_text(
        f"pipeline_type: {PIPELINE_TYPE}\n"
        "sources:\n"
        f"  - name: {PIPELINE_TYPE}\n"
        f"    path: data/{PIPELINE_TYPE}/landing\n"
        "    format: csv\n"
        "target:\n"
        "  name: log_return\n"
        "  type: continuous\n"
        "problem_type: finance\n"
        "train_test_split: 0.80\n"
        "random_state: 42\n"
        "validation:\n"
        "  required_columns: [Date, Ticker, log_return]\n"
        "  min_rows: 100\n"
        "profiling:\n"
        "  minimal: true\n"
        "unsupervised:\n"
        "  enabled: false\n"
        "benchmark:\n"
        "  enabled: false\n"
    )
    (config_dir / "cleaning.yaml").write_text("max_gap_months: 2\nfill_strategy: interpolate\n")
    (config_dir / "features.yaml").write_text("lag_months: [1]\nvolatility_window_months: 3\n")
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  - name: test_m6_random_walk\n    type: random_walk\n    hyperparameters: {}\n"
        "  - name: test_m6_mean\n    type: mean\n    hyperparameters: {window: null}\n"
        "  - name: test_m6_arima\n    type: arima\n    hyperparameters: {order: [1, 0, 0]}\n"
        "  - name: test_m6_cross_sectional_gbm\n    type: cross_sectional_gbm\n"
        "    hyperparameters: {n_estimators: 10, max_depth: 2, random_state: 42}\n"
        "evaluation:\n  champion_metric: rank_ic\n"
    )
    (config_dir / "orchestration.yaml").write_text(
        "dag:\n"
        f"  dag_id: {PIPELINE_TYPE}_pipeline\n"
        "  owner: test\n"
        "  description: test\n"
        '  schedule: "@monthly"\n'
        "  catchup: false\n"
        "  tags: []\n"
        "tasks:\n"
        "  retries: 0\n"
        "  retry_delay_minutes: 1\n"
        "  train_models_retries: 0\n"
        "  enabled:\n"
        "    profile: true\n"
        "    unsupervised_explore: false\n"
        "    drift_report: false\n"
        "directories:\n"
        f"  landing: data/{PIPELINE_TYPE}/landing\n"
        f"  raw: data/{PIPELINE_TYPE}/raw\n"
        f"  interim: data/{PIPELINE_TYPE}/interim\n"
        f"  features: data/{PIPELINE_TYPE}/features\n"
        f"  benchmark: data/{PIPELINE_TYPE}/benchmark\n"
        f"  reports: reports/{PIPELINE_TYPE}\n"
        f"  config: config/{PIPELINE_TYPE}\n"
        '  reports_base_url: "http://localhost:8888"\n'
    )


class TestM6ReturnsRiskIntegration:
    def test_full_pipeline_ingest_through_register_and_serve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_id = "2026-09-23"

        landing_dir = Path(f"data/{PIPELINE_TYPE}/landing")
        _write_landing_csv(landing_dir)
        config_dir = Path(f"config/{PIPELINE_TYPE}")
        _write_config(config_dir)

        raw_dir = Path(f"data/{PIPELINE_TYPE}/raw")
        interim_dir = Path(f"data/{PIPELINE_TYPE}/interim")
        features_dir = Path(f"data/{PIPELINE_TYPE}/features")
        reports_dir = Path(f"reports/{PIPELINE_TYPE}")

        ingest_result = ingest_files(landing_dir=landing_dir, raw_dir=raw_dir, run_id=run_id)
        assert ingest_result["run_id"] == run_id

        validate_result = validate_raw_files(raw_dir=raw_dir, run_id=run_id, config_dir=config_dir)
        assert validate_result is not None

        profile_raw_files(raw_dir=raw_dir, run_id=run_id, reports_dir=reports_dir, config_dir=config_dir)
        adf_result = generate_adf_report(raw_dir=raw_dir, run_id=run_id, reports_dir=reports_dir, config_dir=config_dir)
        assert Path(adf_result["report_path"]).exists()
        assert Path(adf_result["yaml_path"]).exists()

        clean_result = clean_finance_data(raw_dir=raw_dir, interim_dir=interim_dir, run_id=run_id, config_dir=config_dir)
        assert clean_result["row_count"] > 0

        features_result = engineer_finance_features(
            interim_dir=interim_dir, features_dir=features_dir, run_id=run_id, config_dir=config_dir,
        )
        assert features_result["rows_dropped_warmup"] >= 0

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        schema = build_finance_features_schema("log_return")
        validated = schema.validate(train_df)
        assert len(validated) == len(train_df)

        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        # Explicit set_tracking_uri + set_experiment before invoking the
        # stage function — matches tests/test_forecasting_integration.py's
        # established pattern. Without this, a prior test in the full suite
        # can leave mlflow's global active-experiment state pointing at an
        # experiment ID that doesn't exist in this test's fresh sqlite DB,
        # silently failing every model fit (caught by train_finance_models'
        # per-model try/except) — passes in isolation, fails under the full
        # suite otherwise.
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_m6_returns_risk_e2e")
        train_result = train_finance_models(
            features_dir=features_dir, run_id=run_id, config_dir=config_dir, mlflow_tracking_uri=mlflow_uri,
        )
        assert len(train_result["models"]) > 0
        run_ids = {name: info["mlflow_run_id"] for name, info in train_result["models"].items()}

        register_result = register_finance_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri, mlflow_run_ids=run_ids, config_dir=config_dir,
            run_id=run_id, reports_dir=reports_dir, features_dir=features_dir,
        )
        assert len(register_result["registered_models"]) > 0

        eval_report_path = reports_dir / f"{run_id}_evaluation.yaml"
        assert eval_report_path.exists()
        with open(eval_report_path) as f:
            report = yaml.safe_load(f)
        assert report["run_champion_type"] is not None
        assert report["adf_summary"] is not None  # the ADF report ran earlier in this same test

        # Serving smoke test: pick one registered per-asset model, point
        # SERVING_MODEL_NAME/MLFLOW_TRACKING_URI at it, confirm _load_model
        # detects it as finance and the endpoint returns a real prediction.
        import src.serve as serve_module
        per_asset_name = next(
            name for name, info in report["models"].items()
            if info.get("status") == "registered" and info.get("model_type") != "cross_sectional_gbm"
        )
        monkeypatch.setattr(serve_module, "MLFLOW_TRACKING_URI", mlflow_uri)
        monkeypatch.setattr(serve_module, "SERVING_MODEL_NAME", per_asset_name)
        loaded = serve_module._load_model(per_asset_name)
        assert loaded is not None
        serve_module._model_cache.update(loaded)
        try:
            assert serve_module._model_cache["is_finance"] is True

            client = TestClient(serve_module.app)
            response = client.get("/predict/finance-return", params={"horizon_months": 2})
            assert response.status_code == 200
            assert isinstance(response.json()["prediction"], float)
        finally:
            serve_module._model_cache.clear()
