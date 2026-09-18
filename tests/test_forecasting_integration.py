"""End-to-end integration test for the pjm_load_forecast pipeline: calls
every forecasting stage function directly in sequence (ingest through
register), plus a serving smoke test — mirrors tests/test_pipeline.py's
TestIntegration/TestHospitalReadmissionLaggedIntegration pattern (this repo
has no separate tests/test_integration.py). Deferred from Plan 2 since it
needs Plan 3's training stage (and Plans 4-5's evaluation/serving) to
produce something to validate against.

Uses a reduced config (smaller lags/rolling windows, horizon, CV windows,
model hyperparameters) so the full run completes in seconds rather than the
real config's multi-minute scale. drift_report and unsupervised_explore are
skipped: config/pjm_load_forecast/orchestration.yaml sets both to false
(forecasting drift comparison isn't implemented, and PCA/k-means don't apply
to a single univariate series) — there is no forecasting stage function for
either, so a full DAG build for this pipeline only ever wires 7 of the 9
stage slots to begin with.
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from src.forecasting.clean_forecast import clean_forecast_data
from src.forecasting.evaluate_forecast import register_forecast_models_to_mlflow
from src.forecasting.features_forecast import engineer_forecast_features
from src.forecasting.train_forecast import train_forecast_models
from src.ingest import ingest_files
from src.profile import generate_mstl_report, profile_raw_files
from src.schemas.features import build_forecast_features_schema
from src.validate import validate_raw_files

PIPELINE_TYPE = "test_pjm_forecast"


def _write_landing_csv(landing_dir: Path, n_hours: int = 720) -> None:
    """A 30-day synthetic hourly series shaped like PJME_hourly.csv: a
    Datetime column plus PJME_MW, daily-seasonal, always positive (real
    PJM load never approaches zero). 720 hours comfortably clears the
    >336-hour minimum MSTL's weekly (168h) period needs to survive
    statsmodels' internal period < nobs/2 filter."""
    landing_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = 30000 + 5000 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + rng.normal(0, 200, n_hours)
    df = pd.DataFrame({"Datetime": idx.astype(str), "PJME_MW": values})
    df.to_csv(landing_dir / "PJME_hourly_test.csv", index=False)


def _write_config(config_dir: Path) -> None:
    """A reduced-scale pjm_load_forecast-shaped config: same shape as
    config/pjm_load_forecast/*.yaml, smaller lags/horizon/hyperparameters so
    the whole run completes in seconds, plus an orchestration.yaml (not
    needed by the stage functions called directly here, but read by
    src/serve.py's predict_forecast GBM branch during the serving smoke
    test at the end)."""
    config_dir.mkdir(parents=True)
    (config_dir / "pipeline.yaml").write_text(
        f"pipeline_type: {PIPELINE_TYPE}\n"
        "sources:\n"
        f"  - name: {PIPELINE_TYPE}\n"
        f"    path: data/{PIPELINE_TYPE}/landing\n"
        "    format: csv\n"
        "target:\n"
        "  name: PJME_MW\n"
        "  type: continuous\n"
        "problem_type: forecasting\n"
        "train_test_split: 0.80\n"
        "random_state: 42\n"
        "validation:\n"
        "  required_columns:\n"
        "    - Datetime\n"
        "    - PJME_MW\n"
        "  min_rows: 100\n"
        "  per_file_schemas:\n"
        "    - file_pattern: PJME_hourly\n"
        "      required_columns:\n"
        "        - Datetime\n"
        "        - PJME_MW\n"
        "      numeric_bounds:\n"
        "        PJME_MW:\n"
        "          min: 0.0\n"
        "          max: 120000.0\n"
        "      min_rows: 100\n"
        "profiling:\n"
        "  minimal: true\n"
        "unsupervised:\n"
        "  enabled: false\n"
        "benchmark:\n"
        "  enabled: false\n"
    )
    (config_dir / "cleaning.yaml").write_text(
        "max_gap_hours: 6\nfill_strategy: interpolate\n"
    )
    (config_dir / "features.yaml").write_text(
        "lags: [1, 24]\nrolling_windows: [24]\ncalendar_features: true\n"
        "holiday_features: true\nsnapshot_hours: 24\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n"
        f"  - name: {PIPELINE_TYPE}_ets\n"
        "    type: ets\n"
        "    hyperparameters:\n"
        "      seasonal_periods: 24\n"
        "      trend: add\n"
        "      seasonal: add\n"
        f"  - name: {PIPELINE_TYPE}_sarimax\n"
        "    type: sarimax\n"
        "    hyperparameters:\n"
        "      order: [1, 0, 0]\n"
        "      seasonal_order: [1, 0, 0, 24]\n"
        "      use_holiday_exog: true\n"
        f"  - name: {PIPELINE_TYPE}_gbm\n"
        "    type: gbm\n"
        "    hyperparameters:\n"
        "      n_estimators: 20\n"
        "      learning_rate: 0.1\n"
        "      max_depth: 3\n"
        "      num_leaves: 7\n"
        "      random_state: 42\n"
        "      n_jobs: -1\n"
        "evaluation:\n"
        "  horizon_hours: 6\n"
        "  n_windows: 2\n"
        "  champion_metric: cv_mape\n"
    )
    (config_dir / "orchestration.yaml").write_text(
        "dag:\n"
        f"  dag_id: {PIPELINE_TYPE}_pipeline\n"
        "  owner: test\n"
        "  description: test\n"
        '  schedule: "@weekly"\n'
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
        f'  reports_base_url: "http://localhost:8888/{PIPELINE_TYPE}"\n'
    )


class TestPjmForecastEndToEnd:
    """Runs every forecasting stage function directly, in DAG order, against
    a small synthetic PJME_hourly-shaped series — the same style of
    direct-call integration test tests/test_pipeline.py's TestIntegration
    already uses for the tabular pipelines, extended through training,
    registration, and a serving smoke test, since this is the first
    end-to-end run any of Plans 1-5 has exercised."""

    def test_full_pipeline_ingest_through_serving(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        run_id = "2026-09-17"
        landing_dir = Path(f"data/{PIPELINE_TYPE}/landing")
        raw_dir = Path(f"data/{PIPELINE_TYPE}/raw")
        interim_dir = Path(f"data/{PIPELINE_TYPE}/interim")
        features_dir = Path(f"data/{PIPELINE_TYPE}/features")
        reports_dir = Path(f"reports/{PIPELINE_TYPE}")
        config_dir = Path(f"config/{PIPELINE_TYPE}")

        _write_landing_csv(landing_dir)
        _write_config(config_dir)

        # Stage 1: ingest
        ingest_result = ingest_files(landing_dir, raw_dir, run_id)
        assert ingest_result["file_count"] == 1

        # Stage 2: validate_raw
        validate_result = validate_raw_files(raw_dir, run_id, config_dir=config_dir)
        assert validate_result["failed_files"] == []

        # Stage 3: profile (+ MSTL, mirroring dag_factory.py's profile_wrapper
        # dispatch for a forecasting pipeline)
        profile_result = profile_raw_files(raw_dir, run_id, reports_dir=reports_dir, config_dir=config_dir)
        assert "PJME_hourly_test.csv" in profile_result["reports"]
        mstl_result = generate_mstl_report(raw_dir, run_id, reports_dir=reports_dir, config_dir=config_dir)
        assert Path(mstl_result["report_path"]).exists()
        assert 24 in mstl_result["periods_used"]

        # Stage 4: clean
        clean_result = clean_forecast_data(raw_dir, interim_dir, run_id, config_dir=config_dir)
        assert clean_result["duplicates_dropped"] == 0

        # Stage 5: feature_engineer
        features_result = engineer_forecast_features(interim_dir, features_dir, run_id, config_dir=config_dir)
        assert features_result["train_shape"][0] > 0
        assert features_result["test_shape"][0] > 0
        assert (features_dir / run_id / "last_window.parquet").exists()

        # Stage 6: validate_features (mirrors dag_factory.py's
        # validate_features_wrapper inline logic — same three checks, same
        # dispatch to build_forecast_features_schema for a forecasting
        # pipeline; that wrapper is a closure inside build_dag(), not a
        # standalone importable function, so its logic is reproduced here)
        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        assert len(train_df) >= 100
        non_numeric = train_df.select_dtypes(exclude="number").columns.tolist()
        assert non_numeric == []
        build_forecast_features_schema("PJME_MW").validate(train_df)

        # Stage 7: train
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_pjm_forecast_e2e")
        train_result = train_forecast_models(
            features_dir, run_id, config_dir=config_dir, mlflow_tracking_uri=mlflow_uri,
        )
        assert set(train_result["models"]) == {
            f"{PIPELINE_TYPE}_ets", f"{PIPELINE_TYPE}_sarimax", f"{PIPELINE_TYPE}_gbm",
        }
        run_ids = {name: info["mlflow_run_id"] for name, info in train_result["models"].items()}

        # Stage 8: evaluate_and_register (drift_report has no forecasting
        # stage function to call — see this test file's module docstring)
        register_result = register_forecast_models_to_mlflow(
            mlflow_tracking_uri=mlflow_uri,
            mlflow_run_ids=run_ids,
            config_dir=config_dir,
            run_id=run_id,
            reports_dir=reports_dir,
            features_dir=features_dir,
        )
        registered = [
            name for name, info in register_result["registered_models"].items()
            if info["status"] == "registered"
        ]
        assert registered, "At least one model must register for the serving smoke test below"

        # Serving smoke test: load the actual run champion through the real
        # _load_model dispatch path and call GET /predict/forecast — proves
        # the whole chain (train -> register -> serve) produces a usable
        # live forecast, not just that each stage runs without raising.
        report_path = reports_dir / f"{run_id}_evaluation.yaml"
        with open(report_path) as f:
            report = yaml.safe_load(f)
        champion_name = report["run_champion"]
        assert champion_name in registered

        from src.serve import _load_model, _model_cache, app

        monkeypatch.setattr("src.serve.MLFLOW_TRACKING_URI", mlflow_uri)
        load_result = _load_model(champion_name)
        assert load_result is not None
        assert load_result["is_forecasting"] is True

        _model_cache.update(load_result)
        try:
            client = TestClient(app)
            response = client.get("/predict/forecast", params={"horizon_hours": 6})
            assert response.status_code == 200
            body = response.json()
            assert len(body["predictions"]) == 6
            assert body["forecast_model_type"] == load_result["forecast_model_type"]
        finally:
            _model_cache.clear()
