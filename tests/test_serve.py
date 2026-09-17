"""Tests for the pipeline-agnostic FastAPI serving endpoint.

The served model's feature schema and target name are not hardcoded — they come
from _model_cache["feature_columns"]/["target_col"], populated at load time from
that model's own MLflow training run. Tests exercise two distinct simulated
schemas (a small 3-feature set and a larger 6-feature set with different names)
to prove the endpoint genuinely adapts, rather than happening to match one
hardcoded pipeline's shape.
"""
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.serve import app, _load_model, _model_cache


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock MLflow model."""
    model = MagicMock()
    model.predict.return_value = [0.95]
    return model


_CACHE_DEFAULTS = {
    "model": None, "model_name": None, "model_version": None, "model_stage": None,
    "boxcox_lambda": None, "boxcox_offset": None, "feature_columns": None, "target_col": None,
    "is_forecasting": False, "forecast_model_type": None, "pipeline_type": None,
    "lags": None, "rolling_windows": None, "calendar_features": None, "holiday_features": None,
}


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Ensure no state leaks between tests — each test sets up its own cache."""
    _model_cache.update(_CACHE_DEFAULTS)
    yield
    _model_cache.update(_CACHE_DEFAULTS)


# Two distinct schemas to prove genericity — neither is hardcoded into src/serve.py.
SMALL_SCHEMA = ["State", "Nurse communication", "Overall hospital rating"]
SMALL_INPUT = {"State": 0.5, "Nurse communication": -0.3, "Overall hospital rating": 0.8}

WIDE_SCHEMA = ["mspb_1_spending", "hai_1_sir", "hai_2_sir", "overall_star_rating", "tec_imm3_flu_vaccination", "ownership_type"]
WIDE_INPUT = {
    "mspb_1_spending": 0.1, "hai_1_sir": -0.2, "hai_2_sir": 0.3,
    "overall_star_rating": -0.4, "tec_imm3_flu_vaccination": 0.5, "ownership_type": -0.6,
}


def _load(model, model_name="test_model", version="3", stage="Production",
          boxcox_lambda=None, boxcox_offset=None, feature_columns=None, target_col=None):
    _model_cache.update({
        "model": model, "model_name": model_name, "model_version": version, "model_stage": stage,
        "boxcox_lambda": boxcox_lambda, "boxcox_offset": boxcox_offset,
        "feature_columns": feature_columns, "target_col": target_col,
    })


class TestHealth:
    """Tests for health check endpoint."""

    def test_health_returns_healthy_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_reports_model_loaded(self, client, mock_model):
        _load(mock_model, target_col="expression_level")
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True
        assert response.json()["target_col"] == "expression_level"

    def test_health_reports_model_not_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False
        assert response.json()["target_col"] is None


class TestSchema:
    """Tests for the /schema introspection endpoint (replaces a static OpenAPI schema)."""

    def test_schema_reports_small_pipelines_feature_set(self, client, mock_model):
        _load(mock_model, feature_columns=SMALL_SCHEMA, target_col="Excess Readmission Ratio")
        response = client.get("/schema")
        assert response.status_code == 200
        data = response.json()
        assert data["required_features"] == SMALL_SCHEMA
        assert data["target_col"] == "Excess Readmission Ratio"

    def test_schema_reports_a_completely_different_wider_feature_set(self, client, mock_model):
        """Same endpoint, different model loaded — proves the schema isn't hardcoded."""
        _load(mock_model, feature_columns=WIDE_SCHEMA, target_col="expression_level")
        response = client.get("/schema")
        assert response.status_code == 200
        data = response.json()
        assert data["required_features"] == WIDE_SCHEMA
        assert data["target_col"] == "expression_level"
        assert len(data["required_features"]) == 6

    def test_schema_reports_boxcox_applied_flag(self, client, mock_model):
        _load(mock_model, feature_columns=SMALL_SCHEMA, boxcox_lambda=-0.3)
        response = client.get("/schema")
        assert response.json()["boxcox_applied"] is True

    def test_schema_when_no_model_loaded(self, client):
        response = client.get("/schema")
        assert response.status_code == 200
        assert response.json()["required_features"] is None


class TestPredict:
    """Tests for the prediction endpoint against two different loaded schemas."""

    def test_predict_with_small_schema(self, client, mock_model):
        _load(mock_model, model_name="model_a", feature_columns=SMALL_SCHEMA)

        response = client.post("/predict", json=SMALL_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.95
        assert data["model_name"] == "model_a"

    def test_predict_with_wide_different_schema(self, client, mock_model):
        """Same endpoint, a completely different feature set — proves genericity."""
        _load(mock_model, model_name="model_b", feature_columns=WIDE_SCHEMA, target_col="expression_level")

        response = client.post("/predict", json=WIDE_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0.95
        assert data["model_name"] == "model_b"
        assert data["target_col"] == "expression_level"

    def test_predict_passes_features_to_model_in_trained_column_order(self, client, mock_model):
        """Column order must match the trained order, not JSON key order, or a
        tree/linear model would silently score the wrong feature against the
        wrong coefficient/split."""
        reordered_input = {k: SMALL_INPUT[k] for k in reversed(list(SMALL_INPUT))}
        _load(mock_model, feature_columns=SMALL_SCHEMA)

        client.post("/predict", json=reordered_input)

        called_df = mock_model.predict.call_args[0][0]
        assert list(called_df.columns) == SMALL_SCHEMA

    def test_predict_applies_inverse_boxcox(self, client, mock_model):
        _load(mock_model, feature_columns=SMALL_SCHEMA, boxcox_lambda=-0.3)

        response = client.post("/predict", json=SMALL_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction_transformed"] == 0.95
        assert data["prediction"] != 0.95  # inverse-transformed to original scale

    def test_predict_inverse_boxcox_subtracts_persisted_offset(self, client, mock_model):
        """The Box-Cox shift must be subtracted back off at prediction time."""
        from src.serve import _inverse_boxcox

        _load(mock_model, feature_columns=SMALL_SCHEMA, boxcox_lambda=-0.3, boxcox_offset=1.0)

        response = client.post("/predict", json=SMALL_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == pytest.approx(_inverse_boxcox(0.95, -0.3, 1.0))

    def test_predict_fails_when_model_not_loaded(self, client):
        response = client.post("/predict", json=SMALL_INPUT)
        assert response.status_code == 503
        assert "No model loaded" in response.json()["detail"]

    def test_predict_fails_when_model_has_no_feature_schema(self, client, mock_model):
        """A model registered before serving metadata was added — no feature_columns
        artifact — must fail clearly, not crash or silently guess a schema."""
        _load(mock_model, feature_columns=None)

        response = client.post("/predict", json=SMALL_INPUT)
        assert response.status_code == 503
        assert "feature schema" in response.json()["detail"]

    def test_predict_ignores_unknown_extra_fields(self, client, mock_model):
        _load(mock_model, feature_columns=SMALL_SCHEMA)

        response = client.post("/predict", json={**SMALL_INPUT, "some_legacy_field": 1234.0})
        assert response.status_code == 200
        assert response.json()["prediction"] == 0.95

    def test_predict_validates_required_fields_missing(self, client, mock_model):
        _load(mock_model, feature_columns=SMALL_SCHEMA)

        response = client.post("/predict", json={"State": 0.5})
        assert response.status_code == 422
        assert "Missing required feature" in response.json()["detail"]
        assert "Nurse communication" in response.json()["detail"]


class TestForecastingDispatch:
    """Tests for the forecasting-model detection and loading path added to
    _load_model. Tabular models (no cv_mape_mean metric) must be completely
    unaffected — these tests exercise ONLY the new forecasting branch."""

    def test_detects_forecasting_model_via_cv_mape_mean_metric(self, tmp_path, monkeypatch):
        """The presence of cv_mape_mean (never logged by tabular src/train.py)
        is the sole signal used to route to the forecasting load path —
        confirm _load_model actually branches on it."""
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_forecasting_dispatch_gbm")

        model_name = "test_forecast_gbm_model"
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"lag_1h": rng.normal(size=20), "hour": rng.integers(0, 24, size=20)})
        y_train = pd.Series(rng.normal(size=20))
        import lightgbm as lgb
        gbm_model = lgb.LGBMRegressor(n_estimators=5, max_depth=2).fit(X_train, y_train)

        with mlflow.start_run():
            mlflow.set_tags({"model_type": "gbm", "pipeline_type": "test_forecast"})
            mlflow.log_metric("cv_mape_mean", 4.2)
            mlflow.log_param("target_col", "PJME_MW")
            mlflow.log_dict({"columns": ["lag_1h", "hour"]}, "feature_columns.json")
            mlflow.lightgbm.log_model(gbm_model, name="model", registered_model_name=model_name)

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        version = client.get_latest_versions(model_name)[0]
        client.transition_model_version_stage(name=model_name, version=version.version, stage="Production")

        monkeypatch.setattr("src.serve.MLFLOW_TRACKING_URI", mlflow_uri)

        result = _load_model(model_name)

        assert result is not None
        assert result["is_forecasting"] is True
        assert result["forecast_model_type"] == "gbm"
        # Loaded via mlflow.lightgbm, not mlflow.pyfunc — a pyfunc-wrapped model
        # would come back as a PyFuncModel instance instead of the raw sklearn
        # wrapper lightgbm's own flavor returns.
        assert isinstance(result["model"], lgb.LGBMRegressor)
        assert not isinstance(result["model"], mlflow.pyfunc.PyFuncModel)

    def test_tabular_model_unaffected_by_new_dispatch(self, tmp_path, monkeypatch):
        """A run with no cv_mape_mean metric (every existing tabular run)
        must load exactly as before — is_forecasting False, model loaded
        via the existing mlflow.pyfunc path."""
        mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("test_forecasting_dispatch_tabular")

        model_name = "test_tabular_model"
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"State": rng.normal(size=20), "Nurse communication": rng.normal(size=20)})
        y_train = pd.Series(rng.normal(size=20))
        from sklearn.linear_model import LinearRegression
        sk_model = LinearRegression().fit(X_train, y_train)

        with mlflow.start_run():
            mlflow.log_param("target_col", "Excess Readmission Ratio")
            mlflow.log_dict({"columns": ["State", "Nurse communication"]}, "feature_columns.json")
            mlflow.sklearn.log_model(sk_model, name="model", registered_model_name=model_name)

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        version = client.get_latest_versions(model_name)[0]
        client.transition_model_version_stage(name=model_name, version=version.version, stage="Production")

        monkeypatch.setattr("src.serve.MLFLOW_TRACKING_URI", mlflow_uri)

        result = _load_model(model_name)

        assert result is not None
        assert result.get("is_forecasting") is False
        assert result.get("forecast_model_type") is None
        assert result["target_col"] == "Excess Readmission Ratio"
        assert result["feature_columns"] == ["State", "Nurse communication"]
        # Loaded via the existing mlflow.pyfunc path — the only flavor tabular
        # models have ever been loaded through.
        assert isinstance(result["model"], mlflow.pyfunc.PyFuncModel)


class TestPredictForecastEndpoint:
    def test_returns_422_or_400_when_no_forecasting_model_loaded(self, client, monkeypatch):
        """GET /predict/forecast against a tabular deployment (the normal
        case for every pipeline except pjm_load_forecast) must fail clearly,
        not silently return nonsense."""
        _load(MagicMock(), feature_columns=SMALL_SCHEMA)  # tabular shape: is_forecasting stays False

        response = client.get("/predict/forecast", params={"horizon_hours": 4})

        assert response.status_code == 400
        assert "not a forecasting model" in response.json()["detail"]

    def test_forecasts_via_recursive_gbm_path_when_loaded_model_is_gbm(self, tmp_path, client, monkeypatch):
        """A forecasting/gbm-shaped cache should route through
        forecast_with_gbm, seeded from the latest run's last_window.parquet
        snapshot under data/<pipeline_type>/features."""
        idx = pd.date_range("2020-01-01", periods=20, freq="h", name="Datetime")
        snapshot = pd.DataFrame({"PJME_MW": range(20)}, index=idx, dtype=float)
        run_dir = tmp_path / "data" / "test_forecast" / "features" / "2026-09-17"
        run_dir.mkdir(parents=True)
        snapshot.to_parquet(run_dir / "last_window.parquet")
        monkeypatch.chdir(tmp_path)  # load_latest_snapshot resolves data/<pipeline_type>/features relative to cwd

        class _IdentityLag1Model:
            def predict(self, X):
                return X["lag_1h"].to_numpy()

        _model_cache.update({
            "model": _IdentityLag1Model(),
            "model_name": "test_gbm_model",
            "model_version": "1",
            "model_stage": "Production",
            "is_forecasting": True,
            "forecast_model_type": "gbm",
            "pipeline_type": "test_forecast",
            "target_col": "PJME_MW",
            "feature_columns": ["lag_1h"],
            "lags": [1],
            "rolling_windows": [],
            "calendar_features": False,
            "holiday_features": False,
        })

        response = client.get("/predict/forecast", params={"horizon_hours": 4})

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 4
        assert data["forecast_model_type"] == "gbm"

    def test_forecasts_via_native_statsmodels_path_when_loaded_model_is_ets_or_sarimax(self, client, monkeypatch):
        """A forecasting/ets-shaped cache should route through
        forecast_with_statsmodels, forecasting directly from the fitted
        model's own state (no snapshot needed)."""
        from src.forecasting.model_registry import fit_ets

        idx = pd.date_range("2020-01-01", periods=200, freq="h")
        rng = np.random.default_rng(42)
        y = pd.Series(50 + 10 * np.sin(np.arange(200) * 2 * np.pi / 24) + rng.normal(0, 1, 200), index=idx)
        fitted = fit_ets(y, {"seasonal_periods": 24, "trend": "add", "seasonal": "add"})

        _model_cache.update({
            "model": fitted,
            "model_name": "test_ets_model",
            "model_version": "1",
            "model_stage": "Production",
            "is_forecasting": True,
            "forecast_model_type": "ets",
            "target_col": "PJME_MW",
        })

        response = client.get("/predict/forecast", params={"horizon_hours": 6})

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 6
        assert data["forecast_model_type"] == "ets"
