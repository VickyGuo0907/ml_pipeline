"""Tests for the pipeline-agnostic FastAPI serving endpoint.

The served model's feature schema and target name are not hardcoded — they come
from _model_cache["feature_columns"]/["target_col"], populated at load time from
that model's own MLflow training run. Tests exercise two distinct simulated
schemas (a small 3-feature set and a larger 6-feature set with different names)
to prove the endpoint genuinely adapts, rather than happening to match one
hardcoded pipeline's shape.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.serve import app, _model_cache


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


@pytest.fixture(autouse=True)
def reset_model_cache():
    """Ensure no state leaks between tests — each test sets up its own cache."""
    _model_cache.update({
        "model": None, "model_name": None, "model_version": None, "model_stage": None,
        "boxcox_lambda": None, "feature_columns": None, "target_col": None,
    })
    yield
    _model_cache.update({
        "model": None, "model_name": None, "model_version": None, "model_stage": None,
        "boxcox_lambda": None, "feature_columns": None, "target_col": None,
    })


# Two distinct schemas to prove genericity — neither is hardcoded into src/serve.py.
SMALL_SCHEMA = ["State", "Nurse communication", "Overall hospital rating"]
SMALL_INPUT = {"State": 0.5, "Nurse communication": -0.3, "Overall hospital rating": 0.8}

WIDE_SCHEMA = ["mspb_1_spending", "hai_1_sir", "hai_2_sir", "overall_star_rating", "tec_imm3_flu_vaccination", "ownership_type"]
WIDE_INPUT = {
    "mspb_1_spending": 0.1, "hai_1_sir": -0.2, "hai_2_sir": 0.3,
    "overall_star_rating": -0.4, "tec_imm3_flu_vaccination": 0.5, "ownership_type": -0.6,
}


def _load(model, model_name="test_model", version="3", stage="Production",
          boxcox_lambda=None, feature_columns=None, target_col=None):
    _model_cache.update({
        "model": model, "model_name": model_name, "model_version": version, "model_stage": stage,
        "boxcox_lambda": boxcox_lambda, "feature_columns": feature_columns, "target_col": target_col,
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
