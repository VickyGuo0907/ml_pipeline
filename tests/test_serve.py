"""Tests for FastAPI serving endpoint."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.serve import app, PredictionInput, _model_cache


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


class TestHealth:
    """Tests for health check endpoint."""

    def test_health_returns_healthy_status(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_reports_model_loaded(self, client, mock_model):
        """Test health endpoint reports model loaded status."""
        _model_cache["model"] = mock_model
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_health_reports_model_not_loaded(self, client):
        """Test health endpoint reports when model not loaded."""
        _model_cache["model"] = None
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is False


VALID_INPUT = {
    "state": 0.5,
    "care_transition": -0.3,
    "cleanliness": 0.8,
    "communication_about_medicines": 0.2,
    "discharge_information": -0.1,
    "doctor_communication": 0.6,
    "nurse_communication": 0.4,
    "overall_hospital_rating": -0.2,
    "quietness": 0.1,
    "recommend_hospital": 0.3,
    "staff_responsiveness": -0.5,
}


class TestPredict:
    """Tests for prediction endpoint."""

    def test_predict_returns_prediction_when_model_loaded(self, client, mock_model):
        """Test predict endpoint returns prediction value."""
        _model_cache["model"] = mock_model
        _model_cache["model_name"] = "lightgbm_gbm"
        _model_cache["model_version"] = "3"
        _model_cache["model_stage"] = "Production"
        _model_cache["boxcox_lambda"] = None

        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] == 0.95
        assert data["model_name"] == "lightgbm_gbm"
        assert data["model_version"] == "3"
        assert data["model_stage"] == "Production"

    def test_predict_applies_inverse_boxcox(self, client, mock_model):
        """Test predict endpoint inverse-transforms Box-Cox predictions to original ERR scale."""
        _model_cache["model"] = mock_model
        _model_cache["model_name"] = "lightgbm_gbm"
        _model_cache["model_version"] = "3"
        _model_cache["model_stage"] = "Production"
        _model_cache["boxcox_lambda"] = -0.3

        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction_transformed"] == 0.95
        assert data["prediction"] != 0.95  # inverse-transformed to original scale

    def test_predict_fails_when_model_not_loaded(self, client):
        """Test predict endpoint returns 503 when model not loaded."""
        _model_cache["model"] = None

        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 503
        assert "No model loaded" in response.json()["detail"]

    def test_predict_ignores_unknown_extra_fields(self, client, mock_model):
        """Test predict endpoint ignores fields not in the current feature schema."""
        _model_cache["model"] = mock_model
        _model_cache["model_name"] = "lightgbm_gbm"
        _model_cache["model_version"] = "3"
        _model_cache["model_stage"] = "Production"
        _model_cache["boxcox_lambda"] = None

        input_data = {**VALID_INPUT, "some_legacy_field": 1234.0}

        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        assert response.json()["prediction"] == 0.95

    def test_predict_validates_required_fields(self, client):
        """Test predict endpoint validates required input fields."""
        response = client.post("/predict", json={"state": 0.5})
        assert response.status_code == 422  # Validation error


class TestPredictionInput:
    """Tests for PredictionInput model."""

    def test_prediction_input_required_fields(self):
        """Test PredictionInput requires all non-optional fields."""
        with pytest.raises(ValueError):
            PredictionInput(state=0.5)

    def test_prediction_input_with_all_fields(self):
        """Test PredictionInput accepts all fields."""
        input_data = PredictionInput(**VALID_INPUT)
        assert input_data.state == 0.5
        assert input_data.staff_responsiveness == -0.5
