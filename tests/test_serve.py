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


class TestPredict:
    """Tests for prediction endpoint."""

    def test_predict_returns_prediction_when_model_loaded(self, client, mock_model):
        """Test predict endpoint returns prediction value."""
        _model_cache["model"] = mock_model

        input_data = {
            "state_encoded": 1,
            "facility_id_encoded": 42,
            "compared_to_national_mortality_below": 1,
            "compared_to_national_mortality_same": 0,
            "compared_to_national_mortality_above": 0,
            "compared_to_national_safety_below": 0,
            "compared_to_national_safety_same": 1,
            "compared_to_national_safety_above": 0,
            "compared_to_national_readmission_below": 0,
            "compared_to_national_readmission_same": 0,
            "compared_to_national_readmission_above": 1,
            "mortality_rate": 12.5,
            "hcahps_cleanliness": 75.0,
            "hcahps_communication": 80.0,
            "hcahps_responsiveness": 70.0,
            "hcahps_pain_management": 72.0,
            "hcahps_medication": 68.0,
            "hcahps_discharge": 74.0,
            "hcahps_quiet": 71.0,
            "number_of_beds": 150.0,
        }

        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] == 0.95
        assert data["model_version"] == "Production"

    def test_predict_fails_when_model_not_loaded(self, client):
        """Test predict endpoint returns 503 when model not loaded."""
        _model_cache["model"] = None

        input_data = {
            "state_encoded": 1,
            "facility_id_encoded": 42,
            "compared_to_national_mortality_below": 1,
            "compared_to_national_mortality_same": 0,
            "compared_to_national_mortality_above": 0,
            "compared_to_national_safety_below": 0,
            "compared_to_national_safety_same": 1,
            "compared_to_national_safety_above": 0,
            "compared_to_national_readmission_below": 0,
            "compared_to_national_readmission_same": 0,
            "compared_to_national_readmission_above": 1,
            "mortality_rate": 12.5,
            "hcahps_cleanliness": 75.0,
            "hcahps_communication": 80.0,
            "hcahps_responsiveness": 70.0,
            "hcahps_pain_management": 72.0,
            "hcahps_medication": 68.0,
            "hcahps_discharge": 74.0,
            "hcahps_quiet": 71.0,
            "number_of_beds": 150.0,
        }

        response = client.post("/predict", json=input_data)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_predict_with_polynomial_features(self, client, mock_model):
        """Test predict endpoint accepts polynomial features."""
        _model_cache["model"] = mock_model

        input_data = {
            "state_encoded": 1,
            "facility_id_encoded": 42,
            "compared_to_national_mortality_below": 1,
            "compared_to_national_mortality_same": 0,
            "compared_to_national_mortality_above": 0,
            "compared_to_national_safety_below": 0,
            "compared_to_national_safety_same": 1,
            "compared_to_national_safety_above": 0,
            "compared_to_national_readmission_below": 0,
            "compared_to_national_readmission_same": 0,
            "compared_to_national_readmission_above": 1,
            "mortality_rate": 12.5,
            "hcahps_cleanliness": 75.0,
            "hcahps_communication": 80.0,
            "hcahps_responsiveness": 70.0,
            "hcahps_pain_management": 72.0,
            "hcahps_medication": 68.0,
            "hcahps_discharge": 74.0,
            "hcahps_quiet": 71.0,
            "number_of_beds": 150.0,
            "hcahps_cleanliness_poly2": 5625.0,
            "hcahps_communication_poly2": 6400.0,
            "hcahps_cleanliness_communication": 6000.0,
        }

        response = client.post("/predict", json=input_data)
        assert response.status_code == 200
        assert response.json()["prediction"] == 0.95

    def test_predict_validates_required_fields(self, client):
        """Test predict endpoint validates required input fields."""
        response = client.post("/predict", json={"state_encoded": 1})
        assert response.status_code == 422  # Validation error


class TestPredictionInput:
    """Tests for PredictionInput model."""

    def test_prediction_input_required_fields(self):
        """Test PredictionInput requires all non-optional fields."""
        with pytest.raises(ValueError):
            PredictionInput(state_encoded=1)

    def test_prediction_input_with_all_fields(self):
        """Test PredictionInput accepts all fields."""
        input_data = PredictionInput(
            state_encoded=1,
            facility_id_encoded=42,
            compared_to_national_mortality_below=1,
            compared_to_national_mortality_same=0,
            compared_to_national_mortality_above=0,
            compared_to_national_safety_below=0,
            compared_to_national_safety_same=1,
            compared_to_national_safety_above=0,
            compared_to_national_readmission_below=0,
            compared_to_national_readmission_same=0,
            compared_to_national_readmission_above=1,
            mortality_rate=12.5,
            hcahps_cleanliness=75.0,
            hcahps_communication=80.0,
            hcahps_responsiveness=70.0,
            hcahps_pain_management=72.0,
            hcahps_medication=68.0,
            hcahps_discharge=74.0,
            hcahps_quiet=71.0,
            number_of_beds=150.0,
        )
        assert input_data.state_encoded == 1
        assert input_data.facility_id_encoded == 42
