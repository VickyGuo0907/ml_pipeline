"""FastAPI serving endpoint for ML model predictions."""
import os
from typing import Optional
import logging
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="ML Pipeline Server")

# MLflow tracking URI from environment or default
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model cache
_model_cache = {"model": None, "model_name": None}


def load_production_model():
    """Load Production model from MLflow registry."""
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # Get latest Production version of any registered model
        for model in client.search_registered_models():
            for version in model.latest_versions:
                if version.current_stage == "Production":
                    model_uri = f"models:/{model.name}/Production"
                    model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"Loaded Production model: {model.name}")
                    return model
        logger.warning("No Production model found in registry")
        return None
    except Exception as e:
        logger.error(f"Failed to load Production model: {e}")
        return None


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    _model_cache["model"] = load_production_model()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


class PredictionInput(BaseModel):
    """Feature input for model prediction."""
    state_encoded: int
    facility_id_encoded: int
    compared_to_national_mortality_below: int
    compared_to_national_mortality_same: int
    compared_to_national_mortality_above: int
    compared_to_national_safety_below: int
    compared_to_national_safety_same: int
    compared_to_national_safety_above: int
    compared_to_national_readmission_below: int
    compared_to_national_readmission_same: int
    compared_to_national_readmission_above: int
    mortality_rate: float
    hcahps_cleanliness: float
    hcahps_communication: float
    hcahps_responsiveness: float
    hcahps_pain_management: float
    hcahps_medication: float
    hcahps_discharge: float
    hcahps_quiet: float
    number_of_beds: float
    hcahps_cleanliness_poly2: Optional[float] = None
    hcahps_communication_poly2: Optional[float] = None
    hcahps_cleanliness_communication: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
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
                "hcahps_cleanliness_poly2": None,
                "hcahps_communication_poly2": None,
                "hcahps_cleanliness_communication": None,
            }
        }


class PredictionOutput(BaseModel):
    """Model prediction output."""
    prediction: float = Field(..., description="Predicted ExcessReadmissionRatio")
    model_version: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_cache["model"] is not None
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput) -> PredictionOutput:
    """Make prediction using loaded model."""
    if _model_cache["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MLflow registry for Production version."
        )

    try:
        # Convert input to dict and map to feature column names
        input_dict = data.dict()
        feature_dict = {
            "State_encoded": input_dict["state_encoded"],
            "FacilityId_encoded": input_dict["facility_id_encoded"],
            "ComparedToNational_Mortality_Below": input_dict["compared_to_national_mortality_below"],
            "ComparedToNational_Mortality_Same": input_dict["compared_to_national_mortality_same"],
            "ComparedToNational_Mortality_Above": input_dict["compared_to_national_mortality_above"],
            "ComparedToNational_Safety_Below": input_dict["compared_to_national_safety_below"],
            "ComparedToNational_Safety_Same": input_dict["compared_to_national_safety_same"],
            "ComparedToNational_Safety_Above": input_dict["compared_to_national_safety_above"],
            "ComparedToNational_Readmission_Below": input_dict["compared_to_national_readmission_below"],
            "ComparedToNational_Readmission_Same": input_dict["compared_to_national_readmission_same"],
            "ComparedToNational_Readmission_Above": input_dict["compared_to_national_readmission_above"],
            "Mortality_Rate": input_dict["mortality_rate"],
            "HCAHPS_Cleanliness": input_dict["hcahps_cleanliness"],
            "HCAHPS_Communication": input_dict["hcahps_communication"],
            "HCAHPS_Responsiveness": input_dict["hcahps_responsiveness"],
            "HCAHPS_Pain_Management": input_dict["hcahps_pain_management"],
            "HCAHPS_Medication": input_dict["hcahps_medication"],
            "HCAHPS_Discharge": input_dict["hcahps_discharge"],
            "HCAHPS_Quiet": input_dict["hcahps_quiet"],
            "Number_of_Beds": input_dict["number_of_beds"],
        }

        # Add polynomial features if provided
        if input_dict["hcahps_cleanliness_poly2"] is not None:
            feature_dict["HCAHPS_Cleanliness_poly2"] = input_dict["hcahps_cleanliness_poly2"]
        if input_dict["hcahps_communication_poly2"] is not None:
            feature_dict["HCAHPS_Communication_poly2"] = input_dict["hcahps_communication_poly2"]
        if input_dict["hcahps_cleanliness_communication"] is not None:
            feature_dict["HCAHPS_Cleanliness_Communication"] = input_dict["hcahps_cleanliness_communication"]

        # Make prediction
        import pandas as pd
        input_df = pd.DataFrame([feature_dict])
        prediction = _model_cache["model"].predict(input_df)

        return PredictionOutput(
            prediction=float(prediction[0]),
            model_version="Production"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
