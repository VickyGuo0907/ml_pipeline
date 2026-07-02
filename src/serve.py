"""FastAPI serving endpoint for ML model predictions.

Model loading order:
  1. Try Production stage for the configured model name
  2. Fall back to Staging (useful during development before manual promotion)

Predictions are inverse Box-Cox transformed back to the original
Excess Readmission Ratio scale using the lambda logged during training.

Environment variables:
  MLFLOW_TRACKING_URI  — MLflow server (default: http://mlflow-server:5000)
  SERVING_MODEL_NAME   — which registered model to serve (default: lightgbm_gbm)
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
SERVING_MODEL_NAME = os.environ.get("SERVING_MODEL_NAME", "lightgbm_gbm")

# Actual feature column names as produced by the feature engineering stage.
# Order must match the training feature matrix (X_train column order).
_FEATURE_COLUMNS = [
    "State",
    "Care transition",
    "Cleanliness",
    "Communication about medicines",
    "Discharge information",
    "Doctor communication",
    "Nurse communication",
    "Overall hospital rating",
    "Quietness",
    "Recommend hospital",
    "Staff responsiveness",
]

# Global model cache — populated on startup
_model_cache: dict[str, Any] = {
    "model": None,
    "model_name": None,
    "model_version": None,
    "model_stage": None,
    "boxcox_lambda": None,
}


def _load_model(model_name: str) -> dict[str, Any] | None:
    """Load model from MLflow registry, trying Production then Staging.

    Args:
        model_name: Registered model name (from SERVING_MODEL_NAME env var).

    Returns:
        Dict with model, version, stage, and boxcox_lambda — or None if not found.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    for stage in ("Production", "Staging"):
        try:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                continue

            version = versions[0]
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(model_uri)

            # Retrieve Box-Cox lambda logged by train.py for inverse transform
            boxcox_lambda: float | None = None
            try:
                run = client.get_run(version.run_id)
                lambda_str = run.data.params.get("boxcox_lambda")
                if lambda_str is not None:
                    boxcox_lambda = float(lambda_str)
            except Exception as e:
                logger.warning("Could not retrieve boxcox_lambda from run: %s", e)

            logger.info(
                "Loaded %s v%s from %s (boxcox_lambda=%s)",
                model_name, version.version, stage, boxcox_lambda,
            )
            return {
                "model": model,
                "model_name": model_name,
                "model_version": version.version,
                "model_stage": stage,
                "boxcox_lambda": boxcox_lambda,
            }
        except Exception as e:
            logger.debug("No %s model for %s: %s", stage, model_name, e)
            continue

    logger.warning("No Production or Staging model found for '%s'", model_name)
    return None


def _inverse_boxcox(value: float, lam: float) -> float:
    """Inverse Box-Cox transform to return prediction to original ERR scale.

    Args:
        value: Box-Cox transformed prediction.
        lam: Box-Cox lambda used during feature engineering.

    Returns:
        Prediction in original Excess Readmission Ratio scale (clamped to ≥ 0).
    """
    if lam == 0:
        import math
        return math.exp(value)
    return max(0.0, (value * lam + 1) ** (1.0 / lam))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    result = _load_model(SERVING_MODEL_NAME)
    if result:
        _model_cache.update(result)
    yield
    _model_cache.clear()


app = FastAPI(
    title="ML Pipeline Prediction Server",
    description=(
        "Serves the hospital readmission prediction model from MLflow. "
        "Feature inputs must be StandardScaler-normalized (z-scores) as produced "
        "by the pipeline's feature engineering stage. "
        "Predictions are returned in original Excess Readmission Ratio scale "
        "via inverse Box-Cox transform."
    ),
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_stage: Optional[str] = None


class PredictionInput(BaseModel):
    """Scaled HCAHPS feature inputs (StandardScaler z-scores from feature engineering).

    All values are z-scores (mean=0, std=1) as output by the pipeline's
    StandardScaler. Positive = above average, negative = below average.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    state: float = Field(..., description="State (frequency-encoded, then z-scored)")
    care_transition: float = Field(..., description="Care transition score (z-score)")
    cleanliness: float = Field(..., description="Cleanliness score (z-score)")
    communication_about_medicines: float = Field(..., description="Communication about medicines (z-score)")
    discharge_information: float = Field(..., description="Discharge information score (z-score)")
    doctor_communication: float = Field(..., description="Doctor communication score (z-score)")
    nurse_communication: float = Field(..., description="Nurse communication score (z-score)")
    overall_hospital_rating: float = Field(..., description="Overall hospital rating (z-score)")
    quietness: float = Field(..., description="Quietness score (z-score)")
    recommend_hospital: float = Field(..., description="Recommend hospital score (z-score)")
    staff_responsiveness: float = Field(..., description="Staff responsiveness score (z-score)")


class PredictionOutput(BaseModel):
    """Prediction output."""

    prediction: float = Field(
        ...,
        description="Excess Readmission Ratio (original scale, inverse Box-Cox applied)",
    )
    prediction_transformed: Optional[float] = Field(
        None,
        description="Raw model output in Box-Cox space (omitted if no lambda available)",
    )
    model_name: str
    model_version: str
    model_stage: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — reports model name, version, and stage."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_cache.get("model") is not None,
        model_name=_model_cache.get("model_name"),
        model_version=_model_cache.get("model_version"),
        model_stage=_model_cache.get("model_stage"),
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput) -> PredictionOutput:
    """Predict Excess Readmission Ratio from scaled HCAHPS features."""
    if _model_cache.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model loaded for '{SERVING_MODEL_NAME}'. "
                "Promote a model to Production or Staging in MLflow first."
            ),
        )

    try:
        # Map snake_case API fields → actual pipeline column names (spaces preserved)
        feature_values = {
            "State": data.state,
            "Care transition": data.care_transition,
            "Cleanliness": data.cleanliness,
            "Communication about medicines": data.communication_about_medicines,
            "Discharge information": data.discharge_information,
            "Doctor communication": data.doctor_communication,
            "Nurse communication": data.nurse_communication,
            "Overall hospital rating": data.overall_hospital_rating,
            "Quietness": data.quietness,
            "Recommend hospital": data.recommend_hospital,
            "Staff responsiveness": data.staff_responsiveness,
        }
        input_df = pd.DataFrame([feature_values], columns=_FEATURE_COLUMNS)
        raw_prediction = float(_model_cache["model"].predict(input_df)[0])

        # Inverse Box-Cox to return prediction in original ERR scale
        boxcox_lambda = _model_cache.get("boxcox_lambda")
        if boxcox_lambda is not None:
            prediction_original = _inverse_boxcox(raw_prediction, boxcox_lambda)
        else:
            logger.warning("boxcox_lambda not available — returning raw prediction as-is")
            prediction_original = raw_prediction

        return PredictionOutput(
            prediction=prediction_original,
            prediction_transformed=raw_prediction if boxcox_lambda is not None else None,
            model_name=_model_cache["model_name"],
            model_version=str(_model_cache["model_version"]),
            model_stage=_model_cache["model_stage"],
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
