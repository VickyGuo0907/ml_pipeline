"""FastAPI serving endpoint for ML model predictions — pipeline-agnostic.

The request schema is not hardcoded: it is derived at load time from the served
model's own training run. src/train.py logs each run's exact feature column list
(feature_columns.json artifact) and target column name (target_col param), so this
module works unmodified for any pipeline's model — the feature set, feature count,
and target name all come from MLflow, never from code here.

Model loading order:
  1. Try Production stage for the configured model name
  2. Fall back to Staging (useful during development before manual promotion)

Predictions are inverse Box-Cox transformed back to the model's original target
scale using the lambda logged during training, when that pipeline used Box-Cox.

Environment variables:
  MLFLOW_TRACKING_URI  — MLflow server (default: http://mlflow-server:5000)
  SERVING_MODEL_NAME   — which registered model to serve. Required, no default —
                         this module is pipeline-agnostic and has no basis for
                         guessing which pipeline's model an operator wants, so an
                         unset value fails loudly at startup instead of silently
                         serving whatever pipeline a previous default pointed at.
                         Must be a full pipeline-qualified name from that
                         pipeline's models.yaml (e.g.
                         "hospital_readmission_lagged_lightgbm_gbm") — model names
                         are prefixed per pipeline to keep MLflow's shared
                         registry namespace collision-free. See .env.example for
                         the full list of valid names.
"""
import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, RootModel

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
SERVING_MODEL_NAME = os.environ.get("SERVING_MODEL_NAME")

FEATURE_COLUMNS_ARTIFACT = "feature_columns.json"

# Global model cache — populated on startup. feature_columns and target_col come
# from the served model's own training run, not from any hardcoded schema here.
_model_cache: dict[str, Any] = {
    "model": None,
    "model_name": None,
    "model_version": None,
    "model_stage": None,
    "boxcox_lambda": None,
    "feature_columns": None,
    "target_col": None,
}


def _load_model(model_name: str) -> dict[str, Any] | None:
    """Load model from MLflow registry, trying Production then Staging.

    Also fetches the exact feature column list (feature_columns.json artifact) and
    target column name (target_col param) logged by src/train.py for this model's
    source run — this is what lets /predict validate and order input dynamically
    for whichever pipeline trained this model, without any hardcoded schema.

    Args:
        model_name: Registered model name (from SERVING_MODEL_NAME env var).

    Returns:
        Dict with model, version, stage, boxcox_lambda, feature_columns, and
        target_col — or None if no Production/Staging version was found.
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
            run = client.get_run(version.run_id)

            boxcox_lambda: float | None = None
            lambda_str = run.data.params.get("boxcox_lambda")
            if lambda_str is not None:
                boxcox_lambda = float(lambda_str)

            target_col = run.data.params.get("target_col")

            feature_columns: list[str] | None = None
            try:
                local_path = client.download_artifacts(version.run_id, FEATURE_COLUMNS_ARTIFACT)
                import json
                with open(local_path) as f:
                    feature_columns = json.load(f)["columns"]
            except Exception as e:
                logger.warning(
                    "No %s artifact for %s v%s (run %s) — model was likely trained "
                    "before serving metadata was added. Retrain to enable /predict. "
                    "Underlying error: %s",
                    FEATURE_COLUMNS_ARTIFACT, model_name, version.version, version.run_id, e,
                )

            logger.info(
                "Loaded %s v%s from %s (target_col=%s, %s features, boxcox_lambda=%s)",
                model_name, version.version, stage, target_col,
                len(feature_columns) if feature_columns else "unknown", boxcox_lambda,
            )
            return {
                "model": model,
                "model_name": model_name,
                "model_version": version.version,
                "model_stage": stage,
                "boxcox_lambda": boxcox_lambda,
                "feature_columns": feature_columns,
                "target_col": target_col,
            }
        except Exception as e:
            logger.debug("No %s model for %s: %s", stage, model_name, e)
            continue

    logger.warning("No Production or Staging model found for '%s'", model_name)
    return None


def _inverse_boxcox(value: float, lam: float) -> float:
    """Inverse Box-Cox transform to return a prediction to its original target scale.

    Args:
        value: Box-Cox transformed prediction.
        lam: Box-Cox lambda used during feature engineering.

    Returns:
        Prediction in the original target scale (clamped to ≥ 0).
    """
    if lam == 0:
        return math.exp(value)
    return max(0.0, (value * lam + 1) ** (1.0 / lam))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown.

    Raises:
        RuntimeError: If SERVING_MODEL_NAME is unset. This module is
            pipeline-agnostic by design — there's no correct guess for which
            pipeline's model to serve, so a missing value fails startup
            loudly instead of silently falling back to some other
            deployment's model name.
    """
    if not SERVING_MODEL_NAME:
        raise RuntimeError(
            "SERVING_MODEL_NAME environment variable is not set. This server "
            "doesn't default to any pipeline's model — set it to a full "
            "pipeline-qualified registered model name (e.g. "
            "'hospital_readmission_lagged_lightgbm_gbm'). See .env.example."
        )
    result = _load_model(SERVING_MODEL_NAME)
    if result:
        _model_cache.update(result)
    yield
    _model_cache.clear()


app = FastAPI(
    title="ML Pipeline Prediction Server",
    description=(
        "Serves a registered MLflow model from any pipeline in this project. "
        "The required input features, their count, and the predicted target all "
        "depend on which model SERVING_MODEL_NAME points at — call GET /schema "
        "after startup to see exactly what this deployment expects and predicts."
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
    target_col: Optional[str] = None


class SchemaResponse(BaseModel):
    """Describes the input/output contract of the currently loaded model."""

    model_name: Optional[str] = None
    model_version: Optional[str] = None
    target_col: Optional[str] = None
    required_features: Optional[list[str]] = None
    boxcox_applied: bool = False


# Prediction input is an arbitrary {feature_name: value} object rather than a fixed
# set of Pydantic fields — the feature set is only known once a model is loaded
# (see _model_cache["feature_columns"]), and differs per pipeline. Required-field
# validation happens in predict() below, against that model's own trained schema.
class PredictionInput(RootModel[dict[str, float]]):
    """Feature values keyed by their exact trained column name (see GET /schema)."""


class PredictionOutput(BaseModel):
    """Prediction output."""

    prediction: float = Field(
        ...,
        description="Prediction in the original target scale (inverse Box-Cox applied if used)",
    )
    prediction_transformed: Optional[float] = Field(
        None,
        description="Raw model output in Box-Cox space (omitted if no lambda available)",
    )
    target_col: Optional[str] = Field(None, description="Name of the target this model predicts")
    model_name: str
    model_version: str
    model_stage: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — reports model name, version, stage, and predicted target."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_cache.get("model") is not None,
        model_name=_model_cache.get("model_name"),
        model_version=_model_cache.get("model_version"),
        model_stage=_model_cache.get("model_stage"),
        target_col=_model_cache.get("target_col"),
    )


@app.get("/schema", response_model=SchemaResponse)
async def schema() -> SchemaResponse:
    """Report the currently loaded model's required input features and target.

    Call this before /predict — the required feature set depends entirely on
    which model SERVING_MODEL_NAME points at.
    """
    return SchemaResponse(
        model_name=_model_cache.get("model_name"),
        model_version=_model_cache.get("model_version"),
        target_col=_model_cache.get("target_col"),
        required_features=_model_cache.get("feature_columns"),
        boxcox_applied=_model_cache.get("boxcox_lambda") is not None,
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput) -> PredictionOutput:
    """Predict using whichever model SERVING_MODEL_NAME points at.

    Request body is a flat JSON object of {feature_name: value}, using the exact
    column names reported by GET /schema. Extra/unknown keys are ignored; missing
    required keys return 422.
    """
    if _model_cache.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model loaded for '{SERVING_MODEL_NAME}'. "
                "Promote a model to Production or Staging in MLflow first."
            ),
        )

    feature_columns = _model_cache.get("feature_columns")
    if not feature_columns:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{SERVING_MODEL_NAME}' has no recorded feature schema "
                f"(missing {FEATURE_COLUMNS_ARTIFACT} artifact) — it was likely "
                "trained before serving metadata was added. Retrain to enable /predict."
            ),
        )

    payload = data.root
    missing = [c for c in feature_columns if c not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature(s): {missing}. See GET /schema for the full list.",
        )

    try:
        input_df = pd.DataFrame([{c: payload[c] for c in feature_columns}], columns=feature_columns)
        raw_prediction = float(_model_cache["model"].predict(input_df)[0])

        boxcox_lambda = _model_cache.get("boxcox_lambda")
        if boxcox_lambda is not None:
            prediction_original = _inverse_boxcox(raw_prediction, boxcox_lambda)
        else:
            prediction_original = raw_prediction

        return PredictionOutput(
            prediction=prediction_original,
            prediction_transformed=raw_prediction if boxcox_lambda is not None else None,
            target_col=_model_cache.get("target_col"),
            model_name=_model_cache["model_name"],
            model_version=str(_model_cache["model_version"]),
            model_stage=_model_cache["model_stage"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
