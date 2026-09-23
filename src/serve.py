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
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.statsmodels
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, RootModel

from src.finance.rank_ic import reload_model
from src.finance.serve_finance import forecast_per_asset_model, load_latest_asset_features
from src.forecasting.serve_forecast import forecast_with_gbm, forecast_with_statsmodels, load_latest_snapshot
from src.utils.config import load_forecast_features_config, load_orchestration_config

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
SERVING_MODEL_NAME = os.environ.get("SERVING_MODEL_NAME")
# Only required for a cross_sectional_gbm finance deployment — that model
# isn't tied to one asset by training (unlike random_walk/mean/arima, whose
# asset is already fixed by which registered model name SERVING_MODEL_NAME
# points at), so this server needs to be told which asset's latest realized
# features to use.
SERVING_FINANCE_TICKER = os.environ.get("SERVING_FINANCE_TICKER")

FEATURE_COLUMNS_ARTIFACT = "feature_columns.json"

# Global model cache — populated on startup. feature_columns and target_col come
# from the served model's own training run, not from any hardcoded schema here.
_model_cache: dict[str, Any] = {
    "model": None,
    "model_name": None,
    "model_version": None,
    "model_stage": None,
    "boxcox_lambda": None,
    "boxcox_offset": None,
    "feature_columns": None,
    "target_col": None,
    # Forecasting-specific — None/False for every tabular model, so /health,
    # /schema, and POST /predict behave exactly as before for existing
    # deployments. Populated only when _load_model detects cv_mape_mean.
    "is_forecasting": False,
    "forecast_model_type": None,
    "pipeline_type": None,
    "lags": None,
    "rolling_windows": None,
    "calendar_features": None,
    "holiday_features": None,
    # Finance-specific — False/None for every tabular and forecasting
    # model. Populated only when _load_model detects train_error_std.
    "is_finance": False,
    "finance_model_type": None,
    "ticker": None,
    "test_error_std": None,
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
            run = client.get_run(version.run_id)

            boxcox_lambda: float | None = None
            lambda_str = run.data.params.get("boxcox_lambda")
            if lambda_str is not None:
                boxcox_lambda = float(lambda_str)

            boxcox_offset: float | None = None
            offset_str = run.data.params.get("boxcox_offset")
            if offset_str is not None:
                boxcox_offset = float(offset_str)

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

            # A cv_mape_mean metric is logged only by the forecasting training
            # stage (src/forecasting/train_forecast.py) — never by the tabular
            # one. This is the sole signal used to route to the forecasting
            # load path, so no pipeline name is ever hardcoded here.
            is_forecasting = "cv_mape_mean" in run.data.metrics
            forecast_model_type: str | None = None
            pipeline_type: str | None = None
            lags: list[int] | None = None
            rolling_windows: list[int] | None = None
            calendar_features: bool | None = None
            holiday_features: bool | None = None

            # A train_error_std metric is logged only by the finance
            # training stage (src/finance/train_finance.py) — never by the
            # tabular or forecasting ones. Mutually exclusive with
            # is_forecasting's cv_mape_mean check by construction (verified:
            # neither pipeline family logs the other's metric name).
            is_finance = "train_error_std" in run.data.metrics
            finance_model_type: str | None = None
            ticker: str | None = None
            test_error_std: float | None = None

            if is_forecasting:
                forecast_model_type = run.data.tags.get("model_type")
                pipeline_type = run.data.tags.get("pipeline_type")
                if forecast_model_type in ("ets", "sarimax"):
                    model = mlflow.statsmodels.load_model(model_uri)
                elif forecast_model_type == "gbm":
                    model = mlflow.lightgbm.load_model(model_uri)
                else:
                    logger.warning(
                        "Unknown forecasting model_type '%s' for %s v%s — cannot select "
                        "an MLflow flavor to load it with.", forecast_model_type, model_name, version.version,
                    )
                    continue
                if pipeline_type:
                    try:
                        features_cfg = load_forecast_features_config(f"config/{pipeline_type}")
                        lags = features_cfg.lags
                        rolling_windows = features_cfg.rolling_windows
                        calendar_features = features_cfg.calendar_features
                        holiday_features = features_cfg.holiday_features
                    except Exception as e:
                        logger.warning(
                            "Could not load forecast feature config for pipeline '%s': %s", pipeline_type, e,
                        )
            elif is_finance:
                finance_model_type = run.data.tags.get("model_type")
                ticker = run.data.tags.get("ticker")  # None for cross_sectional_gbm
                pipeline_type = run.data.tags.get("pipeline_type")
                if finance_model_type not in ("random_walk", "mean", "arima", "cross_sectional_gbm"):
                    logger.warning(
                        "Unknown finance model_type '%s' for %s v%s — cannot select "
                        "an MLflow flavor to load it with.", finance_model_type, model_name, version.version,
                    )
                    continue
                model = reload_model(finance_model_type, version.run_id, MLFLOW_TRACKING_URI)
                test_error_std_str = version.tags.get("test_error_std")
                if test_error_std_str is not None:
                    test_error_std = float(test_error_std_str)
            else:
                model = mlflow.pyfunc.load_model(model_uri)

            logger.info(
                "Loaded %s v%s from %s (target_col=%s, %s features, is_forecasting=%s, is_finance=%s, "
                "boxcox_lambda=%s)",
                model_name, version.version, stage, target_col,
                len(feature_columns) if feature_columns else "unknown", is_forecasting, is_finance, boxcox_lambda,
            )
            return {
                "model": model,
                "model_name": model_name,
                "model_version": version.version,
                "model_stage": stage,
                "boxcox_lambda": boxcox_lambda,
                "boxcox_offset": boxcox_offset,
                "feature_columns": feature_columns,
                "target_col": target_col,
                "is_forecasting": is_forecasting,
                "forecast_model_type": forecast_model_type,
                "pipeline_type": pipeline_type,
                "lags": lags,
                "rolling_windows": rolling_windows,
                "calendar_features": calendar_features,
                "holiday_features": holiday_features,
                "is_finance": is_finance,
                "finance_model_type": finance_model_type,
                "ticker": ticker,
                "test_error_std": test_error_std,
            }
        except Exception as e:
            logger.debug("No %s model for %s: %s", stage, model_name, e)
            continue

    logger.warning("No Production or Staging model found for '%s'", model_name)
    return None


def _inverse_boxcox(value: float, lam: float, offset: float = 0.0) -> float:
    """Inverse Box-Cox transform to return a prediction to its original target scale.

    The forward transform shifted the target by `offset` before Box-Cox, so the
    inverse must subtract it back off. Older models without a logged offset
    default to 0.0 (their forward shift was only ~1e-6, a no-op in practice).

    Args:
        value: Box-Cox transformed prediction.
        lam: Box-Cox lambda used during feature engineering.
        offset: Shift applied to the target before Box-Cox (from the manifest).

    Returns:
        Prediction in the original target scale (clamped to ≥ 0).
    """
    if lam == 0:
        return math.exp(value) - offset
    return max(0.0, (value * lam + 1) ** (1.0 / lam) - offset)


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


class ForecastPoint(BaseModel):
    """One timestamped forecast value."""

    timestamp: str
    prediction: float


class ForecastOutput(BaseModel):
    """Multi-step forecast output."""

    predictions: list[ForecastPoint]
    horizon_hours: int
    model_name: str
    model_version: str
    model_stage: str
    forecast_model_type: str


class FinanceForecastOutput(BaseModel):
    """Point forecast for one asset's log-return horizon_months out, plus
    the model's logged test-period forecast-error std-dev."""

    prediction: float = Field(..., description="Predicted log-return at horizon_months out")
    horizon_months: int
    ticker: Optional[str] = Field(None, description="Asset this forecast is for")
    test_error_std: Optional[float] = Field(
        None, description="Model's logged test-period forecast-error std-dev — how certain is this model?",
    )
    model_name: str
    model_version: str
    model_stage: str
    finance_model_type: str


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
        boxcox_offset = _model_cache.get("boxcox_offset") or 0.0
        if boxcox_lambda is not None:
            prediction_original = _inverse_boxcox(raw_prediction, boxcox_lambda, boxcox_offset)
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


MAX_FORECAST_HORIZON_HOURS = 168  # 1 week — a sane upper bound on a single request


@app.get("/predict/forecast", response_model=ForecastOutput)
async def predict_forecast(
    horizon_hours: int = Query(..., gt=0, le=MAX_FORECAST_HORIZON_HOURS),
) -> ForecastOutput:
    """Forecast horizon_hours ahead using whichever forecasting model
    SERVING_MODEL_NAME points at. Returns 400 if the currently loaded model
    is not a forecasting model — use POST /predict for tabular models.

    MAX_FORECAST_HORIZON_HOURS (168) is a request-size cap, not an accuracy
    guarantee: each model's rolling-origin CV only validates accuracy out to
    its own models.yaml evaluation.horizon_hours (24 for pjm_load_forecast's
    models today). A request beyond that horizon still returns a forecast —
    the model extrapolates further than its CV evidence covers, and error is
    expected to grow with distance past that point.
    """
    if not _model_cache.get("is_forecasting"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Loaded model '{SERVING_MODEL_NAME}' is not a forecasting model. "
                "Use POST /predict for tabular models."
            ),
        )
    if _model_cache.get("model") is None:
        raise HTTPException(status_code=503, detail=f"No model loaded for '{SERVING_MODEL_NAME}'.")

    forecast_model_type = _model_cache["forecast_model_type"]
    try:
        if forecast_model_type in ("ets", "sarimax"):
            series = forecast_with_statsmodels(_model_cache["model"], horizon_hours)
        elif forecast_model_type == "gbm":
            pipeline_type = _model_cache.get("pipeline_type")
            snapshot = None
            if pipeline_type:
                try:
                    features_dir = load_orchestration_config(f"config/{pipeline_type}").directories.features
                    snapshot = load_latest_snapshot(features_dir)
                except Exception as e:
                    logger.warning(
                        "Could not load orchestration config for pipeline '%s': %s", pipeline_type, e,
                    )
            if snapshot is None:
                raise HTTPException(
                    status_code=503,
                    detail="No feature snapshot available yet — run the pipeline's DAG at least once.",
                )
            series = forecast_with_gbm(
                _model_cache["model"], snapshot, _model_cache["target_col"], _model_cache["feature_columns"],
                horizon_hours, _model_cache["lags"], _model_cache["rolling_windows"],
                _model_cache["calendar_features"], _model_cache["holiday_features"],
            )
        else:
            raise HTTPException(status_code=500, detail=f"Unknown forecast_model_type '{forecast_model_type}'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

    return ForecastOutput(
        predictions=[
            ForecastPoint(timestamp=str(ts), prediction=float(val)) for ts, val in series.items()
        ],
        horizon_hours=horizon_hours,
        model_name=_model_cache["model_name"],
        model_version=str(_model_cache["model_version"]),
        model_stage=_model_cache["model_stage"],
        forecast_model_type=forecast_model_type,
    )


MAX_FINANCE_HORIZON_MONTHS = 24  # 2 years — a sane upper bound on a single request


@app.get("/predict/finance-return", response_model=FinanceForecastOutput)
async def predict_finance_return(
    horizon_months: int = Query(..., gt=0, le=MAX_FINANCE_HORIZON_MONTHS),
) -> FinanceForecastOutput:
    """Point forecast of the log-return horizon_months out, using whichever
    finance model SERVING_MODEL_NAME points at, plus that model's logged
    test-period forecast-error std-dev (the risk-analysis "how certain is
    your model?" figure). Returns 400 if the currently loaded model is not
    a finance model.

    cross_sectional_gbm deployments require SERVING_FINANCE_TICKER to be
    set and only support horizon_months=1 — see this module's SERVING_FINANCE_TICKER
    comment and this plan's Global Constraints for why.
    """
    if not _model_cache.get("is_finance"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Loaded model '{SERVING_MODEL_NAME}' is not a finance model. "
                "Use POST /predict for tabular models or GET /predict/forecast for forecasting models."
            ),
        )
    if _model_cache.get("model") is None:
        raise HTTPException(status_code=503, detail=f"No model loaded for '{SERVING_MODEL_NAME}'.")

    finance_model_type = _model_cache["finance_model_type"]
    try:
        if finance_model_type in ("random_walk", "mean", "arima"):
            predictions = forecast_per_asset_model(_model_cache["model"], finance_model_type, horizon_months)
            prediction = float(predictions[-1])
        elif finance_model_type == "cross_sectional_gbm":
            if not SERVING_FINANCE_TICKER:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "SERVING_FINANCE_TICKER environment variable is not set — a "
                        "cross_sectional_gbm deployment isn't tied to one asset by "
                        "training, so this server needs to be told which asset's "
                        "features to use."
                    ),
                )
            if horizon_months != 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "cross_sectional_gbm serving only supports horizon_months=1 "
                        "in this pipeline — it predicts from the latest known "
                        "realized feature values and has no recursive multi-step "
                        "path. Request horizon_months=1."
                    ),
                )
            pipeline_type = _model_cache.get("pipeline_type")
            feature_columns = _model_cache.get("feature_columns")
            features_dir = (
                load_orchestration_config(f"config/{pipeline_type}").directories.features if pipeline_type else None
            )
            feature_row = (
                load_latest_asset_features(features_dir, SERVING_FINANCE_TICKER, feature_columns)
                if features_dir and feature_columns else None
            )
            if feature_row is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No feature data available yet for '{SERVING_FINANCE_TICKER}' — "
                        "run the pipeline's DAG at least once."
                    ),
                )
            prediction = float(_model_cache["model"].predict(feature_row)[0])
        else:
            raise HTTPException(status_code=500, detail=f"Unknown finance model type '{finance_model_type}'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Finance forecast failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Finance forecast failed: {str(e)}")

    return FinanceForecastOutput(
        prediction=prediction,
        horizon_months=horizon_months,
        ticker=_model_cache.get("ticker") or SERVING_FINANCE_TICKER,
        test_error_std=_model_cache.get("test_error_std"),
        model_name=_model_cache["model_name"],
        model_version=str(_model_cache["model_version"]),
        model_stage=_model_cache["model_stage"],
        finance_model_type=finance_model_type,
    )
