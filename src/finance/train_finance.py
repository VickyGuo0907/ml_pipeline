"""Training stage for finance returns/risk pipelines: fits Random
Walk/Mean/ARIMA per asset (one MLflow run per (model_type, asset) pair) plus
one cross-sectional GBM run across the whole panel, logging each model's
training-period error standard deviation (the risk-analysis requirement's
"how certain is your model?" figure, computed in-sample here - test-period
scoring, rank-IC, and champion selection are a later plan's job).

Reload contract per model_type (relevant to whichever later plan reloads
these runs to score the test period): `random_walk`/`mean` are logged via
`mlflow.pyfunc.log_model` and reload via `mlflow.pyfunc.load_model(...)`,
but the returned `PyFuncModel` wrapper only exposes `.predict(df)` -
`.forecast()` does not survive the mlflow round-trip. `arima` is logged via
`mlflow.statsmodels.log_model` and must be reloaded via
`mlflow.statsmodels.load_model(...)` (NOT `mlflow.pyfunc.load_model(...)`,
which raises an `MlflowException` about `TimeSeriesModel` inputs) to get
back the native `ARIMAResultsWrapper` with a working `.forecast(steps)`.
`cross_sectional_gbm` is logged via `mlflow.lightgbm.log_model` and must be
reloaded via `mlflow.lightgbm.load_model(...)` (NOT
`mlflow.pyfunc.load_model(...).predict(...)`, which raises a `LightGBMError`
about feature-count mismatch), then called as `.predict(df[feature_columns])`.
"""
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.statsmodels
import pandas as pd

from src.finance.model_registry import fit_arima, fit_cross_sectional_gbm, fit_mean, fit_random_walk
from src.utils.config import load_finance_models_config, load_pipeline_config
from src.utils.io import load_manifest, resolve_run_path

logger = logging.getLogger(__name__)

PER_ASSET_MODEL_TYPES = {"random_walk", "mean", "arima"}


def _training_error_std(y_train: pd.Series, fittedvalues: pd.Series) -> float:
    """Standard deviation of (actual - fitted) over the training period.

    NaN fitted rows (e.g. Mean's trailing-window warm-up) are excluded by
    pandas' default skipna behavior on Series subtraction and .std().

    Args:
        y_train: Actual training target values.
        fittedvalues: The model's in-sample fitted/predicted values, same index as y_train.

    Returns:
        Sample standard deviation (ddof=1) of the residuals.
    """
    residuals = y_train - fittedvalues
    return float(residuals.std())


def _train_one_per_asset_model(
    model_cfg: Any,
    y_train: pd.Series,
) -> tuple[Any, str]:
    """Fit one random_walk/mean/arima model and return (fitted, mlflow flavor name).

    Args:
        model_cfg: ModelConfig (name, type, hyperparameters) from models.yaml.
        y_train: This asset's training target series.

    Returns:
        (fitted_model, flavor) where flavor is one of "pyfunc"/"statsmodels",
        used by the caller to select the matching mlflow.<flavor>.log_model call.

    Raises:
        ValueError: If model_cfg.type is not one of PER_ASSET_MODEL_TYPES.
    """
    if model_cfg.type == "random_walk":
        return fit_random_walk(y_train), "pyfunc"
    if model_cfg.type == "mean":
        return fit_mean(y_train, model_cfg.hyperparameters), "pyfunc"
    if model_cfg.type == "arima":
        return fit_arima(y_train, model_cfg.hyperparameters), "statsmodels"
    raise ValueError(f"'{model_cfg.type}' is not a per-asset finance model type")


def train_finance_models(
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
) -> dict[str, Any]:
    """Train all configured finance models and log training-period error std-dev to MLflow.

    Random Walk/Mean/ARIMA are fit per asset (one MLflow run per
    (model_type, asset) pair, tagged so a later plan can aggregate them into
    a per-model-TYPE rank-IC champion). Cross-sectional GBM is fit once
    across the whole stacked panel (one MLflow run).

    Args:
        features_dir: Directory containing train/test parquet files.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).
        mlflow_tracking_uri: MLflow tracking server URI.

    Returns:
        Dictionary with per-(model_type, asset) and per-GBM MLflow run IDs
        and training-period error std-dev. Models/assets that fail to fit
        are logged and skipped, not raised.

    Raises:
        FileNotFoundError: If feature files don't exist.
    """
    features_path = resolve_run_path(features_dir, run_id)
    train_path = features_path / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    pipeline_config = load_pipeline_config(config_dir)
    models_config = load_finance_models_config(config_dir)
    target_col = pipeline_config.target.name

    train_df = pd.read_parquet(train_path)
    features_manifest = load_manifest(features_path)
    ticker_mapping: dict[str, int] = features_manifest["ticker_mapping"]
    code_to_ticker = {code: ticker for ticker, code in ticker_mapping.items()}
    feature_columns: list[str] = features_manifest["feature_columns"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    training_results: dict[str, Any] = {"run_id": run_id, "models": {}}

    for model_cfg in models_config.models:
        if model_cfg.type in PER_ASSET_MODEL_TYPES:
            for code, ticker in sorted(code_to_ticker.items()):
                asset_key = f"{model_cfg.name}_{ticker}"
                try:
                    asset_df = train_df[train_df["ticker_encoded"] == code].sort_values("date_ordinal")
                    y_train = pd.Series(
                        asset_df[target_col].to_numpy(),
                        index=pd.PeriodIndex(
                            [pd.Period(ordinal=int(n), freq="M") for n in asset_df["date_ordinal"]], freq="M",
                        ),
                    )
                    if len(y_train) < 2:
                        raise ValueError(f"Not enough training rows for {ticker} ({len(y_train)})")

                    fitted, flavor = _train_one_per_asset_model(model_cfg, y_train)

                    with mlflow.start_run(run_name=f"{run_id}_{asset_key}"):
                        mlflow.set_tags({
                            "model_name": model_cfg.name,
                            "model_type": model_cfg.type,
                            "ticker": ticker,
                            "run_id": run_id,
                            "pipeline_type": pipeline_config.pipeline_type,
                        })
                        mlflow.log_param("train_rows", len(y_train))
                        mlflow.log_params({
                            k: (v if v is not None else "null") for k, v in model_cfg.hyperparameters.items()
                        })

                        train_error_std = _training_error_std(y_train, fitted.fittedvalues)
                        mlflow.log_metric("train_error_std", train_error_std)

                        if flavor == "pyfunc":
                            mlflow.pyfunc.log_model(python_model=fitted, name="model")
                        else:
                            mlflow.statsmodels.log_model(fitted, name="model")

                        mlflow_run_id = mlflow.active_run().info.run_id

                    training_results["models"][asset_key] = {
                        "mlflow_run_id": mlflow_run_id,
                        "model_type": model_cfg.type,
                        "ticker": ticker,
                        "train_error_std": train_error_std,
                    }
                    logger.info(
                        "Trained %s: train_error_std=%.6f (%d rows)",
                        asset_key, train_error_std, len(y_train),
                    )
                except Exception as e:
                    logger.warning("Training skipped for %s: %s", asset_key, e)

        elif model_cfg.type == "cross_sectional_gbm":
            try:
                with mlflow.start_run(run_name=f"{run_id}_{model_cfg.name}"):
                    mlflow.set_tags({
                        "model_name": model_cfg.name,
                        "model_type": model_cfg.type,
                        "run_id": run_id,
                        "pipeline_type": pipeline_config.pipeline_type,
                    })
                    mlflow.log_param("feature_count", len(feature_columns))
                    mlflow.log_params({
                        k: (v if v is not None else "null") for k, v in model_cfg.hyperparameters.items()
                    })
                    mlflow.log_dict({"columns": feature_columns}, "feature_columns.json")

                    model = fit_cross_sectional_gbm(train_df, target_col, feature_columns, model_cfg.hyperparameters)
                    predictions = model.predict(train_df[feature_columns])
                    train_error_std = float((train_df[target_col] - predictions).std())
                    mlflow.log_metric("train_error_std", train_error_std)
                    mlflow.lightgbm.log_model(model, name="model")
                    mlflow_run_id = mlflow.active_run().info.run_id

                training_results["models"][model_cfg.name] = {
                    "mlflow_run_id": mlflow_run_id,
                    "model_type": model_cfg.type,
                    "train_error_std": train_error_std,
                }
                logger.info("Trained %s: train_error_std=%.6f", model_cfg.name, train_error_std)
            except Exception as e:
                logger.warning("Training skipped for %s: %s", model_cfg.name, e)

        else:
            logger.warning("Unknown finance model type '%s' for %s - skipped", model_cfg.type, model_cfg.name)

    return training_results
