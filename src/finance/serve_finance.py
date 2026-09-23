"""Live finance model serving: given a model already reloaded via
src.finance.rank_ic.reload_model, produce a point forecast.

random_walk/mean/arima forecast natively from their own persisted fitted
state (same as Plan 4's evaluation-stage scoring) — trustworthiness
degrades the longer it's been since training, addressed operationally via
retraining cadence, not architecturally here (same acknowledged limitation
pjm_load_forecast's serving already carries for its own statsmodels models).

cross_sectional_gbm serves from the latest known REALIZED feature row for
one configured asset (no recursive multi-step path — see this plan's
Global Constraints for why that's explicitly out of scope).
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import find_latest_run_id, load_manifest, resolve_run_path


def forecast_per_asset_model(model: Any, model_type: str, horizon_months: int) -> np.ndarray:
    """Forecast horizon_months ahead from an already-reloaded per-asset model.

    Args:
        model: Object returned by src.finance.rank_ic.reload_model() for
            model_type in {random_walk, mean, arima}.
        model_type: One of random_walk, mean, arima.
        horizon_months: Number of future months to forecast.

    Returns:
        Array of horizon_months predicted values, in chronological order.
    """
    if model_type == "arima":
        return np.asarray(model.forecast(steps=horizon_months))
    # random_walk/mean: reloaded via mlflow.pyfunc.load_model, whose
    # PyFuncModel wrapper's .predict(model_input) determines output length
    # from model_input's row count — content is ignored by these two types
    # (Plan 3/4's already-established contract).
    return np.asarray(model.predict(pd.DataFrame(index=range(horizon_months))))


def load_latest_asset_features(
    features_dir: str | Path, ticker: str, feature_columns: list[str],
) -> pd.DataFrame | None:
    """Load the most recent known feature row for one asset, for
    cross_sectional_gbm serving.

    Args:
        features_dir: Pipeline features directory (e.g. data/m6_returns_risk/features).
        ticker: Asset to serve (from SERVING_FINANCE_TICKER).
        feature_columns: Exact column order the model expects.

    Returns:
        A single-row DataFrame with feature_columns as columns — the most
        recent (train+test combined) row for this asset — or None if no
        run has features for this ticker yet.
    """
    latest_run_id = find_latest_run_id(features_dir)
    if latest_run_id is None:
        return None
    run_path = resolve_run_path(features_dir, latest_run_id)
    try:
        manifest = load_manifest(run_path)
    except FileNotFoundError:
        return None
    ticker_mapping: dict[str, int] = manifest.get("ticker_mapping", {})
    if ticker not in ticker_mapping:
        return None
    code = ticker_mapping[ticker]

    frames = []
    for name in ("train.parquet", "test.parquet"):
        path = run_path / name
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    asset_rows = combined[combined["ticker_encoded"] == code].sort_values("date_ordinal")
    if asset_rows.empty:
        return None
    return asset_rows.tail(1)[feature_columns]
