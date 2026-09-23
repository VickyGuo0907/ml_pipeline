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
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import find_latest_run_id, load_manifest, resolve_run_path

logger = logging.getLogger(__name__)


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
    features_dir: str | Path,
    ticker: str,
    feature_columns: list[str],
    target_col: str = "log_return",
) -> pd.DataFrame | None:
    """Construct the feature row for the NEXT (not-yet-realized) month for
    one asset, for cross_sectional_gbm serving.

    The latest existing feature row on file (train+test's most recent row
    for this ticker) describes an ALREADY-REALIZED month — its own lag/
    volatility features are relative to ITS OWN target, not the month
    after it. This function instead builds a genuine next-period row:
    lag_N_return columns use this asset's N most-recently-realized actual
    returns (read from target_col directly, not the pre-computed lag
    columns), and trailing_Wm_vol columns are the rolling std of the W
    most-recently-realized returns — exactly what engineer_finance_features
    (Plan 2) would compute for the row immediately after the last one on
    file, without that row needing to exist yet.

    Args:
        features_dir: Pipeline features directory (e.g. data/m6_returns_risk/features).
        ticker: Asset to serve (from SERVING_FINANCE_TICKER).
        feature_columns: Exact column order the model expects — parsed for
            lag_<N>_return / trailing_<W>m_vol naming, matching Plan 2's
            engineer_finance_features column-naming convention exactly.
        target_col: Raw target column name (e.g. 'log_return') used to
            reconstruct lag/volatility features from realized values.

    Returns:
        A single-row DataFrame with feature_columns as columns, describing
        the next unrealized month — or None if no run has features for
        this ticker yet, or there isn't enough history to build every
        requested feature.
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
    if asset_rows.empty or target_col not in asset_rows.columns:
        return None

    realized = asset_rows[target_col]
    next_row: dict[str, float | int] = {}
    for col in feature_columns:
        if col == "ticker_encoded":
            next_row[col] = code
        elif col.startswith("lag_") and col.endswith("_return"):
            n = int(col.split("_")[1])
            if len(realized) < n:
                logger.warning(
                    "Not enough history to build %s for %s (need %d, have %d)", col, ticker, n, len(realized),
                )
                return None
            next_row[col] = realized.iloc[-n]
        elif col.startswith("trailing_") and col.endswith("m_vol"):
            window = int(col.split("_")[1].rstrip("m"))
            if len(realized) < window:
                logger.warning(
                    "Not enough history to build %s for %s (need %d, have %d)", col, ticker, window, len(realized),
                )
                return None
            next_row[col] = realized.tail(window).std()
        else:
            logger.warning("Don't know how to construct next-period feature '%s' for %s", col, ticker)
            return None

    return pd.DataFrame([next_row])[feature_columns]


def forecast_arima_with_target_period(model: Any, horizon_months: int) -> tuple[np.ndarray, str | None]:
    """Forecast horizon_months ahead from a fitted ARIMA model, also
    returning the real calendar month this forecast targets.

    Only ARIMA can report this — Plan 3 fits it with a real monthly
    PeriodIndex, which survives the mlflow.statsmodels round-trip and
    .forecast()'s own output index. random_walk/mean lose their index
    across the mlflow.pyfunc round-trip (Plan 3/4's established contract —
    .predict(model_input) only ever returns a plain array), so no
    target_period is derivable for those two types; callers should treat
    a None return as "relative horizon only, no absolute calendar label."

    Args:
        model: Object returned by src.finance.rank_ic.reload_model() for
            model_type == "arima".
        horizon_months: Number of future months to forecast.

    Returns:
        (predictions, target_period) — predictions is an array of
        horizon_months values; target_period is the calendar month string
        the LAST predicted value targets (e.g. "2024-05"), or None if the
        forecast result carries no usable index.
    """
    series = model.forecast(steps=horizon_months)
    target_period = str(series.index[-1]) if hasattr(series, "index") and len(series.index) else None
    return np.asarray(series), target_period
