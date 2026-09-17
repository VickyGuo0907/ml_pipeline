"""Shared rolling-origin scoring utilities for time-series forecasting
pipelines: MAPE, origin selection, and per-model-family scoring.

Used in two places: Plan 3's training-stage CV (origins sampled from the
training set, statsmodels scored with dynamic=True since origins are
in-sample) and Plan 4's evaluation-stage test scoring (origins sampled from
the test set, statsmodels scored with dynamic=False since origins are
out-of-sample — passing dynamic=origin for an out-of-sample point raises for
ETSModel and is a no-op for SARIMAX, so dynamic=False is the one setting
that works correctly for both model types on out-of-sample data).

CAVEAT (dynamic=False / out-of-sample scoring only): statsmodels' get_prediction
does not re-anchor a fresh horizon-length forecast at each origin the way
score_gbm_origins does. It produces one continuous forecast starting right
after the fitted sample's end, and start=origin/end=end just slices that
continuous forecast. That means the model's effective forecast lead time at a
given origin is (origin - train_end) + horizon_hours, not horizon_hours — it
grows the further an origin sits past the end of training. So for
dynamic=False, test_mape from this function is NOT strictly horizon-anchored
the way GBM's score_gbm_origins is, and the two are not directly comparable
in an apples-to-apples sense. This is a known limitation, not a bug; a proper
fix (re-anchoring each origin via e.g. SARIMAXResults.append(refit=False) or
ETSModel(hist).smooth(params)) is a real algorithmic change, deliberately out
of scope here — see evaluate_forecast.py's module docstring for the same
caveat applied to test_mape_mean. This caveat does NOT apply to dynamic=True
(in-sample CV), where sharing one origin set does keep both model families on
identical, horizon-anchored cutoffs.
"""
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.recursive import recursive_forecast


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error, as a percentage (0-100+ scale).

    Safe for PJM load values (always well above zero); no zero-guard needed
    for this pipeline's target column.
    """
    actual_arr = actual.to_numpy(dtype=float)
    predicted_arr = predicted.to_numpy(dtype=float)
    return float(np.mean(np.abs((actual_arr - predicted_arr) / actual_arr)) * 100)


def select_cv_origins(
    index: pd.DatetimeIndex,
    n_windows: int,
    horizon_hours: int,
    min_history_hours: int = 1,
) -> list[pd.Timestamp]:
    """Pick up to n_windows evenly-spaced rolling-origin timestamps from index.

    Each origin leaves at least horizon_hours of real data after it (inclusive
    of the origin itself), so the true values needed to score that window
    actually exist in index. Each origin also leaves at least min_history_hours
    of real data before it — never index[0] itself — so GBM scoring
    (score_gbm_origins) has enough real history before the origin for its
    longest configured lag/rolling window to be fully populated (a naive
    1-hour margin leaves e.g. lag_168h/rolling_mean_168h as NaN at the first
    origin, which LightGBM still predicts on, producing a garbage outlier
    score). statsmodels scoring doesn't need this margin (get_prediction
    relies on the already-fitted model, not rebuilt lag features), but
    sharing one origin set keeps both model families scored at identical
    cutoff timestamps, so the margin must satisfy GBM's stricter requirement.
    Note: identical cutoffs is NOT the same as an apples-to-apples comparison
    for out-of-sample (dynamic=False) scoring — see score_statsmodels_origins'
    module-level caveat below for why statsmodels' effective forecast horizon
    at a shared cutoff is not horizon_hours the way GBM's is.

    Args:
        index: The DatetimeIndex to pick origins from (training set for CV,
            test set for final evaluation).
        n_windows: Number of origins to pick.
        horizon_hours: Forecast horizon — origins within horizon_hours - 1
            hours of the end of index are excluded.
        min_history_hours: Minimum real history required before an origin
            (e.g. max of configured lags/rolling_windows). Defaults to 1.

    Returns:
        List of up to n_windows Timestamps (fewer if the series is too short
        to support that many distinct positions), or an empty list if the
        series can't support even one full horizon plus min_history_hours of
        history.
    """
    usable_start = max(min_history_hours, 1)
    usable_count = len(index) - horizon_hours + 1
    if usable_count <= usable_start:
        return []
    usable = index[usable_start:usable_count]
    n = min(n_windows, len(usable))
    positions = np.linspace(0, len(usable) - 1, n).astype(int)
    return [usable[p] for p in sorted(set(positions))]


def score_statsmodels_origins(
    fitted: Any,
    y: pd.Series,
    origins: list[pd.Timestamp],
    horizon_hours: int,
    dynamic: bool = True,
    exog: pd.Series | pd.DataFrame | None = None,
) -> list[float]:
    """Score a fitted ETS/SARIMAX result at each origin via get_prediction.

    Args:
        fitted: Fitted ETSResultsWrapper or SARIMAXResultsWrapper.
        y: The full target series covering both the fitted range and every
            origin/horizon being scored (the training series alone for
            in-sample CV; train+test concatenated for out-of-sample test
            evaluation).
        origins: Rolling-origin timestamps from select_cv_origins.
        horizon_hours: Forecast horizon per origin.
        dynamic: True for origins WITHIN the fitted sample (passes
            dynamic=origin, so get_prediction simulates forward instead of
            reading true historical values it already has). False for
            origins OUTSIDE the fitted sample (omits dynamic= entirely —
            required for ETSModel, which raises "Cannot anchor simulation
            outside of the sample" if dynamic=origin is passed for an
            out-of-sample point; harmless but pointless for SARIMAX, which
            just warns "has no effect" and produces the same result either
            way, since every out-of-sample prediction is already a genuine
            simulation). See this module's docstring for a caveat on what
            dynamic=False scoring actually measures.
        exog: Exogenous regressor values covering the full range in `y`,
            required only when `fitted` was fit with exog (e.g. SARIMAX with
            use_holiday_exog) AND dynamic=False (out-of-sample). statsmodels'
            out-of-sample get_prediction needs exog for every step from the
            end of the fitted sample through `end` — not just origin:end — so
            this function slices `exog` from just past the fitted sample's
            end through each origin's `end` itself. Ignored when dynamic=True
            (in-sample predictions reuse the exog already stored in `fitted`)
            or when the model has no exog component.

    Returns:
        List of MAPE scores, one per origin that had enough trailing data.
    """
    scores = []
    # Only needed for the dynamic=False + exog case; resolved once up front
    # since it's the same for every origin (the fitted sample's end doesn't
    # move between origins).
    train_end = pd.Timestamp(fitted.model.data.dates[-1]) if exog is not None else None
    for origin in origins:
        end = origin + pd.Timedelta(hours=horizon_hours - 1)
        if end > y.index[-1]:
            continue
        if dynamic:
            pred = fitted.get_prediction(start=origin, end=end, dynamic=origin).predicted_mean
        elif exog is not None:
            exog_window = exog.loc[train_end + pd.Timedelta(hours=1): end]
            pred = fitted.get_prediction(start=origin, end=end, exog=exog_window).predicted_mean
        else:
            pred = fitted.get_prediction(start=origin, end=end).predicted_mean
        actual = y.loc[origin:end]
        scores.append(mape(actual, pred))
    return scores


def score_gbm_origins(
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    origins: list[pd.Timestamp],
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    calendar_features: bool,
    holiday_features: bool,
) -> list[float]:
    """Score a fitted GBM model at each origin via recursive_forecast.

    Args:
        model: Fitted sklearn-compatible estimator.
        df: Feature matrix (DatetimeIndex) covering both the fitted range and
            every origin/horizon being scored, including target_col — the
            training feature matrix alone for in-sample CV; train+test
            concatenated for out-of-sample test evaluation.
        target_col: Target column name.
        feature_columns: Exact column order the model expects.
        origins: Rolling-origin timestamps from select_cv_origins.
        horizon_hours: Forecast horizon per origin.
        lags, rolling_windows, calendar_features, holiday_features: Feature
            config, passed through to recursive_forecast.

    Returns:
        List of MAPE scores, one per origin that had enough trailing data
        and at least one hour of history before it.
    """
    scores = []
    y = df[target_col]
    for origin in origins:
        end = origin + pd.Timedelta(hours=horizon_hours - 1)
        if end > y.index[-1]:
            continue
        history = y.loc[: origin - pd.Timedelta(hours=1)]
        if history.empty:
            continue
        pred = recursive_forecast(
            model, history, horizon_hours, feature_columns,
            lags, rolling_windows, calendar_features, holiday_features,
        )
        actual = y.loc[origin:end]
        scores.append(mape(actual, pred))
    return scores
