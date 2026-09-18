"""Feature engineering stage for time-series forecasting pipelines:
lag/rolling/calendar/holiday features and the chronological train/test
split.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.utils.config import load_forecast_features_config, load_pipeline_config
from src.utils.io import load_manifest, resolve_run_path, write_manifest

logger = logging.getLogger(__name__)


def engineer_forecast_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Build lag/rolling/calendar/holiday features and split chronologically.

    Features are computed on the full series before splitting — lags and
    rolling stats legitimately span the split boundary, since that's real
    historical data a forecaster would have at prediction time. The split
    itself is purely chronological (first train_test_split fraction of rows
    by time order -> train, the rest -> test), never random. Warm-up rows
    (where the longest lag/rolling window has no history yet) are dropped,
    not filled.

    Also writes last_window.parquet — the trailing snapshot_hours rows of
    the full engineered series — for Plan 5's serving stage to seed live
    forecasts with the latest known actuals.

    Args:
        interim_dir: Directory containing cleaned interim data.
        features_dir: Output directory for feature matrices.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).

    Returns:
        Dictionary with feature matrix paths, shapes, and transform metadata.

    Raises:
        FileNotFoundError: If the interim manifest is absent.
    """
    interim_path = resolve_run_path(interim_dir, run_id)
    features_path = resolve_run_path(features_dir, run_id)
    clean_manifest = load_manifest(interim_path)  # raises FileNotFoundError if absent

    features_config = load_forecast_features_config(config_dir)
    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = pd.read_parquet(clean_manifest["output_path"])
    df[target_col] = df[target_col].astype(float)

    feature_df = df.copy()

    for lag in features_config.lags:
        feature_df[f"lag_{lag}h"] = feature_df[target_col].shift(lag)

    for window in features_config.rolling_windows:
        shifted = feature_df[target_col].shift(1)
        feature_df[f"rolling_mean_{window}h"] = shifted.rolling(window).mean()
        feature_df[f"rolling_std_{window}h"] = shifted.rolling(window).std()

    if features_config.calendar_features:
        feature_df["hour"] = feature_df.index.hour
        feature_df["day_of_week"] = feature_df.index.dayofweek
        feature_df["month"] = feature_df.index.month
        feature_df["is_weekend"] = (feature_df.index.dayofweek >= 5).astype(int)
        # Raw calendar year, fed to GBM directly (no scaling needed for a
        # tree model) as an explicit long-term trend signal — lags alone
        # only encode recent history, not multi-year drift. Matches the
        # dataset's own well-known reference approach (robikscube's XGBoost
        # tutorial for this exact dataset includes `year` in its feature set).
        feature_df["year"] = feature_df.index.year

    if features_config.holiday_features:
        calendar = USFederalHolidayCalendar()
        # USFederalHolidayCalendar's largest holiday-free gap is ~105 days
        # (Washington's Birthday in Feb to Memorial Day in late May); 200
        # days of padding comfortably exceeds that worst case.
        padding = pd.Timedelta(days=200)
        holidays = calendar.holidays(
            start=feature_df.index.min() - padding,
            end=feature_df.index.max() + padding,
        )
        normalized = feature_df.index.normalize()
        feature_df["is_holiday"] = normalized.isin(holidays).astype(int)
        holiday_arr = holidays.values
        feature_df["days_to_nearest_holiday"] = [
            int(np.abs(holiday_arr - ts).min() / np.timedelta64(1, "D")) if len(holiday_arr) else np.nan
            for ts in normalized.values
        ]

    rows_before_dropna = len(feature_df)
    feature_df = feature_df.dropna()
    rows_dropped_warmup = rows_before_dropna - len(feature_df)

    if feature_df.empty:
        raise ValueError(
            "All rows were dropped during warm-up/NaN cleanup — the feature "
            "matrix is empty. This usually means the configured lag/rolling "
            "windows are longer than the available data, or (with "
            "holiday_features enabled) the holiday lookup window doesn't "
            "reach any real holiday for this date range."
        )

    split_idx = int(len(feature_df) * pipeline_config.train_test_split)
    train_df = feature_df.iloc[:split_idx].copy()
    test_df = feature_df.iloc[split_idx:].copy()

    snapshot_df = feature_df.tail(features_config.snapshot_hours).copy()

    features_path.mkdir(parents=True, exist_ok=True)
    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"
    snapshot_path = features_path / "last_window.parquet"

    # Writes directly via to_parquet (default index=True), NOT the shared
    # write_parquet() helper in src/utils/io.py — see clean_forecast.py's
    # identical note for why.
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    snapshot_df.to_parquet(snapshot_path)

    split_cutoff = str(test_df.index.min()) if len(test_df) > 0 else None
    feature_columns = [c for c in train_df.columns if c != target_col]

    write_manifest(features_path, {
        "run_id": run_id,
        "source": "engineered forecast features",
        "stage": "feature_engineer_forecast",
        "feature_columns": feature_columns,
        "rows_dropped_warmup": rows_dropped_warmup,
        "split_cutoff": split_cutoff,
        "train": {"path": str(train_path), "rows": len(train_df), "columns": len(train_df.columns)},
        "test": {"path": str(test_path), "rows": len(test_df), "columns": len(test_df.columns)},
        "snapshot": {"path": str(snapshot_path), "rows": len(snapshot_df)},
    })

    logger.info(
        "Forecast features ready: train=%s test=%s snapshot=%d rows, %d warm-up rows dropped",
        train_df.shape, test_df.shape, len(snapshot_df), rows_dropped_warmup,
    )

    return {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "snapshot_path": str(snapshot_path),
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "feature_count": len(feature_columns),
        "rows_dropped_warmup": rows_dropped_warmup,
    }
