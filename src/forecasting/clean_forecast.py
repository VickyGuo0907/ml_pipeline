"""Clean stage for time-series forecasting pipelines: parse, sort, dedupe,
and gap-handle a continuous hourly series.
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_forecast_cleaning_config, load_pipeline_config
from src.utils.io import READERS, load_manifest, resolve_run_path, write_manifest

logger = logging.getLogger(__name__)


def _detect_gaps(is_missing: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Group a boolean missing-hour mask into contiguous gap runs.

    Args:
        is_missing: Boolean Series indexed by a complete hourly DatetimeIndex,
            True where that hour's value is missing.

    Returns:
        List of (gap_start, gap_end, length_hours) for each contiguous run
        of missing hours, in chronological order.
    """
    if not is_missing.any():
        return []
    missing_ts = is_missing[is_missing].index.to_series()
    new_run = missing_ts.diff() != pd.Timedelta(hours=1)
    group_id = new_run.cumsum()
    gaps = []
    for _, group in missing_ts.groupby(group_id):
        start, end = group.iloc[0], group.iloc[-1]
        length_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
        gaps.append((start, end, length_hours))
    return gaps


def clean_forecast_data(
    raw_dir: str | Path,
    interim_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Parse, sort, dedupe, and gap-handle the raw hourly series.

    Reindexes to a complete hourly DatetimeIndex spanning the series' min to
    max timestamp. Gaps at or under cleaning.yaml's max_gap_hours are filled
    (interpolate or ffill, per fill_strategy); longer gaps fail loud rather
    than being silently smoothed over, since a multi-day outage papered over
    would corrupt lag features for weeks afterward.

    Args:
        raw_dir: Base directory containing raw data.
        interim_dir: Output directory for cleaned data.
        run_id: Run identifier to locate/version data.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).

    Returns:
        Dictionary with cleaning statistics.

    Raises:
        FileNotFoundError: If the raw manifest is absent.
        ValueError: If the raw directory doesn't have exactly one file, the
            file's format is unsupported, or a gap exceeds max_gap_hours.
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    interim_path = resolve_run_path(interim_dir, run_id)
    manifest = load_manifest(raw_path)  # raises FileNotFoundError if absent

    files = list(manifest.get("files", {}))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one raw file for a forecasting pipeline, found "
            f"{len(files)}: {files}"
        )
    filename = files[0]
    reader = READERS.get(Path(filename).suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file format: {filename}")

    cleaning_config = load_forecast_cleaning_config(config_dir)
    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = reader(raw_path / filename)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    initial_row_count = len(df)

    df = df.sort_values("Datetime")
    before_dedup = len(df)
    df = df.drop_duplicates(subset="Datetime", keep="first")
    duplicates_dropped = before_dedup - len(df)

    df = df.set_index("Datetime")

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h", name="Datetime")
    reindexed = df.reindex(full_range)

    is_missing = reindexed[target_col].isna()
    gaps = _detect_gaps(is_missing)
    for gap_start, gap_end, length_hours in gaps:
        if length_hours > cleaning_config.max_gap_hours:
            raise ValueError(
                f"Gap of {length_hours}h from {gap_start} to {gap_end} exceeds "
                f"max_gap_hours={cleaning_config.max_gap_hours}. Refusing to "
                f"silently interpolate over a multi-day outage."
            )

    if cleaning_config.fill_strategy == "interpolate":
        reindexed[target_col] = reindexed[target_col].interpolate(method="linear")
    else:
        reindexed[target_col] = reindexed[target_col].ffill()

    interim_path.mkdir(parents=True, exist_ok=True)
    output_path = interim_path / f"{Path(filename).stem}.parquet"
    # Writes directly via to_parquet (default index=True), NOT the shared
    # write_parquet() helper in src/utils/io.py — that helper hardcodes
    # index=False, which would drop the DatetimeIndex every downstream
    # forecasting stage and Plan 1's build_forecast_features_schema require.
    reindexed.to_parquet(output_path)

    gaps_filled_hours = sum(g[2] for g in gaps)
    write_manifest(interim_path, {
        "run_id": run_id,
        "source": filename,
        "stage": "clean_forecast",
        "output_path": str(output_path),
        "row_count_before": initial_row_count,
        "row_count_after": len(reindexed),
        "duplicates_dropped": duplicates_dropped,
        "gaps_found": len(gaps),
        "gaps_filled_hours": gaps_filled_hours,
        "fill_strategy": cleaning_config.fill_strategy,
    })

    logger.info(
        "Cleaned %s: %d rows -> %d rows, %d duplicates dropped, %d gaps filled (%dh)",
        filename, initial_row_count, len(reindexed), duplicates_dropped, len(gaps), gaps_filled_hours,
    )

    return {
        "run_id": run_id,
        "output_path": str(output_path),
        "row_count": len(reindexed),
        "duplicates_dropped": duplicates_dropped,
        "gaps_found": len(gaps),
        "gaps_filled_hours": gaps_filled_hours,
    }
