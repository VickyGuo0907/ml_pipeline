"""Clean stage for finance returns/risk pipelines: parse, sort, dedupe, and
gap-handle a long-format (Date, Ticker, target) monthly return panel, per
asset.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_finance_cleaning_config, load_pipeline_config
from src.utils.io import READERS, load_manifest, resolve_run_path, write_manifest, write_parquet

logger = logging.getLogger(__name__)


def _detect_gaps(is_missing: pd.Series) -> list[tuple[Any, Any, int]]:
    """Group a boolean missing-month mask into contiguous gap runs.

    Args:
        is_missing: Boolean Series indexed by a complete, gap-free monthly
            PeriodIndex (one entry per calendar month, reindexed from that
            asset's own min to max Date), True where that month's value is
            missing. Grouped by integer row POSITION, not calendar
            arithmetic — avoids variable-month-length pitfalls entirely,
            since the index is already a complete monthly range by
            construction; only masked values are missing.

    Returns:
        List of (gap_start, gap_end, length_months) for each contiguous run
        of missing months, in chronological order. gap_start/gap_end are the
        index values (Periods) bounding the run.
    """
    if not is_missing.any():
        return []
    positions = np.flatnonzero(is_missing.to_numpy())
    new_run = np.diff(positions, prepend=positions[0] - 2) != 1
    group_id = np.cumsum(new_run)
    idx = is_missing.index
    gaps = []
    for gid in np.unique(group_id):
        group_positions = positions[group_id == gid]
        start = idx[group_positions[0]]
        end = idx[group_positions[-1]]
        length_months = len(group_positions)
        gaps.append((start, end, length_months))
    return gaps


def _clean_one_asset(
    asset_df: pd.DataFrame,
    ticker: str,
    target_col: str,
    max_gap_months: int,
    fill_strategy: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Dedup, reindex onto a complete monthly range, gap-check, and fill one asset's series.

    Args:
        asset_df: Rows for a single ticker, with a 'Date' column (datetime64) and target_col.
        ticker: This asset's ticker symbol (for the output Ticker column and error messages).
        target_col: Target column name (e.g. 'log_return').
        max_gap_months: Gaps longer than this fail loud.
        fill_strategy: 'interpolate' or 'ffill'.

    Returns:
        (cleaned_df, stats): cleaned_df has columns ['Date', 'Ticker', target_col]
        and a plain RangeIndex; stats has duplicates_dropped/gaps_found/gaps_filled_months.

    Raises:
        ValueError: If any gap exceeds max_gap_months.
    """
    asset_df = asset_df.sort_values("Date")
    before_dedup = len(asset_df)
    # A duplicate (Date, Ticker) row is averaged, not arbitrarily resolved by
    # keeping one — mirrors clean_forecast_data's DST-duplicate reasoning for
    # the hourly PJM pipeline: both readings' information is kept.
    asset_df = asset_df.groupby("Date", as_index=False)[target_col].mean()
    duplicates_dropped = before_dedup - len(asset_df)

    periods = pd.PeriodIndex(asset_df["Date"], freq="M")
    series = pd.Series(asset_df[target_col].to_numpy(), index=periods).sort_index()

    full_range = pd.period_range(series.index.min(), series.index.max(), freq="M")
    reindexed = series.reindex(full_range)

    is_missing = reindexed.isna()
    gaps = _detect_gaps(is_missing)
    for gap_start, gap_end, length_months in gaps:
        if length_months > max_gap_months:
            raise ValueError(
                f"Gap of {length_months}mo from {gap_start} to {gap_end} for "
                f"{ticker} exceeds max_gap_months={max_gap_months}. Refusing "
                f"to silently interpolate over a multi-month outage."
            )

    # "interpolate" is a deliberate smoothing default for short, rare gaps in
    # this POC-scale pipeline, not a blind inheritance from the hourly PJM
    # pipeline's config vocabulary — see this plan's Global Constraints.
    if fill_strategy == "interpolate":
        reindexed = reindexed.interpolate(method="linear")
    else:
        reindexed = reindexed.ffill()

    cleaned = pd.DataFrame({
        "Date": reindexed.index.to_timestamp(),
        "Ticker": ticker,
        target_col: reindexed.to_numpy(),
    })

    gaps_filled_months = sum(g[2] for g in gaps)
    stats = {
        "duplicates_dropped": duplicates_dropped,
        "gaps_found": len(gaps),
        "gaps_filled_months": gaps_filled_months,
    }
    return cleaned, stats


def clean_finance_data(
    raw_dir: str | Path,
    interim_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Parse, sort, dedupe, and gap-handle the raw per-asset monthly return panel.

    Processes each asset (Ticker) independently: a gap or duplicate in one
    asset's series never affects another asset's rows. Each asset is
    reindexed onto its own complete monthly range (its own min-to-max Date)
    — assets with different listing dates naturally produce different-length
    series, which is correct, not a bug.

    Args:
        raw_dir: Base directory containing raw data.
        interim_dir: Output directory for cleaned data.
        run_id: Run identifier to locate/version data.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).

    Returns:
        Dictionary with cleaning statistics.

    Raises:
        FileNotFoundError: If the raw manifest is absent.
        ValueError: If the raw directory doesn't have exactly one file, the
            file's format is unsupported, or any asset's gap exceeds
            max_gap_months.
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    interim_path = resolve_run_path(interim_dir, run_id)
    manifest = load_manifest(raw_path)  # raises FileNotFoundError if absent

    files = list(manifest.get("files", {}))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one raw file for a finance pipeline, found "
            f"{len(files)}: {files}"
        )
    filename = files[0]
    reader = READERS.get(Path(filename).suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file format: {filename}")

    cleaning_config = load_finance_cleaning_config(config_dir)
    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = reader(raw_path / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    initial_row_count = len(df)

    cleaned_parts = []
    duplicates_dropped = 0
    gaps_found = 0
    gaps_filled_months = 0
    per_asset_stats: dict[str, dict[str, int]] = {}
    for ticker, asset_df in df.groupby("Ticker"):
        cleaned_asset, stats = _clean_one_asset(
            asset_df, ticker, target_col, cleaning_config.max_gap_months, cleaning_config.fill_strategy,
        )
        cleaned_parts.append(cleaned_asset)
        duplicates_dropped += stats["duplicates_dropped"]
        gaps_found += stats["gaps_found"]
        gaps_filled_months += stats["gaps_filled_months"]
        per_asset_stats[ticker] = stats

    cleaned = pd.concat(cleaned_parts, ignore_index=True)
    cleaned = cleaned.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    interim_path.mkdir(parents=True, exist_ok=True)
    output_path = interim_path / f"{Path(filename).stem}.parquet"
    write_parquet(cleaned, output_path)

    write_manifest(interim_path, {
        "run_id": run_id,
        "source": filename,
        "stage": "clean_finance",
        "output_path": str(output_path),
        "row_count_before": initial_row_count,
        "row_count_after": len(cleaned),
        "duplicates_dropped": duplicates_dropped,
        "gaps_found": gaps_found,
        "gaps_filled_months": gaps_filled_months,
        "fill_strategy": cleaning_config.fill_strategy,
        "per_asset_stats": per_asset_stats,
    })

    logger.info(
        "Cleaned %s: %d assets, %d rows -> %d rows, %d duplicates dropped, %d gaps filled (%dmo)",
        filename, cleaned["Ticker"].nunique(), initial_row_count, len(cleaned),
        duplicates_dropped, gaps_found, gaps_filled_months,
    )

    return {
        "run_id": run_id,
        "output_path": str(output_path),
        "row_count": len(cleaned),
        "duplicates_dropped": duplicates_dropped,
        "gaps_found": gaps_found,
        "gaps_filled_months": gaps_filled_months,
    }
