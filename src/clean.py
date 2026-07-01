"""Data cleaning stage: type coercion, imputation, deduplication."""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_cleaning_config, load_pipeline_config
from src.utils.io import READERS, WRITERS, load_manifest, resolve_run_path, write_manifest
from src.utils.transforms import IMPUTE_REGISTRY, drop_pattern_columns

logger = logging.getLogger(__name__)


def _apply_type_coercion(df: pd.DataFrame, sentinels: list[str]) -> pd.DataFrame:
    """Replace sentinel strings and coerce object columns to numeric where possible.

    Sentinel strings (e.g. "Not Available") are declared in pipeline.yaml under
    validation.sentinel_values so no Python changes are needed for new datasets.

    Args:
        df: Raw DataFrame.
        sentinels: Strings to replace with NaN before numeric coercion.

    Returns:
        DataFrame with sentinel replacement and numeric coercion applied.
    """
    if sentinels:
        df = df.replace({v: None for v in sentinels})
    for col in df.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > df[col].notna().sum() * 0.5:
            df[col] = coerced
    return df


def _drop_high_missing(
    df: pd.DataFrame,
    threshold: float = 0.5,
    protect: list[str] | None = None,
) -> pd.DataFrame:
    """Drop columns where missing fraction exceeds threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowed fraction of missing values (0–1).
        protect: Column names exempt from the drop (e.g. sparse pivot-join value columns).

    Returns:
        DataFrame with sparse columns removed.
    """
    protect = protect or []
    missing_ratio = df.isnull().sum() / len(df)
    to_drop = [c for c in missing_ratio[missing_ratio > threshold].index if c not in protect]
    if to_drop:
        logger.info("Dropping %d high-missing columns (>%.0f%%): %s", len(to_drop), threshold * 100, to_drop)
    return df.drop(columns=to_drop)


def _clean_single_file(
    df: pd.DataFrame,
    cleaning_config: Any,
    sentinels: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply cleaning transformations to a DataFrame.

    Args:
        df: Raw DataFrame loaded by the caller.
        cleaning_config: Validated CleaningConfig.
        sentinels: Strings to replace with NaN (from pipeline.yaml validation.sentinel_values).

    Returns:
        Tuple of (cleaned DataFrame, stats dict).
    """
    initial_shape = df.shape

    df = _apply_type_coercion(df, sentinels)
    df = _drop_high_missing(df, threshold=0.5, protect=cleaning_config.protect_columns)
    df, pattern_dropped = drop_pattern_columns(df, cleaning_config.drop_column_patterns)

    impute_fn = IMPUTE_REGISTRY.get(cleaning_config.impute_strategy, IMPUTE_REGISTRY["median"])
    df = impute_fn(df)

    df = df.drop_duplicates()

    return df, {
        "initial_shape": initial_shape,
        "final_shape": df.shape,
        "rows_removed": initial_shape[0] - df.shape[0],
        "cols_removed": initial_shape[1] - df.shape[1],
        "pattern_dropped": pattern_dropped,
    }


def clean_raw_data(
    raw_dir: str | Path,
    interim_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Clean raw data files: type coercion, imputation, deduplication.

    Dispatches each file to a format-specific reader/writer based on extension.
    Output format matches input format (CSV in → CSV out, Parquet in → Parquet out).
    Currently supports: CSV, Parquet.

    Args:
        raw_dir: Base directory containing raw data.
        interim_dir: Output directory for cleaned data.
        run_id: Run identifier to locate/version data.
        config_dir: Pipeline config directory (e.g. config/biomedical_clinical).

    Returns:
        Dictionary with cleaning statistics per file.

    Raises:
        FileNotFoundError: If manifest doesn't exist.
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    interim_path = resolve_run_path(interim_dir, run_id)
    manifest = load_manifest(raw_path)

    cleaning_config = load_cleaning_config(config_dir)
    pipeline_config = load_pipeline_config(config_dir)
    sentinels = pipeline_config.validation.sentinel_values
    interim_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"run_id": run_id, "files": {}}

    for filename in manifest.get("files", {}):
        suffix = Path(filename).suffix.lower()
        reader = READERS.get(suffix)
        writer = WRITERS.get(suffix)
        if reader is None or writer is None:
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            continue

        df = reader(file_path)
        df, stats = _clean_single_file(df, cleaning_config, sentinels)

        output_path = interim_path / filename
        writer(df, output_path)

        results["files"][filename] = {**stats, "output_path": str(output_path)}
        logger.info("Cleaned %s: %s → %s", filename, stats["initial_shape"], stats["final_shape"])

    write_manifest(interim_path, {
        "run_id": run_id,
        "source": "cleaned raw data",
        "stage": "clean",
        "files": {k: v.get("output_path", "") for k, v in results["files"].items()},
    })

    return results
