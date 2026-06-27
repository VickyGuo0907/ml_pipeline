"""Data cleaning stage: type coercion, imputation, deduplication."""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.config import load_cleaning_config
from src.utils.transforms import IMPUTE_REGISTRY, drop_pattern_columns

logger = logging.getLogger(__name__)


def _apply_type_coercion(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel strings and coerce object columns to numeric where possible.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with numeric coercion applied.
    """
    df = df.replace({"Too Few to Report": None, "Not Available": None})
    for col in df.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Only replace if most non-null values successfully converted
        if coerced.notna().sum() > df[col].notna().sum() * 0.5:
            df[col] = coerced
    return df


def _drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where missing fraction exceeds threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum allowed fraction of missing values (0–1).

    Returns:
        DataFrame with sparse columns removed.
    """
    missing_ratio = df.isnull().sum() / len(df)
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    if to_drop:
        logger.info("Dropping %d high-missing columns (>%.0f%%): %s", len(to_drop), threshold * 100, to_drop)
    return df.drop(columns=to_drop)


def _clean_single_file(
    file_path: Path,
    cleaning_config: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a single CSV file per cleaning configuration.

    Returns:
        Tuple of (cleaned DataFrame, stats dict).
    """
    df = pd.read_csv(file_path)
    initial_shape = df.shape

    df = _apply_type_coercion(df)
    df = _drop_high_missing(df, threshold=0.5)

    # Drop columns matching bad-imputation patterns from config
    df, pattern_dropped = drop_pattern_columns(df, cleaning_config.drop_column_patterns)

    # Impute missing values using the configured strategy
    impute_fn = IMPUTE_REGISTRY.get(cleaning_config.impute_strategy, IMPUTE_REGISTRY["median"])
    df = impute_fn(df)

    # Remove exact duplicate rows
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
    """Clean raw CSV files: type coercion, imputation, deduplication.

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
    raw_path = Path(raw_dir) / run_id
    interim_path = Path(interim_dir) / run_id
    manifest_path = raw_path / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    cleaning_config = load_cleaning_config(config_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    results: dict[str, Any] = {"run_id": run_id, "files": {}}

    for filename in manifest.get("files", {}):
        if not filename.endswith(".csv"):
            continue
        file_path = raw_path / filename
        if not file_path.exists():
            continue

        df, stats = _clean_single_file(file_path, cleaning_config)
        output_path = interim_path / filename
        df.to_csv(output_path, index=False)
        results["files"][filename] = {**stats, "output_path": str(output_path)}
        logger.info("Cleaned %s: %s → %s", filename, stats["initial_shape"], stats["final_shape"])

    interim_manifest = {
        "run_id": run_id,
        "source": "cleaned raw data",
        "stage": "clean",
        "files": {k: v.get("output_path", "") for k, v in results["files"].items()},
    }
    with open(interim_path / "manifest.yaml", "w") as f:
        yaml.dump(interim_manifest, f, default_flow_style=False, sort_keys=False)

    return results
