"""Data validation stage using Pandera schemas."""
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml
from pandera.errors import SchemaError

from src.schemas.raw import raw_schema


def _read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def _read_parquet(file_path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(file_path)


# Maps file extension → reader. Register new formats here without touching validate_raw_files.
READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": _read_csv,
    ".parquet": _read_parquet,
}


def validate_raw_files(
    raw_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Validate raw data files against schema.

    Dispatches each file to a format-specific reader based on extension,
    then validates the resulting DataFrame with the raw pandera schema.
    Currently supports: CSV, Parquet.

    Args:
        raw_dir: Base directory containing raw data
        run_id: Run identifier to locate data

    Returns:
        Dictionary with validation results and file counts

    Raises:
        FileNotFoundError: If raw directory or manifest doesn't exist
        SchemaError: If any file fails schema validation
    """
    raw_path = Path(raw_dir) / run_id
    manifest_path = raw_path / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    validation_results: dict[str, Any] = {
        "run_id": run_id,
        "validated_files": {},
        "failed_files": [],
    }

    for filename in manifest.get("files", {}).keys():
        suffix = Path(filename).suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = reader(file_path)
        try:
            raw_schema.validate(df, lazy=False)
            validation_results["validated_files"][filename] = {
                "rows": len(df),
                "columns": len(df.columns),
                "status": "passed",
            }
        except SchemaError as e:
            validation_results["failed_files"].append(
                {"filename": filename, "error": str(e)}
            )
            raise

    return validation_results
