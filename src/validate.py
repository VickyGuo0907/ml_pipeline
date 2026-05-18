"""Data validation stage using Pandera schemas."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandera.errors import SchemaError

from src.schemas.raw import raw_schema


def validate_raw_files(
    raw_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Validate raw CSV files against schema.

    Args:
        raw_dir: Base directory containing raw data
        run_id: Run identifier to locate data

    Returns:
        Dictionary with validation results and file counts

    Raises:
        FileNotFoundError: If raw directory or manifest doesn't exist
        ValidationError: If any file fails schema validation
    """
    raw_path = Path(raw_dir) / run_id
    manifest_path = raw_path / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load manifest
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Validate each CSV file
    validation_results = {
        "run_id": run_id,
        "validated_files": {},
        "failed_files": [],
    }

    for filename in manifest.get("files", {}).keys():
        if not filename.endswith(".csv"):
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load and validate
        df = pd.read_csv(file_path)
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
