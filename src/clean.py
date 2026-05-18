"""Data cleaning stage: type coercion, missing handling, deduplication."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def clean_raw_data(
    raw_dir: str | Path,
    interim_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Clean raw CSV files: type coercion, missing values, deduplication.

    Args:
        raw_dir: Base directory containing raw data
        interim_dir: Output directory for cleaned data
        run_id: Run identifier to locate/version data

    Returns:
        Dictionary with cleaning statistics

    Raises:
        FileNotFoundError: If manifest doesn't exist
    """
    raw_path = Path(raw_dir) / run_id
    interim_path = Path(interim_dir) / run_id
    manifest_path = raw_path / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Create interim directory
    interim_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    cleaning_results = {
        "run_id": run_id,
        "files": {},
    }

    # Clean each CSV file
    for filename in manifest.get("files", {}).keys():
        if not filename.endswith(".csv"):
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            continue

        # Load data
        df = pd.read_csv(file_path)
        initial_shape = df.shape

        # Replace "Too Few to Report" with NaN before numeric conversion
        df = df.replace({"Too Few to Report": None})

        # Type coercion: convert numeric columns
        numeric_cols = df.select_dtypes(include=["object"]).columns
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except (ValueError, TypeError):
                pass  # Keep as-is if coercion fails

        # Handle missing values: drop columns with >50% missing
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # Remove duplicates
        df = df.drop_duplicates()

        final_shape = df.shape

        # Save cleaned data
        output_path = interim_path / filename
        df.to_csv(output_path, index=False)

        cleaning_results["files"][filename] = {
            "initial_shape": initial_shape,
            "final_shape": final_shape,
            "rows_removed": initial_shape[0] - final_shape[0],
            "cols_removed": initial_shape[1] - final_shape[1],
            "output_path": str(output_path),
        }

    # Write interim manifest
    interim_manifest = {
        "run_id": run_id,
        "source": "cleaned raw data",
        "stage": "clean",
        "files": {k: v.get("output_path", "") for k, v in cleaning_results["files"].items()},
    }
    interim_manifest_path = interim_path / "manifest.yaml"
    with open(interim_manifest_path, "w") as f:
        yaml.dump(interim_manifest, f, default_flow_style=False, sort_keys=False)

    return cleaning_results
