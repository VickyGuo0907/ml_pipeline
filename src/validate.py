"""Data validation stage using Pandera schemas."""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from src.utils.config import PerFileSchemaConfig, ValidationConfig, load_pipeline_config
from src.utils.io import READERS, load_manifest, resolve_run_path

logger = logging.getLogger(__name__)


def _replace_sentinels(df: pd.DataFrame, sentinels: list[str]) -> pd.DataFrame:
    """Replace dataset-specific missing-value sentinels with NaN before pandera validation.

    Pandera's coerce=True cannot convert arbitrary strings to float, causing spurious
    coercion failures on nullable numeric columns. Sentinel strings are declared in
    pipeline.yaml under validation.sentinel_values so no Python changes are needed for new datasetsets.

    Args:
        df: Raw DataFrame that may contain sentinel strings.
        sentinels: Strings to replace with NaN (from CleaningConfig.sentinel_values).

    Returns:
        Copy of df with sentinel strings replaced by NaN.
    """
    if not sentinels:
        return df
    return df.replace({v: None for v in sentinels})


def _build_schema(validation_config: ValidationConfig | PerFileSchemaConfig) -> DataFrameSchema:
    """Build a Pandera DataFrameSchema from pipeline validation config.

    Required columns → must exist and be non-null in every file.
    Numeric bounds → range-checked when present; optional unless also required.

    Args:
        validation_config: Validated ValidationConfig from pipeline.yaml.

    Returns:
        Configured DataFrameSchema with strict=False (extra columns allowed).
    """
    columns: dict[str, Column] = {}

    for col in validation_config.required_columns:
        columns[col] = Column(nullable=False, required=True, coerce=True)

    for col, bounds in validation_config.numeric_bounds.items():
        checks = []
        if bounds.min is not None:
            checks.append(Check.greater_than_or_equal_to(bounds.min))
        if bounds.max is not None:
            checks.append(Check.less_than_or_equal_to(bounds.max))
        is_required = col in validation_config.required_columns
        columns[col] = Column(
            float,
            checks=checks,
            nullable=not is_required,
            required=is_required,
            coerce=True,
        )

    return DataFrameSchema(columns=columns, strict=False, coerce=True)


def _resolve_schema(
    filename: str,
    validation_config: ValidationConfig,
) -> tuple[DataFrameSchema, int]:
    """Return (schema, min_rows) for a file — first matching per-file entry wins, global is fallback.

    Args:
        filename: Name of the file being validated.
        validation_config: Global ValidationConfig from pipeline.yaml.

    Returns:
        Tuple of (pandera DataFrameSchema, min_rows threshold).
    """
    for pf in validation_config.per_file_schemas:
        if pf.file_pattern in filename:
            logger.debug("Per-file schema matched '%s' → pattern '%s'", filename, pf.file_pattern)
            return _build_schema(pf), pf.min_rows
    return _build_schema(validation_config), validation_config.min_rows


def _validate_single_file(
        df: pd.DataFrame,
        filename: str,
        schema: DataFrameSchema,
        min_rows: int,
        sentinels: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate one DataFrame. Returns (success_entry, failure_entry) — one will be None.

    Args:
        df: Loaded DataFrame.
        filename: File name (used in error messages).
        schema: Pandera schema to validate against.
        min_rows: Minimum required row count.
        sentinels: Strings to treat as NaN before schema validation.

    Returns:
        Tuple of (success_info dict or None, failure_info dict or None).
    """
    if len(df) < min_rows:
        return None, {
            "filename": filename,
            "error": f"{len(df)} rows, minimum required is {min_rows}",
        }
    try:
        schema.validate(_replace_sentinels(df, sentinels), lazy=True)
        return {"rows": len(df), "columns": len(df.columns), "status": "passed"}, None
    except SchemaErrors as exc:
        return None, {"filename": filename, "errors": str(exc)}


def validate_raw_files(
        raw_dir: str | Path,
        run_id: str,
        config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Validate raw data files against config-driven Pandera schema.

    Validation rules (required columns, numeric bounds, min row count) are loaded
    from pipeline.yaml so each pipeline can declare its own expectations without
    touching Python code. All files are validated before any error is raised,
    giving a full picture of failures across the entire landing batch.

    Args:
        raw_dir: Base directory containing raw data.
        run_id: Run identifier to locate the versioned data directory.
        config_dir: Pipeline config directory (e.g. config/biomedical_clinical).

    Returns:
        Dictionary with validated_files and failed_files per file.

    Raises:
        FileNotFoundError: If manifest or a listed data file is missing.
        RuntimeError: If one or more files fail validation (all failures reported).
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    manifest = load_manifest(raw_path)

    pipeline_config = load_pipeline_config(config_dir)
    sentinels = pipeline_config.validation.sentinel_values

    results: dict[str, Any] = {"run_id": run_id, "validated_files": {}, "failed_files": []}

    for filename in manifest.get("files", {}).keys():
        reader = READERS.get(Path(filename).suffix.lower())
        if reader is None:
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = reader(file_path)
        schema, min_rows = _resolve_schema(filename, pipeline_config.validation)
        success, failure = _validate_single_file(df, filename, schema, min_rows, sentinels)

        if success:
            results["validated_files"][filename] = success
            logger.info("Validated %s: %d rows, %d columns", filename, success["rows"], success["columns"])
        else:
            results["failed_files"].append(failure)
            logger.error("Validation failed for %s: %s", filename, failure.get("error") or failure.get("errors"))

    if results["failed_files"]:
        failed_names = [f["filename"] for f in results["failed_files"]]
        raise RuntimeError(
            f"Validation failed for {len(failed_names)} file(s): {failed_names}. "
            f"Details: {results['failed_files']}"
        )

    return results
