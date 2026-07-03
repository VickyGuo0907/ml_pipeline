"""Data profiling stage using ydata-profiling."""
import logging
from pathlib import Path
from typing import Any

import ydata_profiling.model.pandas.describe_categorical_pandas as _ydp_cat
from ydata_profiling import ProfileReport

from src.utils.config import load_pipeline_config
from src.utils.io import READERS, load_manifest, resolve_run_path

logger = logging.getLogger(__name__)

# scipy 1.14+ returns a Python float instead of a numpy scalar in edge cases,
# causing ydata-profiling's chi_square helper to crash on `.ndim`. The function
# is imported by name into describe_categorical_pandas, so the patch must target
# that module's namespace directly (not the originating summary_algorithms module).
_orig_chi_square = _ydp_cat.chi_square


def _safe_chi_square(histogram: Any) -> dict[str, Any]:
    """chi_square wrapper that tolerates scipy/numpy scalar type mismatches."""
    try:
        return _orig_chi_square(histogram)
    except AttributeError:
        return {"statistic": None, "pvalue": None}


_ydp_cat.chi_square = _safe_chi_square


def profile_raw_files(
        raw_dir: str | Path,
        run_id: str,
        reports_dir: str | Path = "reports",
        config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Generate profiling reports for raw data files.

    Profiles the raw data as-is (before sentinel replacement or cleaning) so the
    report reflects the true shape of the incoming data, including missing-value
    sentinel strings. One HTML report is written per file.

    Currently supports: CSV, Parquet.

    Args:
        raw_dir: Base directory containing raw data.
        run_id: Run identifier to locate data.
        reports_dir: Output directory for HTML reports.
        config_dir: Pipeline config directory (reserved for future use).

    Returns:
        Dictionary with report paths, row counts, and column counts per file.

    Raises:
        FileNotFoundError: If manifest doesn't exist.
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    manifest = load_manifest(raw_path)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    pipeline_config = load_pipeline_config(config_dir)
    minimal = pipeline_config.profiling.minimal

    profiling_results: dict[str, Any] = {"run_id": run_id, "reports": {}, "minimal": minimal}

    for filename in manifest.get("files", {}).keys():
        suffix = Path(filename).suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            logger.warning("File listed in manifest not found, skipping: %s", file_path)
            continue

        df = reader(file_path)
        logger.info(
            "Profiling %s: %d rows × %d cols (minimal=%s)", filename, len(df), len(df.columns), minimal
        )

        stem = Path(filename).stem
        report_name = f"{run_id}_{stem}_profile.html"
        report_path = reports_path / report_name

        try:
            profile = ProfileReport(df, title=f"Data Profile: {filename}", minimal=minimal)
            profile.to_file(report_path)
        except Exception as exc:
            logger.warning(
                "Profile failed for %s (%s: %s) — falling back to minimal",
                filename, type(exc).__name__, exc,
            )
            profile = ProfileReport(df, title=f"Data Profile: {filename}", minimal=True)
            profile.to_file(report_path)

        profiling_results["reports"][filename] = {
            "report_path": str(report_path),
            "rows": len(df),
            "columns": len(df.columns),
        }
        logger.info("Report written: %s", report_path)

    return profiling_results
