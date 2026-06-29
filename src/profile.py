"""Data profiling stage using ydata-profiling."""
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml
from ydata_profiling import ProfileReport


def _read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def _read_parquet(file_path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(file_path)


# Maps file extension → reader. Register new formats here without touching profile_raw_files.
READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": _read_csv,
    ".parquet": _read_parquet,
}


def profile_raw_files(
    raw_dir: str | Path,
    run_id: str,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Generate profiling reports for raw data files.

    Dispatches each file to a format-specific reader based on extension,
    then generates a ydata-profiling HTML report per file.
    Currently supports: CSV, Parquet.

    Args:
        raw_dir: Base directory containing raw data
        run_id: Run identifier to locate data
        reports_dir: Output directory for HTML reports

    Returns:
        Dictionary with report paths and file counts

    Raises:
        FileNotFoundError: If manifest doesn't exist
    """
    raw_path = Path(raw_dir) / run_id
    manifest_path = raw_path / "manifest.yaml"
    reports_path = Path(reports_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    reports_path.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    profiling_results: dict[str, Any] = {
        "run_id": run_id,
        "reports": {},
    }

    for filename in manifest.get("files", {}).keys():
        suffix = Path(filename).suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            continue

        df = reader(file_path)

        profile = ProfileReport(
            df,
            title=f"Data Profile: {filename}",
            minimal=True,
        )

        stem = Path(filename).stem
        report_name = f"{run_id}_{stem}_profile.html"
        report_path = reports_path / report_name
        profile.to_file(report_path)

        profiling_results["reports"][filename] = {
            "report_path": str(report_path),
            "rows": len(df),
            "columns": len(df.columns),
        }

    return profiling_results
