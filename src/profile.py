"""Data profiling stage using ydata-profiling."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ydata_profiling import ProfileReport


def profile_raw_files(
    raw_dir: str | Path,
    run_id: str,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Generate profiling reports for raw CSV files.

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

    # Create reports directory
    reports_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    profiling_results = {
        "run_id": run_id,
        "reports": {},
    }

    # Profile each CSV file
    for filename in manifest.get("files", {}).keys():
        if not filename.endswith(".csv"):
            continue

        file_path = raw_path / filename
        if not file_path.exists():
            continue

        # Load data
        df = pd.read_csv(file_path)

        # Generate profile report
        profile = ProfileReport(
            df,
            title=f"Data Profile: {filename}",
            minimal=True,  # Use minimal mode for faster generation
        )

        # Save report
        report_name = f"{run_id}_{filename.replace('.csv', '_profile.html')}"
        report_path = reports_path / report_name
        profile.to_file(report_path)

        profiling_results["reports"][filename] = {
            "report_path": str(report_path),
            "rows": len(df),
            "columns": len(df.columns),
        }

    return profiling_results
