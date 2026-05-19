"""Data drift monitoring using Evidently AI."""
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
except ImportError:
    Report = None
    DatasetDriftMetric = None


def generate_drift_report(
    features_dir: str | Path,
    run_id: str,
    previous_run_id: str | None = None,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Generate data drift report comparing current to previous training set.

    Uses Evidently AI to detect statistical drift in feature distributions.

    Args:
        features_dir: Directory containing feature matrices
        run_id: Current run identifier
        previous_run_id: Previous run ID for comparison (if available)
        reports_dir: Output directory for drift reports

    Returns:
        Dictionary with drift report information

    Raises:
        FileNotFoundError: If current feature files don't exist
    """
    features_path = Path(features_dir) / run_id
    train_path = features_path / "train.parquet"
    reports_path = Path(reports_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    # Create reports directory
    reports_path.mkdir(parents=True, exist_ok=True)

    # Load current training data
    current_df = pd.read_parquet(train_path)

    drift_results = {
        "run_id": run_id,
        "current_shape": current_df.shape,
    }

    # If previous run available, compare; otherwise create baseline
    if previous_run_id:
        previous_path = Path(features_dir) / previous_run_id / "train.parquet"

        if previous_path.exists():
            previous_df = pd.read_parquet(previous_path)

            # Generate drift report if evidently is available
            if Report is not None and DatasetDriftMetric is not None:
                try:
                    report = Report(metrics=[DatasetDriftMetric()])
                    report.run(reference_data=previous_df, current_data=current_df)

                    # Save report
                    report_path = reports_path / f"{run_id}_drift_report.html"
                    report.save_html(report_path)

                    drift_results["comparison_run_id"] = previous_run_id
                    drift_results["previous_shape"] = previous_df.shape
                    drift_results["report_path"] = str(report_path)
                    drift_results["drift_detected"] = report.as_dict()["metrics"][0].get("result", {}).get("drift_detected", None)
                except Exception as e:
                    drift_results["warning"] = f"Evidently report generation failed: {str(e)}"
                    drift_results["comparison_run_id"] = previous_run_id
                    drift_results["previous_shape"] = previous_df.shape
            else:
                drift_results["warning"] = "Evidently AI not available; skipping drift report"
                drift_results["comparison_run_id"] = previous_run_id
                drift_results["previous_shape"] = previous_df.shape
        else:
            drift_results["warning"] = f"Previous run data not found: {previous_path}"
            drift_results["baseline_run_id"] = run_id
    else:
        # No previous data: create baseline
        if Report is not None and DatasetDriftMetric is not None:
            try:
                report = Report(metrics=[DatasetDriftMetric()])
                report.run(reference_data=current_df, current_data=current_df)

                report_path = reports_path / f"{run_id}_baseline_drift_report.html"
                report.save_html(report_path)

                drift_results["report_path"] = str(report_path)
            except Exception as e:
                drift_results["warning"] = f"Evidently baseline report failed: {str(e)}"
        else:
            drift_results["warning"] = "Evidently AI not available; skipping baseline report"

        drift_results["type"] = "baseline"
        drift_results["note"] = "No previous run available for comparison; using current data as baseline"

    return drift_results
