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


def _run_drift_metric(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Any | None:
    """Run Evidently's DatasetDriftMetric and return the Report object.

    Args:
        reference_df: Baseline feature matrix to compare against.
        current_df: Current feature matrix.

    Returns:
        The Evidently Report after running, or None if Evidently is unavailable
        or the run itself raised.
    """
    if Report is None or DatasetDriftMetric is None:
        return None
    try:
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=reference_df, current_data=current_df)
        return report
    except Exception:
        return None


def compute_drift_detected(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool | None:
    """Return whether dataset drift was detected between two feature matrices.

    Cheap, boolean-only entry point — does not write an HTML report. Used by
    evaluate.py's regression check to give a regression flag drift context.

    Args:
        reference_df: Baseline feature matrix to compare against.
        current_df: Current feature matrix.

    Returns:
        True/False if the comparison ran successfully, or None if Evidently is
        unavailable or the comparison failed.
    """
    report = _run_drift_metric(reference_df, current_df)
    if report is None:
        return None
    return report.as_dict()["metrics"][0].get("result", {}).get("dataset_drift", None)


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

    reports_path.mkdir(parents=True, exist_ok=True)

    current_df = pd.read_parquet(train_path)

    drift_results: dict[str, Any] = {
        "run_id": run_id,
        "current_shape": current_df.shape,
    }

    if previous_run_id:
        previous_path = Path(features_dir) / previous_run_id / "train.parquet"

        if previous_path.exists():
            previous_df = pd.read_parquet(previous_path)
            report = _run_drift_metric(previous_df, current_df)

            drift_results["comparison_run_id"] = previous_run_id
            drift_results["previous_shape"] = previous_df.shape

            if report is not None:
                report_path = reports_path / f"{run_id}_drift_report.html"
                report.save_html(str(report_path))
                drift_results["report_path"] = str(report_path)
                drift_results["drift_detected"] = report.as_dict()["metrics"][0].get("result", {}).get("dataset_drift", None)
            else:
                drift_results["warning"] = "Evidently AI not available or report generation failed; skipping drift report"
        else:
            drift_results["warning"] = f"Previous run data not found: {previous_path}"
            drift_results["baseline_run_id"] = run_id
    else:
        report = _run_drift_metric(current_df, current_df)
        if report is not None:
            report_path = reports_path / f"{run_id}_baseline_drift_report.html"
            report.save_html(str(report_path))
            drift_results["report_path"] = str(report_path)
        else:
            drift_results["warning"] = "Evidently AI not available or baseline report generation failed"

        drift_results["type"] = "baseline"
        drift_results["note"] = "No previous run available for comparison; using current data as baseline"

    return drift_results
