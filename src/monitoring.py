"""Data drift monitoring using Evidently AI."""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import resolve_run_path

try:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
except ImportError:
    Report = None
    DatasetDriftMetric = None

logger = logging.getLogger(__name__)


def _align_columns(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to their common columns before drift comparison.

    Two runs' feature matrices can legitimately have different columns — e.g. the
    near-zero-variance filter in feature engineering can keep or drop a column
    differently depending on that run's population, or a pipeline's predictor set
    can change between runs. Evidently assumes reference and current share the
    same columns and raises a KeyError deep inside its drift calculation if they
    don't; comparing only what both runs actually have is the correct behavior,
    not a workaround — a column absent from one side can't have its drift
    assessed against the other anyway.

    Args:
        reference_df: Baseline feature matrix to compare against.
        current_df: Current feature matrix.

    Returns:
        (reference_df, current_df) restricted to their common columns, in the
        same relative order as reference_df.
    """
    common = [c for c in reference_df.columns if c in current_df.columns]
    reference_only = set(reference_df.columns) - set(common)
    current_only = set(current_df.columns) - set(common)
    if reference_only or current_only:
        logger.warning(
            "Drift comparison: %d common column(s); reference-only=%s, current-only=%s",
            len(common), sorted(reference_only), sorted(current_only),
        )
    return reference_df[common], current_df[common]


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
    reference_df, current_df = _align_columns(reference_df, current_df)
    try:
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=reference_df, current_data=current_df)
        return report
    except Exception:
        logger.exception("Evidently drift report failed")
        return None


def _extract_dataset_drift(report: Any) -> bool | None:
    """Pull the dataset_drift boolean out of an Evidently Report's dict form.

    No try/except here — both call sites already wrap this in their own,
    with different fallback/logging behavior on failure, so the exception
    is left to propagate to whichever caller invoked this.

    Args:
        report: An Evidently Report after .run() has completed.

    Returns:
        The dataset_drift boolean, or None if the report doesn't carry one.
    """
    return report.as_dict()["metrics"][0].get("result", {}).get("dataset_drift", None)


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
    try:
        return _extract_dataset_drift(report)
    except Exception:
        logger.exception("Could not read dataset_drift result from Evidently report")
        return None


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
    features_path = resolve_run_path(features_dir, run_id)
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
        previous_path = resolve_run_path(features_dir, previous_run_id) / "train.parquet"

        if previous_path.exists():
            previous_df = pd.read_parquet(previous_path)
            report = _run_drift_metric(previous_df, current_df)

            drift_results["comparison_run_id"] = previous_run_id
            drift_results["previous_shape"] = previous_df.shape

            if report is not None:
                report_path = reports_path / f"{run_id}_drift_report.html"
                try:
                    report.save_html(str(report_path))
                    drift_results["report_path"] = str(report_path)
                    drift_results["drift_detected"] = _extract_dataset_drift(report)
                except Exception as e:
                    logger.exception("Evidently report rendering failed")
                    drift_results["warning"] = f"Evidently report rendering failed: {e}"
            else:
                drift_results["warning"] = "Evidently AI not available or report generation failed; skipping drift report"
        else:
            drift_results["warning"] = f"Previous run data not found: {previous_path}"
            drift_results["baseline_run_id"] = run_id
    else:
        report = _run_drift_metric(current_df, current_df)
        if report is not None:
            report_path = reports_path / f"{run_id}_baseline_drift_report.html"
            try:
                report.save_html(str(report_path))
                drift_results["report_path"] = str(report_path)
            except Exception as e:
                logger.exception("Evidently baseline report rendering failed")
                drift_results["warning"] = f"Evidently baseline report rendering failed: {e}"
        else:
            drift_results["warning"] = "Evidently AI not available or baseline report generation failed"

        drift_results["type"] = "baseline"
        drift_results["note"] = "No previous run available for comparison; using current data as baseline"

    return drift_results
