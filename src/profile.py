"""Data profiling stage using ydata-profiling."""
import base64
import io
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — this module runs inside an Airflow worker, never a GUI context
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import ydata_profiling.model.pandas.describe_categorical_pandas as _ydp_cat
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
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


def generate_mstl_report(
    raw_dir: str | Path,
    run_id: str,
    reports_dir: str | Path = "reports",
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Generate an MSTL (multiple seasonal-trend decomposition) diagnostic
    report for a forecasting pipeline's raw hourly series.

    Independently parses/sorts/dedupes the raw CSV — does NOT share state
    with the clean stage (which runs after profile in the DAG's fixed task
    order: ingest -> validate_raw -> profile -> clean -> ...). This is a
    lightweight, diagnostic-only pass (linear interpolation over any small
    gaps, not the pipeline's authoritative gap-handling policy), exactly
    the same independence profile_raw_files already has from clean's output.

    Decomposes into trend + one seasonal component per configured period +
    residual. Daily (24h) and weekly (168h) periods are always attempted;
    the annual (8766h) period is only included when the series covers at
    least two full annual cycles — otherwise MSTL's decomposition for that
    period would be meaningless (or fail outright for too little data).

    Args:
        raw_dir: Base directory containing raw data.
        run_id: Run identifier to locate data.
        reports_dir: Output directory for the HTML report.
        config_dir: Pipeline config directory.

    Returns:
        Dictionary with report_path and the list of periods actually used.

    Raises:
        FileNotFoundError: If the raw manifest is absent.
        ValueError: If the raw directory doesn't have exactly one file.
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    manifest = load_manifest(raw_path)  # raises FileNotFoundError if absent

    files = list(manifest.get("files", {}))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one raw file for a forecasting pipeline, found "
            f"{len(files)}: {files}"
        )
    filename = files[0]
    reader = READERS.get(Path(filename).suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file format: {filename}")

    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = reader(raw_path / filename)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset="Datetime", keep="first")
    df = df.set_index("Datetime")

    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h", name="Datetime")
    series = df[target_col].reindex(full_range).interpolate(method="linear")

    DAILY, WEEKLY, ANNUAL = 24, 168, 8766
    periods = [DAILY, WEEKLY]
    if len(series) > 2 * ANNUAL:
        periods.append(ANNUAL)

    result = MSTL(series, periods=periods).fit()

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    report_path = reports_path / f"{run_id}_mstl.html"

    n_panels = 2 + len(periods)  # trend + one per seasonal period + residual
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)

    axes[0].plot(result.trend.index, result.trend.values)
    axes[0].set_title("Trend")

    for i, period in enumerate(periods, start=1):
        col = f"seasonal_{period}"
        axes[i].plot(result.seasonal[col].index, result.seasonal[col].values)
        label = {DAILY: "Daily (24h)", WEEKLY: "Weekly (168h)", ANNUAL: "Annual (8766h)"}.get(
            period, f"{period}h"
        )
        axes[i].set_title(f"Seasonal — {label}")

    axes[-1].plot(result.resid.index, result.resid.values)
    axes[-1].set_title("Residual")

    fig.suptitle(f"MSTL Decomposition — {filename} ({run_id})")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")

    html = (
        f"<html><head><title>MSTL Decomposition — {run_id}</title></head>"
        f"<body><h1>MSTL Decomposition — {filename} ({run_id})</h1>"
        f"<p>Periods used: {periods}</p>"
        f'<img src="data:image/png;base64,{img_b64}" alt="MSTL decomposition: trend, seasonal, residual" />'
        f"</body></html>"
    )
    report_path.write_text(html)

    logger.info("MSTL report written: %s (periods=%s)", report_path, periods)
    return {"report_path": str(report_path), "periods_used": periods}


def generate_adf_report(
    raw_dir: str | Path,
    run_id: str,
    reports_dir: str | Path = "reports",
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Generate an ADF (Augmented Dickey-Fuller) stationarity diagnostic
    report for a finance pipeline's raw long-format return panel.

    Runs one ADF test per asset directly on its raw log-return series - no
    dependency on the clean/features stages, which run later in the DAG's
    fixed task order (mirrors generate_mstl_report's independence from
    clean for pjm_load_forecast). Writes both a human-readable HTML report
    and a YAML summary (reports/<pipeline>/<run_id>_adf_report.yaml, in
    exactly the shape src.finance.evaluate_finance._load_adf_summary
    already reads back) — satisfies requirement #1 (confirm return-
    stationarity, discuss the Random Walk benchmark).

    Args:
        raw_dir: Base directory containing raw data.
        run_id: Run identifier to locate data.
        reports_dir: Output directory for the HTML/YAML reports.
        config_dir: Pipeline config directory.

    Returns:
        Dictionary with report_path, yaml_path, and per-ticker ADF results.

    Raises:
        FileNotFoundError: If the raw manifest is absent.
        ValueError: If the raw directory doesn't have exactly one file, or
            the file's format is unsupported.
    """
    raw_path = resolve_run_path(raw_dir, run_id)
    manifest = load_manifest(raw_path)  # raises FileNotFoundError if absent

    files = list(manifest.get("files", {}))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one raw file for a finance pipeline, found "
            f"{len(files)}: {files}"
        )
    filename = files[0]
    reader = READERS.get(Path(filename).suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported file format: {filename}")

    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = reader(raw_path / filename)
    df["Date"] = pd.to_datetime(df["Date"])

    adf_results: dict[str, dict[str, Any]] = {}
    for ticker, asset_df in df.groupby("Ticker"):
        series = asset_df.sort_values("Date")[target_col].dropna()
        if len(series) < 8:
            logger.warning("Skipping ADF test for %s: only %d observations", ticker, len(series))
            continue
        adf_statistic, p_value, _, n_obs, _, _ = adfuller(series, autolag="AIC")
        adf_results[ticker] = {
            "adf_statistic": float(adf_statistic),
            "p_value": float(p_value),
            "is_stationary": bool(p_value < 0.05),
            "n_obs": int(n_obs),
        }

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    yaml_path = reports_path / f"{run_id}_adf_report.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(adf_results, f, default_flow_style=False, sort_keys=False)

    n_stationary = sum(1 for r in adf_results.values() if r["is_stationary"])
    n_total = len(adf_results)
    rw_discussion = (
        f"{n_stationary}/{n_total} assets' monthly log-returns reject the unit-root "
        f"null hypothesis at p&lt;0.05 (i.e. are stationary) - consistent with the "
        f"Efficient Market Hypothesis's implication that returns (unlike price "
        f"levels, which are almost never stationary) should not be predictable "
        f"from their own past values alone, motivating the Random Walk as a "
        f"legitimate baseline rather than a naive strawman."
    )

    rows_html = "".join(
        f"<tr><td>{ticker}</td><td>{r['adf_statistic']:.4f}</td><td>{r['p_value']:.4f}</td>"
        f"<td>{'Stationary' if r['is_stationary'] else 'Non-stationary'}</td><td>{r['n_obs']}</td></tr>"
        for ticker, r in sorted(adf_results.items())
    )
    html = (
        f"<html><head><title>ADF Stationarity Report — {run_id}</title></head>"
        f"<body><h1>ADF Stationarity Report — {filename} ({run_id})</h1>"
        f"<table border='1'><tr><th>Ticker</th><th>ADF statistic</th><th>p-value</th>"
        f"<th>Result (p&lt;0.05)</th><th>N obs</th></tr>{rows_html}</table>"
        f"<h2>Random Walk benchmark discussion</h2><p>{rw_discussion}</p>"
        f"</body></html>"
    )
    report_path = reports_path / f"{run_id}_adf_report.html"
    report_path.write_text(html)

    logger.info("ADF report written: %s (%d/%d assets stationary)", report_path, n_stationary, n_total)
    return {"report_path": str(report_path), "yaml_path": str(yaml_path), "adf_results": adf_results}
