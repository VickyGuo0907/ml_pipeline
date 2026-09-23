"""Evaluation/registration stage for finance returns/risk pipelines:
reloads each trained (model_type, asset) model from MLflow, scores it
against the held-out test period, assembles a (month, ticker, predicted,
actual) table per model type, and computes cross-sectional rank correlation
(Information Coefficient) averaged across test months - directly
operationalizing the brief's rank-driven long/short industry insight.
Registers every (model_type, asset) pair to MLflow Staging (never
auto-promoted to Production) and tags run_champion: true on every run
belonging to the highest-IC model type.

Stationarity/ADF analysis (requirement #1) is generated separately by the
pipeline's profile stage (src/profile.py's generate_adf_report, a later
plan, mirroring the MSTL precedent already used for pjm_load_forecast) -
not duplicated here. This stage's report notes that fact instead of
fabricating a stationarity section from data it doesn't have.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.evaluate import _write_evaluation_report
from src.finance.rank_ic import reload_model, spearman_ic
from src.utils.config import load_finance_models_config, load_pipeline_config
from src.utils.io import load_manifest, resolve_run_path


def _load_adf_summary(reports_dir: str | Path, run_id: str) -> dict[str, Any] | None:
    """Pull through the ADF stationarity summary if the profile stage's
    generate_adf_report has already produced one for this run (src/profile.py).
    Returns None if that report doesn't exist yet for this run_id - this
    function never fails or fabricates ADF numbers for a report file that
    isn't there.

    Args:
        reports_dir: Pipeline reports directory.
        run_id: Run identifier.

    Returns:
        Parsed ADF report dict (ticker -> {adf_statistic, p_value,
        is_stationary, n_obs}), or None if not yet available.
    """
    adf_path = Path(reports_dir) / f"{run_id}_adf_report.yaml"
    if not adf_path.exists():
        return None
    with open(adf_path) as f:
        return yaml.safe_load(f)

logger = logging.getLogger(__name__)

PER_ASSET_MODEL_TYPES = {"random_walk", "mean", "arima"}


def register_finance_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
    config_dir: str | Path = "config",
    run_id: str = "unknown",
    reports_dir: str | Path = "reports",
    features_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score each trained finance model on the held-out test set via cross-
    sectional rank-IC, register passing models to MLflow Staging, and tag
    the run champion (the highest-IC model TYPE, every one of its per-asset
    runs tagged).

    Args:
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_run_ids: Per-(model_type, asset) MLflow run IDs from the train stage.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).
        run_id: Run identifier.
        reports_dir: Directory for the evaluation audit report.
        features_dir: Directory containing the test parquet + features
            manifest. Required — this stage's core job is scoring against
            the real test set.
        benchmark_dir: Unused — this pipeline has no champion/challenger
            design (config/m6_returns_risk/pipeline.yaml has
            benchmark.enabled: false). Accepted only to match the locked
            signature dag_factory.py calls.

    Returns:
        Dictionary with per-(model_type, asset) registration results.

    Raises:
        ValueError: If mlflow_run_ids or features_dir is missing, or if no
            model could be registered at all. A single (model_type, asset)
            registration failing (e.g. a stale/invalid MLflow run ID) is
            recorded per-model in the returned dict with status "error"
            and does not block the other, otherwise-good per-asset
            registrations from completing.
    """
    if not mlflow_run_ids:
        raise ValueError("mlflow_run_ids required for model registration")
    if features_dir is None:
        raise ValueError("features_dir required to load the test set for rank-IC scoring")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    pipeline_cfg = load_pipeline_config(config_dir)
    models_cfg = load_finance_models_config(config_dir)
    target_col = pipeline_cfg.target.name

    features_path = resolve_run_path(features_dir, run_id)
    test_df = pd.read_parquet(features_path / "test.parquet")
    features_manifest = load_manifest(features_path)
    ticker_mapping: dict[str, int] = features_manifest["ticker_mapping"]
    code_to_ticker = {code: ticker for ticker, code in ticker_mapping.items()}
    feature_columns: list[str] = features_manifest["feature_columns"]

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
    timestamp = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {"run_id": run_id, "evaluated_at": timestamp, "models": {}}
    registration_results: dict[str, Any] = {"registered_models": {}}
    infra_failures: list[str] = []
    rows_by_type: dict[str, list[dict[str, Any]]] = {}
    error_std_by_type: dict[str, list[float]] = {}
    runs_by_type: dict[str, list[tuple[str, str]]] = {}

    for model_name, mlflow_run_id in mlflow_run_ids.items():
        try:
            run = mlflow.get_run(mlflow_run_id)
            run_tags = run.data.tags
            model_type = run_tags.get("model_type", "unknown")
            pipeline_type = run_tags.get("pipeline_type", "unknown")

            model_uri = f"runs:/{mlflow_run_id}/model"
            loaded = reload_model(model_type, mlflow_run_id, mlflow_tracking_uri)

            if model_type in PER_ASSET_MODEL_TYPES:
                ticker = run_tags["ticker"]
                asset_test = test_df[test_df["ticker_encoded"] == ticker_mapping[ticker]].sort_values("date_ordinal")
                if asset_test.empty:
                    raise ValueError(f"No test rows for {ticker}")
                steps = len(asset_test)
                if model_type == "arima":
                    predictions = np.asarray(loaded.forecast(steps=steps))
                else:
                    predictions = np.asarray(loaded.predict(pd.DataFrame({"_dummy": range(steps)})))
                actual = asset_test[target_col].to_numpy()
                months = asset_test["date_ordinal"].to_numpy()
                tickers_for_rows = [ticker] * steps
            elif model_type == "cross_sectional_gbm":
                predictions = np.asarray(loaded.predict(test_df[feature_columns]))
                actual = test_df[target_col].to_numpy()
                months = test_df["date_ordinal"].to_numpy()
                tickers_for_rows = [code_to_ticker[c] for c in test_df["ticker_encoded"]]
            else:
                raise ValueError(f"Unknown finance model type '{model_type}' for {model_name}")

            test_error_std = float(pd.Series(actual - predictions).std())

            for month, ticker_val, pred, act in zip(months, tickers_for_rows, predictions, actual):
                rows_by_type.setdefault(model_type, []).append({
                    "month": month, "ticker": ticker_val, "predicted": pred, "actual": act,
                })
            error_std_by_type.setdefault(model_type, []).append(test_error_std)

            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            version = registered_model.version

            version_tags = {
                "deployment": "staging",
                "registered_by": "pipeline",
                "registered_at": timestamp,
                "environment": "development",
                "pipeline_type": pipeline_type,
                "pipeline_run_id": run_tags.get("run_id", run_id),
                "model_type": model_type,
                "source_run_id": mlflow_run_id,
                "test_error_std": f"{test_error_std:.6f}",
            }
            if "ticker" in run_tags:
                version_tags["ticker"] = run_tags["ticker"]
            for key, value in version_tags.items():
                client.set_model_version_tag(model_name, version, key, value)

            client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
            try:
                client.set_registered_model_alias(model_name, "staging", version)
            except Exception as alias_err:
                logger.warning("Could not set alias for %s v%s: %s", model_name, version, alias_err)

            runs_by_type.setdefault(model_type, []).append((model_name, version))

            report["models"][model_name] = {
                "status": "registered", "version": version, "model_type": model_type,
                "test_error_std": test_error_std,
            }
            registration_results["registered_models"][model_name] = {
                "status": "registered", "version": version, "stage": "Staging",
                "source_run_id": mlflow_run_id, "model_uri": model_uri,
                "test_error_std": test_error_std, "registered_at": timestamp,
            }
            logger.info(
                "REGISTERED %s v%s to Staging (model_type=%s, test_error_std=%.6f)",
                model_name, version, model_type, test_error_std,
            )
        except Exception as e:
            logger.error("Error evaluating/registering %s: %s", model_name, e)
            report["models"][model_name] = {"status": "error", "error": str(e)}
            registration_results["registered_models"][model_name] = {
                "status": "error", "error": str(e), "source_run_id": mlflow_run_id,
            }
            infra_failures.append(model_name)

    ic_by_type: dict[str, float] = {}
    avg_error_std_by_type: dict[str, float] = {}
    for model_type, rows in rows_by_type.items():
        ic_by_type[model_type] = spearman_ic(pd.DataFrame(rows))
        avg_error_std_by_type[model_type] = float(pd.Series(error_std_by_type[model_type]).mean())

    report["ic_leaderboard"] = ic_by_type
    report["error_std_leaderboard"] = avg_error_std_by_type
    report["error_std_note"] = (
        "error_std_leaderboard measures forecast-error dispersion (Plan 3's "
        "risk-analysis figure), not model quality - it is shift-invariant, so "
        "models that predict a constant (e.g. random_walk and mean with "
        "window=null) can produce numerically identical values despite being "
        "different models. Use ic_leaderboard, not this field, to compare "
        "model quality."
    )
    report["stationarity_note"] = (
        "ADF stationarity analysis is generated separately by the pipeline's "
        "profile stage (src/profile.py's generate_adf_report), not duplicated here."
    )
    report["adf_summary"] = _load_adf_summary(reports_dir, run_id)
    report["scoring_methodology_note"] = (
        "random_walk/mean/arima are scored with a STATIC multi-step forecast "
        "(fitted once, forecasting the entire test horizon from the train "
        "boundary); cross_sectional_gbm is scored with FRESH realized features "
        "at every test month (an effectively 1-step-ahead information set). "
        "The two are not a strictly apples-to-apples comparison - the IC "
        "leaderboard should be read with this in mind, not as a fully fair "
        "head-to-head. Walk-forward re-scoring of the per-asset models is a "
        "candidate follow-up, not implemented here."
    )

    registered = [k for k, v in report["models"].items() if v["status"] == "registered"]
    if registered:
        valid_types = [t for t in ic_by_type if not pd.isna(ic_by_type[t])]
        if valid_types:
            champion_type = max(valid_types, key=lambda t: ic_by_type[t])
            report["run_champion_type"] = champion_type
            report["champion_metric"] = models_cfg.evaluation.champion_metric
            for model_name, version in runs_by_type.get(champion_type, []):
                try:
                    client.set_model_version_tag(model_name, version, "run_champion", "true")
                except Exception as e:
                    logger.warning("Could not tag run champion %s v%s: %s", model_name, version, e)
        else:
            report["run_champion_type"] = None

        mean_ic = ic_by_type.get("mean")
        arima_ic = ic_by_type.get("arima")
        if mean_ic is not None and arima_ic is not None and not pd.isna(mean_ic) and not pd.isna(arima_ic):
            if mean_ic >= arima_ic:
                report["shrinkage_discussion"] = (
                    f"Mean (IC={mean_ic:.4f}) matched or outperformed ARIMA (IC={arima_ic:.4f}) "
                    f"in this run - consistent with the shrinkage principle: on noisy financial "
                    f"returns, a simple historical average often generalizes better than a fitted "
                    f"autoregressive model, whose extra parameters mostly capture noise."
                )
            else:
                report["shrinkage_discussion"] = (
                    f"ARIMA (IC={arima_ic:.4f}) outperformed Mean (IC={mean_ic:.4f}) in this run - "
                    f"the shrinkage principle is a general tendency, not a guarantee for every "
                    f"sample; a larger test period would clarify whether this holds up."
                )
        else:
            report["shrinkage_discussion"] = "Insufficient data to compare Mean and ARIMA's rank-IC in this run."
    else:
        report["run_champion_type"] = None

    _write_evaluation_report(report, run_id, reports_dir)

    logger.info(
        "Finance evaluation complete: %d registered, %d errors", len(registered), len(infra_failures),
    )

    # A single (model_type, asset) registration failing (e.g. a stale run
    # ID) is recorded per-model above and must not block the other,
    # otherwise-good per-asset registrations - only total failure is fatal.
    if not registered:
        raise ValueError(f"No finance models could be scored/registered for run {run_id}.")

    return registration_results
