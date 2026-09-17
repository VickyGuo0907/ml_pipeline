"""Evaluation/registration stage for time-series forecasting pipelines:
rolling-origin test-set scoring, champion selection, and MLflow Staging
registration.

Plan 3's cv_mape_mean is a training-set-only estimate (rolling-origin CV
within train.parquet). This stage adds the genuine held-out generalization
estimate: each already-trained model is loaded back from MLflow by its
logged flavor and scored via the same rolling-origin mechanism, but at
origins drawn from the TEST period — statsmodels models score with
dynamic=False (see src/forecasting/rolling_origin.py's docstring for why
out-of-sample origins need this), GBM scores via the same recursive_forecast
path as training, just against test-period origins with a longer combined
train+test history to draw lag features from.

NO auto-promotion to Production — manual UI click only, mirrors the
tabular src/evaluate.py's registration pattern exactly.

CAVEAT on test_mape_mean for ets/sarimax: this stage scores statsmodels
models with dynamic=False (out-of-sample), which is NOT horizon-anchored the
way GBM's test_mape_mean is — see src/forecasting/rolling_origin.py's module
docstring for why. In short, an ets/sarimax test_mape_mean reflects a forecast
lead time that grows with how far each origin sits past the end of training,
while GBM's reflects a genuine fresh horizon_hours-ahead forecast at every
origin. Do not treat test_mape_mean as an apples-to-apples number across
model families when comparing ets/sarimax to gbm.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.statsmodels
import pandas as pd

from src.evaluate import _write_evaluation_report
from src.forecasting.rolling_origin import score_gbm_origins, score_statsmodels_origins, select_cv_origins
from src.utils.config import load_forecast_features_config, load_forecast_models_config, load_pipeline_config
from src.utils.io import resolve_run_path

logger = logging.getLogger(__name__)


def _select_forecast_champion(model_metrics: dict[str, dict[str, float | None]], champion_metric: str) -> str:
    """Pick the run champion: lowest value of champion_metric (MAPE — lower is
    better). Falls back to test_mape if any registered model is missing the
    configured metric — mirrors the tabular evaluate.py's cv_r2-missing
    fallback to test_rmse, applied here to cv_mape/test_mape instead.

    Args:
        model_metrics: {model_name: {"cv_mape": float|None, "test_mape": float}}.
        champion_metric: The metric key to prefer (currently always "cv_mape",
            per ForecastEvaluationConfig.champion_metric's Literal type).

    Returns:
        The winning model name.
    """
    key = champion_metric
    if not all(m.get(champion_metric) is not None for m in model_metrics.values()):
        logger.warning(
            "champion_metric='%s' missing for at least one model; falling back to test_mape",
            champion_metric,
        )
        key = "test_mape"
    return min(model_metrics, key=lambda name: model_metrics[name][key])


def register_forecast_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
    config_dir: str | Path = "config",
    run_id: str = "unknown",
    reports_dir: str | Path = "reports",
    features_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score each trained forecasting model on the held-out test set, register
    passing models to MLflow Staging, and tag the run champion.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_run_ids: Per-model MLflow run IDs from the train stage.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).
        run_id: Run identifier.
        reports_dir: Directory for the evaluation audit report.
        features_dir: Directory containing train/test parquet files. Required
            (unlike the tabular signature, where it's optional) — this
            stage's core job is scoring against the real test set.
        benchmark_dir: Unused — champion/challenger for forecasting is not
            designed yet (config/pjm_load_forecast/pipeline.yaml has
            benchmark.enabled: false). Accepted only to match the locked
            signature dag_factory.py calls.

    Returns:
        Dictionary with per-model registration results.

    Raises:
        ValueError: If mlflow_run_ids or features_dir is missing, or if no
            model could be registered.
        RuntimeError: If any model fails to score or register — infrastructure
            error (e.g. MLflow unreachable) or a modeling/scoring error (e.g.
            a model missing exog it was fit with) are both reported this way;
            see the per-model "error" field in the written report for which.
    """
    if not mlflow_run_ids:
        raise ValueError("mlflow_run_ids required for model registration")
    if features_dir is None:
        raise ValueError("features_dir required to load train/test parquet for rolling-origin test scoring")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    pipeline_cfg = load_pipeline_config(config_dir)
    models_cfg = load_forecast_models_config(config_dir)
    features_cfg = load_forecast_features_config(config_dir)
    eval_cfg = models_cfg.evaluation
    target_col = pipeline_cfg.target.name

    features_path = resolve_run_path(features_dir, run_id)
    train_df = pd.read_parquet(features_path / "train.parquet")
    test_df = pd.read_parquet(features_path / "test.parquet")
    full_df = pd.concat([train_df, test_df]).sort_index()
    full_y = full_df[target_col]
    feature_columns = [c for c in train_df.columns if c != target_col]

    origins = select_cv_origins(test_df.index, eval_cfg.n_windows, eval_cfg.horizon_hours)

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
    timestamp = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {"run_id": run_id, "evaluated_at": timestamp, "models": {}}
    registration_results: dict[str, Any] = {"registered_models": {}}
    infra_failures: list[str] = []
    model_metrics: dict[str, dict[str, float | None]] = {}

    for model_name, mlflow_run_id in mlflow_run_ids.items():
        try:
            run = mlflow.get_run(mlflow_run_id)
            metrics = run.data.metrics
            run_tags = run.data.tags
            model_type = run_tags.get("model_type", "unknown")
            pipeline_type = run_tags.get("pipeline_type", "unknown")
            cv_mape_mean = metrics.get("cv_mape_mean")

            model_uri = f"runs:/{mlflow_run_id}/model"
            if model_type in ("ets", "sarimax"):
                loaded = mlflow.statsmodels.load_model(model_uri)
                # SARIMAX fit with use_holiday_exog (see model_registry.fit_sarimax)
                # needs its exog column re-supplied for out-of-sample scoring —
                # k_exog > 0 tells us the loaded model actually has a regression
                # component (ETS never does; SARIMAX only does when configured).
                exog = None
                if getattr(loaded.model, "k_exog", 0) > 0 and "is_holiday" in full_df.columns:
                    exog = full_df["is_holiday"]
                window_scores = score_statsmodels_origins(
                    loaded, full_y, origins, eval_cfg.horizon_hours, dynamic=False, exog=exog,
                )
            elif model_type == "gbm":
                loaded = mlflow.lightgbm.load_model(model_uri)
                window_scores = score_gbm_origins(
                    loaded, full_df, target_col, feature_columns, origins, eval_cfg.horizon_hours,
                    features_cfg.lags, features_cfg.rolling_windows,
                    features_cfg.calendar_features, features_cfg.holiday_features,
                )
            else:
                raise ValueError(f"Unknown forecasting model type '{model_type}' for {model_name}")

            if not window_scores:
                raise ValueError(f"No scorable rolling-origin windows on the test set for {model_name}.")
            test_mape_mean = float(pd.Series(window_scores).mean())
            test_mape_std = float(pd.Series(window_scores).std())

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
                "test_mape_mean": f"{test_mape_mean:.4f}",
                "test_mape_std": f"{test_mape_std:.4f}",
            }
            if cv_mape_mean is not None:
                version_tags["cv_mape_mean"] = f"{cv_mape_mean:.4f}"
            for key, value in version_tags.items():
                client.set_model_version_tag(model_name, version, key, value)

            client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
            try:
                client.set_registered_model_alias(model_name, "staging", version)
            except Exception as alias_err:
                logger.warning("Could not set alias for %s v%s: %s", model_name, version, alias_err)

            logger.info(
                "REGISTERED %s v%s to Staging (test_mape_mean=%.2f%%, cv_mape_mean=%s)",
                model_name, version, test_mape_mean,
                f"{cv_mape_mean:.2f}%" if cv_mape_mean is not None else "n/a",
            )

            report["models"][model_name] = {
                "status": "registered",
                "version": version,
                "test_mape_mean": test_mape_mean,
                "test_mape_std": test_mape_std,
                "cv_mape_mean": cv_mape_mean,
                "n_windows_scored": len(window_scores),
            }
            model_metrics[model_name] = {"cv_mape": cv_mape_mean, "test_mape": test_mape_mean}
            registration_results["registered_models"][model_name] = {
                "status": "registered",
                "version": version,
                "stage": "Staging",
                "source_run_id": mlflow_run_id,
                "model_uri": model_uri,
                "test_mape_mean": test_mape_mean,
                "cv_mape_mean": cv_mape_mean,
                "registered_at": timestamp,
            }
        except Exception as e:
            logger.error("Error evaluating/registering %s: %s", model_name, e)
            report["models"][model_name] = {"status": "error", "error": str(e)}
            registration_results["registered_models"][model_name] = {
                "status": "error", "error": str(e), "source_run_id": mlflow_run_id,
            }
            infra_failures.append(model_name)

    registered = [k for k, v in report["models"].items() if v["status"] == "registered"]
    if registered:
        champion_name = _select_forecast_champion(
            {n: model_metrics[n] for n in registered}, eval_cfg.champion_metric,
        )
        report["run_champion"] = champion_name
        report["champion_metric"] = eval_cfg.champion_metric
        champion_version = report["models"][champion_name]["version"]
        try:
            client.set_model_version_tag(champion_name, champion_version, "run_champion", "true")
        except Exception as e:
            logger.warning("Could not tag run champion %s v%s: %s", champion_name, champion_version, e)
    else:
        report["run_champion"] = None

    _write_evaluation_report(report, run_id, reports_dir)

    logger.info(
        "Forecast evaluation complete: %d registered, %d errors", len(registered), len(infra_failures),
    )

    if infra_failures:
        raise RuntimeError(
            f"Registration failed for {len(infra_failures)} model(s): {', '.join(infra_failures)}."
        )
    if not registered:
        raise ValueError(f"No forecasting models could be scored/registered for run {run_id}.")

    return registration_results
