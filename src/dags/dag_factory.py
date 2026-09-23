"""DAG factory — generates one Airflow DAG per pipeline config directory.

Scans config/ for directories that contain orchestration.yaml, merges each
with config/base/defaults.yaml, and registers a DAG in globals().

Adding a new pipeline requires only a new config/<pipeline>/ directory with
an orchestration.yaml — no changes to this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta  # noqa: E402
import logging  # noqa: E402

from airflow import DAG  # noqa: E402
from airflow.models import TaskInstance  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

from typing import Any, Callable  # noqa: E402

from src.benchmark import create_benchmark_snapshot  # noqa: E402
from src.clean import clean_raw_data  # noqa: E402
from src.evaluate import register_models_to_mlflow  # noqa: E402
from src.explore import run_unsupervised_analysis  # noqa: E402
from src.features import engineer_features  # noqa: E402
from src.finance.clean_finance import clean_finance_data  # noqa: E402
from src.finance.evaluate_finance import register_finance_models_to_mlflow  # noqa: E402
from src.finance.features_finance import engineer_finance_features  # noqa: E402
from src.finance.train_finance import train_finance_models  # noqa: E402
from src.forecasting.clean_forecast import clean_forecast_data  # noqa: E402
from src.forecasting.evaluate_forecast import register_forecast_models_to_mlflow  # noqa: E402
from src.forecasting.features_forecast import engineer_forecast_features  # noqa: E402
from src.forecasting.train_forecast import train_forecast_models  # noqa: E402
from src.ingest import ingest_files  # noqa: E402
from src.monitoring import generate_drift_report  # noqa: E402
from src.profile import generate_mstl_report, profile_raw_files  # noqa: E402
from src.train import train_models  # noqa: E402
from src.utils.config import OrchestrationConfig, ProblemType, discover_pipelines, load_pipeline_config, load_pipeline_orchestration_config  # noqa: E402
from src.utils.io import find_previous_run_id, resolve_run_path  # noqa: E402
from src.validate import validate_raw_files  # noqa: E402

import pandas as pd  # noqa: E402
from src.schemas.features import build_features_schema, build_finance_features_schema, build_forecast_features_schema  # noqa: E402

logger = logging.getLogger(__name__)

# Single source of truth for task IDs — used in both task_id= definitions
# and xcom_pull(task_ids=...) calls so renaming a task never silently breaks XCom.
_TASK_INGEST = "01_ingest_files"
_TASK_VALIDATE_RAW = "02_validate_raw_schema"
_TASK_PROFILE = "03_profile_data"
_TASK_CLEAN = "04_clean_data"
_TASK_FEATURES = "05_engineer_features"
_TASK_VALIDATE_FEATURES = "06_validate_features_schema"
_TASK_EXPLORE = "06b_unsupervised_explore"
_TASK_BENCHMARK = "06c_create_benchmark"
_TASK_TRAIN = "07_train_models"
_TASK_REGISTER = "08_register_to_mlflow"
_TASK_DRIFT = "09_drift_report"


def _select_forecasting_stage_functions(problem_type: ProblemType) -> dict[str, Callable] | None:
    """Return forecasting stage functions for a forecasting pipeline, else None.

    None signals "use the default tabular functions" - build_dag() falls
    back to clean_raw_data/engineer_features/train_models/register_models_to_mlflow
    unchanged for every problem_type except forecasting, so the three
    existing pipelines are unaffected by this dispatch.

    Args:
        problem_type: The pipeline's problem_type from pipeline.yaml.

    Returns:
        Dict of the four forecasting stage functions, or None.
    """
    if problem_type != ProblemType.FORECASTING:
        return None
    return {
        "clean": clean_forecast_data,
        "features": engineer_forecast_features,
        "train": train_forecast_models,
        "register": register_forecast_models_to_mlflow,
    }


def _select_finance_stage_functions(problem_type: ProblemType) -> dict[str, Callable] | None:
    """Return finance stage functions for a finance pipeline, else None.

    None signals "use the default tabular functions" - build_dag() falls
    back to clean_raw_data/engineer_features/train_models/register_models_to_mlflow
    unchanged for every problem_type except finance, so the three tabular
    pipelines and pjm_load_forecast (forecasting) are unaffected by this
    dispatch.

    Args:
        problem_type: The pipeline's problem_type from pipeline.yaml.

    Returns:
        Dict of the four finance stage functions, or None.
    """
    if problem_type != ProblemType.FINANCE:
        return None
    return {
        "clean": clean_finance_data,
        "features": engineer_finance_features,
        "train": train_finance_models,
        "register": register_finance_models_to_mlflow,
    }


def _select_features_schema_builder(problem_type: ProblemType) -> Callable[[str], Any]:
    """Return the feature-matrix schema builder appropriate for this problem_type.

    Forecasting feature matrices carry a DatetimeIndex (row order/spacing is
    meaningful); finance feature matrices are a long-format panel (plain
    integer index, many rows sharing the same Date across assets); tabular
    feature matrices carry a plain integer index too. Finance and tabular
    feature schemas are currently structurally identical (both use coerce=True,
    strict=False, and a plain integer index), kept as separate functions only
    so finance-specific column requirements can be added later without touching
    the tabular schema's contract.

    Args:
        problem_type: The pipeline's problem_type from pipeline.yaml.

    Returns:
        build_forecast_features_schema for forecasting pipelines,
        build_finance_features_schema for finance pipelines,
        build_features_schema for every other problem_type.
    """
    if problem_type == ProblemType.FORECASTING:
        return build_forecast_features_schema
    if problem_type == ProblemType.FINANCE:
        return build_finance_features_schema
    return build_features_schema


def build_dag(config: OrchestrationConfig) -> DAG:
    """Build a complete pipeline DAG from an orchestration config.

    Args:
        config: Merged orchestration config for this pipeline

    Returns:
        Configured Airflow DAG with all pipeline tasks wired
    """
    pipeline_cfg = load_pipeline_config(config.directories.config)
    # Exactly one of these returns non-None for any given problem_type (or
    # neither does, for tabular) - forecasting is checked first only because
    # it was added first, the two are mutually exclusive by construction.
    _stage_fns = (
        _select_forecasting_stage_functions(pipeline_cfg.problem_type)
        or _select_finance_stage_functions(pipeline_cfg.problem_type)
    )
    _clean_fn = _stage_fns["clean"] if _stage_fns else clean_raw_data
    _features_fn = _stage_fns["features"] if _stage_fns else engineer_features
    _train_fn = _stage_fns["train"] if _stage_fns else train_models
    _register_fn = _stage_fns["register"] if _stage_fns else register_models_to_mlflow
    _features_schema_builder = _select_features_schema_builder(pipeline_cfg.problem_type)

    default_args = {
        "owner": config.dag.owner,
        "start_date": datetime.strptime(config.dag.start_date, "%Y-%m-%d"),
        "retries": config.tasks.retries,
        "retry_delay": timedelta(minutes=config.tasks.retry_delay_minutes),
        "email_on_failure": False,
        "email_on_retry": False,
    }

    dag = DAG(
        config.dag.dag_id,
        default_args=default_args,
        description=config.dag.description,
        schedule=config.dag.schedule,
        catchup=config.dag.catchup,
        tags=config.dag.tags,
    )

    def _pull_run_id(context: dict) -> str:
        """Pull run_id pushed by the ingest task."""
        ti: TaskInstance = context["task_instance"]
        return ti.xcom_pull(task_ids=_TASK_INGEST, key="run_id")

    def ingest_wrapper(**context) -> dict:
        """Ingest data and push run_id to cross-task storage."""
        result = ingest_files(
            landing_dir=config.directories.landing,
            raw_dir=config.directories.raw,
            run_id=context["ds"],
        )
        ti: TaskInstance = context["task_instance"]
        ti.xcom_push(key="run_id", value=result["run_id"])
        return result

    def validate_raw_wrapper(**context) -> dict:
        """Validate raw data against config-driven pandera schema."""
        return validate_raw_files(
            raw_dir=config.directories.raw,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    def profile_wrapper(**context) -> dict:
        """Generate ydata-profiling HTML reports, plus an MSTL seasonality
        decomposition report for forecasting pipelines."""
        run_id = _pull_run_id(context)
        result = profile_raw_files(
            raw_dir=config.directories.raw,
            run_id=run_id,
            reports_dir=config.directories.reports,
            config_dir=config.directories.config,
        )
        if pipeline_cfg.problem_type == ProblemType.FORECASTING:
            try:
                result["mstl"] = generate_mstl_report(
                    raw_dir=config.directories.raw,
                    run_id=run_id,
                    reports_dir=config.directories.reports,
                    config_dir=config.directories.config,
                )
            except Exception as e:
                logger.warning("MSTL report generation failed: %s", e)
        return result

    def clean_wrapper(**context) -> dict:
        """Clean raw data. Tabular pipelines: impute, drop bad cols, dedup.
        Forecasting pipelines: gap detection/fill on the hourly series
        (dispatched via _clean_fn, set above from problem_type)."""
        return _clean_fn(
            raw_dir=config.directories.raw,
            interim_dir=config.directories.interim,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    def features_wrapper(**context) -> dict:
        """Engineer features. Tabular pipelines: encode, Box-Cox, VIF, scale,
        split. Forecasting pipelines: lag/rolling/calendar/holiday features,
        chronological split (dispatched via _features_fn, set above from
        problem_type)."""
        return _features_fn(
            interim_dir=config.directories.interim,
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    def validate_features_wrapper(**context) -> dict:
        """Validate feature matrix: target column type, all-numeric, minimum row count."""
        run_id = _pull_run_id(context)
        train_df = pd.read_parquet(resolve_run_path(config.directories.features, run_id) / "train.parquet")

        target_col = pipeline_cfg.target.name

        if len(train_df) < 100:
            raise ValueError(f"Train set too small: {len(train_df)} rows (minimum 100)")

        non_numeric = train_df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns in feature matrix: {non_numeric}")

        _features_schema_builder(target_col).validate(train_df)
        return {"validated_rows": len(train_df), "target_col": target_col}

    def explore_wrapper(**context) -> dict:
        """PCA + k-means unsupervised analysis (runs in parallel with validate_features)."""
        return run_unsupervised_analysis(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
            reports_dir=config.directories.reports,
        )

    def benchmark_wrapper(**context) -> dict:
        """Refresh the fixed benchmark set — no-op unless triggered with conf.refresh_benchmark."""
        conf = context["dag_run"].conf or {}
        if not conf.get("refresh_benchmark", False):
            logger.info(
                "Skipping benchmark refresh — trigger with conf={'refresh_benchmark': true} to refresh"
            )
            return {"skipped": True}
        return create_benchmark_snapshot(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            benchmark_dir=config.directories.benchmark,
        )

    def train_wrapper(**context) -> dict:
        """Train all models. Tabular pipelines: log R² + RMSE to MLflow.
        Forecasting pipelines: rolling-origin CV, log cv_mape_mean/cv_mape_std
        (dispatched via _train_fn, set above from problem_type)."""
        result = _train_fn(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
            mlflow_tracking_uri=config.mlflow.tracking_uri,
        )
        run_ids = {name: info["mlflow_run_id"] for name, info in result["models"].items()}
        ti: TaskInstance = context["task_instance"]
        ti.xcom_push(key="mlflow_run_ids", value=run_ids)
        return result

    def register_wrapper(**context) -> dict:
        """Evaluate models against thresholds and register passing ones to MLflow Staging."""
        ti: TaskInstance = context["task_instance"]
        mlflow_run_ids = ti.xcom_pull(task_ids=_TASK_TRAIN, key="mlflow_run_ids")
        return _register_fn(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_run_ids=mlflow_run_ids,
            config_dir=config.directories.config,
            run_id=_pull_run_id(context),
            reports_dir=config.directories.reports,
            features_dir=config.directories.features,
            benchmark_dir=config.directories.benchmark,
        )

    def drift_wrapper(**context) -> dict:
        """Generate Evidently drift report comparing current vs previous features."""
        run_id = _pull_run_id(context)
        if not run_id:
            raise ValueError(f"[{config.dag.dag_id}] run_id not found in cross-task storage")
        previous_run_id = find_previous_run_id(config.directories.features, run_id)
        return generate_drift_report(
            features_dir=config.directories.features,
            run_id=run_id,
            previous_run_id=previous_run_id,
            reports_dir=config.directories.reports,
        )

    _reports_url = config.directories.reports_base_url
    _enabled = config.tasks.enabled

    with dag:
        # --- Always-required tasks ---
        ingest_task = PythonOperator(
            task_id=_TASK_INGEST,
            python_callable=ingest_wrapper,
        )
        validate_raw_task = PythonOperator(
            task_id=_TASK_VALIDATE_RAW,
            python_callable=validate_raw_wrapper,
        )
        clean_task = PythonOperator(
            task_id=_TASK_CLEAN,
            python_callable=clean_wrapper,
        )
        feature_task = PythonOperator(
            task_id=_TASK_FEATURES,
            python_callable=features_wrapper,
        )
        validate_features_task = PythonOperator(
            task_id=_TASK_VALIDATE_FEATURES,
            python_callable=validate_features_wrapper,
        )
        train_task = PythonOperator(
            task_id=_TASK_TRAIN,
            python_callable=train_wrapper,
            retries=config.tasks.train_models_retries,
        )
        benchmark_task = PythonOperator(
            task_id=_TASK_BENCHMARK,
            python_callable=benchmark_wrapper,
            doc_md=(
                "## Benchmark Snapshot\n\n"
                "No-op on a normal scheduled run. Trigger this DAG with "
                "`conf={\"refresh_benchmark\": true}` to refresh the fixed "
                "benchmark set used for champion/challenger regression checks."
            ),
        )
        register_task = PythonOperator(
            task_id=_TASK_REGISTER,
            python_callable=register_wrapper,
        )

        # --- Optional tasks (controlled by orchestration.yaml tasks.enabled) ---
        if _enabled.profile:
            profile_task = PythonOperator(
                task_id=_TASK_PROFILE,
                python_callable=profile_wrapper,
                doc_md=f"## Data Profile Reports\n\n[Open reports →]({_reports_url}/)",
            )

        if _enabled.unsupervised_explore:
            explore_task = PythonOperator(
                task_id=_TASK_EXPLORE,
                python_callable=explore_wrapper,
                doc_md=f"## Unsupervised Analysis Reports\n\n[Open reports →]({_reports_url}/)",
            )

        if _enabled.drift_report:
            drift_task = PythonOperator(
                task_id=_TASK_DRIFT,
                python_callable=drift_wrapper,
                doc_md=f"## Drift Report\n\n[Open reports →]({_reports_url}/)",
            )

        # --- Dependency wiring ---
        # Core: ingest → validate_raw → [profile?] → clean → features
        chain = ingest_task >> validate_raw_task
        if _enabled.profile:
            chain = chain >> profile_task
        chain = chain >> clean_task >> feature_task

        # After features: explore runs in parallel with validate_features when enabled
        if _enabled.unsupervised_explore:
            chain >> [validate_features_task, explore_task]
        else:
            chain >> validate_features_task

        # Core training chain: validate_features → benchmark → train → register → [drift?]
        end = validate_features_task >> benchmark_task >> train_task >> register_task
        if _enabled.drift_report:
            end >> drift_task

    return dag


# Register one DAG per pipeline config directory into Airflow's global namespace
for _pipeline_dir in discover_pipelines("config"):
    _config = load_pipeline_orchestration_config(_pipeline_dir, base_dir="config/base")
    globals()[_config.dag.dag_id] = build_dag(_config)
