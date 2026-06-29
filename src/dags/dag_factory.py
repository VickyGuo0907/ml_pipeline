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

from airflow import DAG  # noqa: E402
from airflow.models import TaskInstance  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

from src.clean import clean_raw_data  # noqa: E402
from src.evaluate import register_models_to_mlflow  # noqa: E402
from src.explore import run_unsupervised_analysis  # noqa: E402
from src.features import engineer_features  # noqa: E402
from src.ingest import ingest_files  # noqa: E402
from src.monitoring import generate_drift_report  # noqa: E402
from src.profile import profile_raw_files  # noqa: E402
from src.train import train_models  # noqa: E402
from src.utils.config import OrchestrationConfig, discover_pipelines, load_pipeline_orchestration_config  # noqa: E402
from src.validate import validate_raw_files  # noqa: E402

import pandas as pd  # noqa: E402
from src.schemas.features import features_schema  # noqa: E402


def build_dag(config: OrchestrationConfig) -> DAG:
    """Build a complete pipeline DAG from an orchestration config.

    Args:
        config: Merged orchestration config for this pipeline

    Returns:
        Configured Airflow DAG with all pipeline tasks wired
    """
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
        return ti.xcom_pull(task_ids="01_ingest_files", key="run_id")

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
        """Validate raw data against pandera schema."""
        return validate_raw_files(
            raw_dir=config.directories.raw,
            run_id=_pull_run_id(context),
        )

    def profile_wrapper(**context) -> dict:
        """Generate ydata-profiling HTML reports."""
        return profile_raw_files(
            raw_dir=config.directories.raw,
            run_id=_pull_run_id(context),
            reports_dir=config.directories.reports,
        )

    def clean_wrapper(**context) -> dict:
        """Clean raw data: impute, drop bad cols, dedup."""
        return clean_raw_data(
            raw_dir=config.directories.raw,
            interim_dir=config.directories.interim,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    def features_wrapper(**context) -> dict:
        """Engineer features: encode, Box-Cox, VIF, scale, split."""
        return engineer_features(
            interim_dir=config.directories.interim,
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    def validate_features_wrapper(**context) -> dict:
        """Validate feature matrix with pandera schema."""
        run_id = _pull_run_id(context)
        train_df = pd.read_parquet(f"{config.directories.features}/{run_id}/train.parquet")
        features_schema.validate(train_df)
        return {"validated_rows": len(train_df)}

    def explore_wrapper(**context) -> dict:
        """PCA + k-means unsupervised analysis (runs in parallel with validate_features)."""
        return run_unsupervised_analysis(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
            reports_dir=config.directories.reports,
        )

    def train_wrapper(**context) -> dict:
        """Train all models and log R² + RMSE to MLflow."""
        result = train_models(
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
        """Register all trained models to MLflow Staging."""
        ti: TaskInstance = context["task_instance"]
        mlflow_run_ids = ti.xcom_pull(task_ids="07_train_models", key="mlflow_run_ids")
        return register_models_to_mlflow(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_run_ids=mlflow_run_ids,
        )

    def drift_wrapper(**context) -> dict:
        """Generate Evidently drift report comparing current vs previous features."""
        run_id = _pull_run_id(context)
        if not run_id:
            raise ValueError(f"[{config.dag.dag_id}] run_id not found in cross-task storage")
        return generate_drift_report(
            features_dir=config.directories.features,
            run_id=run_id,
            previous_run_id=None,
            reports_dir=config.directories.reports,
        )

    with dag:
        ingest_task = PythonOperator(
            task_id="01_ingest_files",
            python_callable=ingest_wrapper,

        )
        validate_raw_task = PythonOperator(
            task_id="02_validate_raw_schema",
            python_callable=validate_raw_wrapper,

        )
        profile_task = PythonOperator(
            task_id="03_profile_data",
            python_callable=profile_wrapper,

        )
        clean_task = PythonOperator(
            task_id="04_clean_data",
            python_callable=clean_wrapper,

        )
        feature_task = PythonOperator(
            task_id="05_engineer_features",
            python_callable=features_wrapper,

        )
        validate_features_task = PythonOperator(
            task_id="06_validate_features_schema",
            python_callable=validate_features_wrapper,

        )
        explore_task = PythonOperator(
            task_id="06b_unsupervised_explore",
            python_callable=explore_wrapper,

        )
        train_task = PythonOperator(
            task_id="07_train_models",
            python_callable=train_wrapper,

            retries=config.tasks.train_models_retries,
        )
        register_task = PythonOperator(
            task_id="08_register_to_mlflow",
            python_callable=register_wrapper,

        )
        drift_task = PythonOperator(
            task_id="09_drift_report",
            python_callable=drift_wrapper,

        )

        # Pipeline flow — explore runs in parallel with validate_features
        (
            ingest_task
            >> validate_raw_task
            >> profile_task
            >> clean_task
            >> feature_task
            >> [validate_features_task, explore_task]
        )
        validate_features_task >> train_task >> register_task >> drift_task

    return dag


# Register one DAG per pipeline config directory into Airflow's global namespace
for _pipeline_dir in discover_pipelines("config"):
    _config = load_pipeline_orchestration_config(_pipeline_dir, base_dir="config/base")
    globals()[_config.dag.dag_id] = build_dag(_config)
