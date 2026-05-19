"""ML Pipeline DAG - End-to-end data pipeline with Airflow.

Orchestrates: ingest → validate → profile → clean → feature_engineer →
validate_features → train → evaluate/register → drift_report
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add airflow working directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clean import clean_raw_data
from src.evaluate import register_models_to_mlflow
from src.features import engineer_features
from src.ingest import ingest_files
from src.monitoring import generate_drift_report
from src.profile import profile_raw_files
from src.train import train_models
from src.utils import load_orchestration_config
from src.validate import validate_raw_files

# Load orchestration configuration
config = load_orchestration_config("config")

# Build default arguments from config
default_args = {
    "owner": config.dag.owner,
    "retries": config.tasks.retries,
    "retry_delay": timedelta(minutes=config.tasks.retry_delay_minutes),
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
}

# DAG definition
with DAG(
    config.dag.dag_id,
    default_args=default_args,
    description=config.dag.description,
    schedule_interval=config.dag.schedule_interval,
    catchup=config.dag.catchup,
    tags=config.dag.tags,
) as dag:

    # Stage 1: Ingest
    def ingest_wrapper(**context):
        """Ingest data and pass run_id downstream."""
        result = ingest_files(
            landing_dir=config.directories.landing,
            raw_dir=config.directories.raw,
            run_id=context["ds"],  # Use Airflow logical date as run_id
        )
        context["task_instance"].xcom_push(key="run_id", value=result["run_id"])
        return result

    ingest_task = PythonOperator(
        task_id="01_ingest_files",
        python_callable=ingest_wrapper,
        provide_context=True,
        doc="Move files from data/landing to data/raw/<run_id>, write manifest.yaml",
    )

    # Stage 2: Validate Raw
    def validate_raw_wrapper(**context):
        """Validate raw data against schema."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        result = validate_raw_files(
            raw_dir=config.directories.raw,
            run_id=run_id,
        )
        return result

    validate_raw_task = PythonOperator(
        task_id="02_validate_raw_schema",
        python_callable=validate_raw_wrapper,
        provide_context=True,
        doc="Pandera schema check per source file",
    )

    # Stage 3: Profile
    def profile_wrapper(**context):
        """Generate data profiling reports."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        result = profile_raw_files(
            raw_dir=config.directories.raw,
            run_id=run_id,
            reports_dir=config.directories.reports,
        )
        return result

    profile_task = PythonOperator(
        task_id="03_profile_data",
        python_callable=profile_wrapper,
        provide_context=True,
        doc="ydata-profiling HTML report per source",
    )

    # Stage 4: Clean
    def clean_wrapper(**context):
        """Clean raw data."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        result = clean_raw_data(
            raw_dir=config.directories.raw,
            interim_dir=config.directories.interim,
            run_id=run_id,
        )
        return result

    clean_task = PythonOperator(
        task_id="04_clean_data",
        python_callable=clean_wrapper,
        provide_context=True,
        doc="Type coercion, missing handling, dedup → data/interim/<run_id>",
    )

    # Stage 5: Feature Engineer
    def features_wrapper(**context):
        """Engineer features."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        result = engineer_features(
            interim_dir=config.directories.interim,
            features_dir=config.directories.features,
            run_id=run_id,
            config_dir=config.directories.config,
        )
        return result

    feature_task = PythonOperator(
        task_id="05_engineer_features",
        python_callable=features_wrapper,
        provide_context=True,
        doc="Joins, encoding, NZV filter, train/test split → data/features/<run_id>",
    )

    # Stage 6: Validate Features
    def validate_features_wrapper(**context):
        """Validate feature matrix."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        # Load and validate feature matrix
        from src.schemas.features import features_schema
        import pandas as pd

        features_path = f"{config.directories.features}/{run_id}"
        train_df = pd.read_parquet(f"{features_path}/train.parquet")
        features_schema.validate(train_df)
        return {"validated_rows": len(train_df)}

    validate_features_task = PythonOperator(
        task_id="06_validate_features_schema",
        python_callable=validate_features_wrapper,
        provide_context=True,
        doc="Pandera schema check on feature matrix",
    )

    # Stage 7: Train models
    def train_wrapper(**context):
        """Train both models."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        result = train_models(
            features_dir=config.directories.features,
            run_id=run_id,
            config_dir=config.directories.config,
            mlflow_tracking_uri=config.mlflow.tracking_uri,
        )
        # Store run IDs for registration
        run_ids = {
            name: info["mlflow_run_id"]
            for name, info in result["models"].items()
        }
        context["task_instance"].xcom_push(key="mlflow_run_ids", value=run_ids)
        return result

    train_linear_task = PythonOperator(
        task_id="07_train_models",
        python_callable=train_wrapper,
        provide_context=True,
        retries=config.tasks.train_models_retries,
        doc="Train sklearn linear regression and LightGBM, log metrics to MLflow",
    )

    # Stage 8: Register Models
    def register_wrapper(**context):
        """Register trained models to MLflow."""
        mlflow_run_ids = context["task_instance"].xcom_pull(
            task_ids="07_train_models", key="mlflow_run_ids"
        )
        result = register_models_to_mlflow(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_run_ids=mlflow_run_ids,
        )
        return result

    register_task = PythonOperator(
        task_id="08_register_to_mlflow",
        python_callable=register_wrapper,
        provide_context=True,
        doc="Register both models to MLflow Staging (NO auto-promotion)",
    )

    # Stage 9: Drift Report
    def drift_wrapper(**context):
        """Generate drift report."""
        run_id = context["task_instance"].xcom_pull(
            task_ids="01_ingest_files", key="run_id"
        )
        if not run_id:
            raise ValueError("run_id not found in xcom from ingest task")
        result = generate_drift_report(
            features_dir=config.directories.features,
            run_id=run_id,
            previous_run_id=None,  # Can be updated to compare with previous runs
            reports_dir=config.directories.reports,
        )
        return result

    drift_task = PythonOperator(
        task_id="09_drift_report",
        python_callable=drift_wrapper,
        provide_context=True,
        doc="Evidently AI drift comparison: current features vs previous training set",
    )

    # Pipeline flow
    (
        ingest_task
        >> validate_raw_task
        >> profile_task
        >> clean_task
        >> feature_task
        >> validate_features_task
        >> train_linear_task
        >> register_task
        >> drift_task
    )
