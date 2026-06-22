"""ML Pipeline DAG — End-to-end data pipeline with Airflow.

Orchestrates:
  ingest → validate_raw → profile → clean → feature_engineer
  → [validate_features, explore (parallel)]
  → train → evaluate/register → drift_report

The active pipeline variant is set by orchestration.yaml → directories.config.
Swap to config/fraud, config/claims, etc. to run a different pipeline.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clean import clean_raw_data
from src.evaluate import register_models_to_mlflow
from src.explore import run_unsupervised_analysis
from src.features import engineer_features
from src.ingest import ingest_files
from src.monitoring import generate_drift_report
from src.profile import profile_raw_files
from src.train import train_models
from src.utils import load_orchestration_config
from src.validate import validate_raw_files

config = load_orchestration_config("config")

default_args = {
    "owner": config.dag.owner,
    "retries": config.tasks.retries,
    "retry_delay": timedelta(minutes=config.tasks.retry_delay_minutes),
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    config.dag.dag_id,
    default_args=default_args,
    description=config.dag.description,
    schedule_interval=config.dag.schedule_interval,
    catchup=config.dag.catchup,
    tags=config.dag.tags,
) as dag:

    def ingest_wrapper(**context: dict) -> dict:
        """Ingest data and push run_id to XCom."""
        result = ingest_files(
            landing_dir=config.directories.landing,
            raw_dir=config.directories.raw,
            run_id=context["ds"],
        )
        context["task_instance"].xcom_push(key="run_id", value=result["run_id"])
        return result

    ingest_task = PythonOperator(
        task_id="01_ingest_files",
        python_callable=ingest_wrapper,
        provide_context=True,
        doc="Move files from data/landing to data/raw/<run_id>, write manifest.yaml",
    )

    def _pull_run_id(context: dict) -> str:
        return context["task_instance"].xcom_pull(task_ids="01_ingest_files", key="run_id")

    def validate_raw_wrapper(**context: dict) -> dict:
        """Validate raw data against pandera schema."""
        return validate_raw_files(raw_dir=config.directories.raw, run_id=_pull_run_id(context))

    validate_raw_task = PythonOperator(
        task_id="02_validate_raw_schema",
        python_callable=validate_raw_wrapper,
        provide_context=True,
    )

    def profile_wrapper(**context: dict) -> dict:
        """Generate ydata-profiling HTML reports."""
        return profile_raw_files(
            raw_dir=config.directories.raw,
            run_id=_pull_run_id(context),
            reports_dir=config.directories.reports,
        )

    profile_task = PythonOperator(
        task_id="03_profile_data",
        python_callable=profile_wrapper,
        provide_context=True,
    )

    def clean_wrapper(**context: dict) -> dict:
        """Clean raw data: impute, drop bad cols, dedup."""
        return clean_raw_data(
            raw_dir=config.directories.raw,
            interim_dir=config.directories.interim,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    clean_task = PythonOperator(
        task_id="04_clean_data",
        python_callable=clean_wrapper,
        provide_context=True,
        doc="Type coercion, MICE imputation, pattern drops → data/interim/<run_id>",
    )

    def features_wrapper(**context: dict) -> dict:
        """Engineer features: encode, Box-Cox, VIF, scale, split."""
        return engineer_features(
            interim_dir=config.directories.interim,
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
        )

    feature_task = PythonOperator(
        task_id="05_engineer_features",
        python_callable=features_wrapper,
        provide_context=True,
        doc="freq-encode → Box-Cox → VIF prune → scale → train/test split",
    )

    def validate_features_wrapper(**context: dict) -> dict:
        """Validate feature matrix with pandera schema."""
        import pandas as pd
        from src.schemas.features import features_schema

        run_id = _pull_run_id(context)
        features_path = f"{config.directories.features}/{run_id}"
        train_df = pd.read_parquet(f"{features_path}/train.parquet")
        features_schema.validate(train_df)
        return {"validated_rows": len(train_df)}

    validate_features_task = PythonOperator(
        task_id="06_validate_features_schema",
        python_callable=validate_features_wrapper,
        provide_context=True,
    )

    def explore_wrapper(**context: dict) -> dict:
        """SVG Stage 3a: PCA + k-means unsupervised analysis (non-blocking)."""
        return run_unsupervised_analysis(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
            reports_dir=config.directories.reports,
        )

    explore_task = PythonOperator(
        task_id="06b_unsupervised_explore",
        python_callable=explore_wrapper,
        provide_context=True,
        doc="PCA variance analysis + k-means segmentation on engineered features",
    )

    def train_wrapper(**context: dict) -> dict:
        """Train all models from registry and log R² + RMSE to MLflow."""
        result = train_models(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            config_dir=config.directories.config,
            mlflow_tracking_uri=config.mlflow.tracking_uri,
        )
        run_ids = {name: info["mlflow_run_id"] for name, info in result["models"].items()}
        context["task_instance"].xcom_push(key="mlflow_run_ids", value=run_ids)
        return result

    train_task = PythonOperator(
        task_id="07_train_models",
        python_callable=train_wrapper,
        provide_context=True,
        retries=config.tasks.train_models_retries,
        doc="Train OLS, ElasticNet, Ridge, LASSO, RF, LightGBM — log R² + RMSE to MLflow",
    )

    def register_wrapper(**context: dict) -> dict:
        """Register all trained models to MLflow Staging."""
        mlflow_run_ids = context["task_instance"].xcom_pull(
            task_ids="07_train_models", key="mlflow_run_ids"
        )
        return register_models_to_mlflow(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_run_ids=mlflow_run_ids,
        )

    register_task = PythonOperator(
        task_id="08_register_to_mlflow",
        python_callable=register_wrapper,
        provide_context=True,
        doc="Register models to MLflow Staging (NO auto-promotion to Production)",
    )

    def drift_wrapper(**context: dict) -> dict:
        """Generate Evidently drift report comparing current vs previous features."""
        run_id = _pull_run_id(context)
        if not run_id:
            raise ValueError("run_id not found in XCom from ingest task")
        return generate_drift_report(
            features_dir=config.directories.features,
            run_id=run_id,
            previous_run_id=None,
            reports_dir=config.directories.reports,
        )

    drift_task = PythonOperator(
        task_id="09_drift_report",
        python_callable=drift_wrapper,
        provide_context=True,
    )

    # Pipeline flow
    # explore runs in parallel with validate_features — neither blocks the other
    (
        ingest_task
        >> validate_raw_task
        >> profile_task
        >> clean_task
        >> feature_task
        >> [validate_features_task, explore_task]
    )
    validate_features_task >> train_task >> register_task >> drift_task
