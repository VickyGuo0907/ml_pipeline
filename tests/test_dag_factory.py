"""Tests for the DAG factory's forecasting dispatch.

Airflow's DAG/PythonOperator can be constructed without a running
scheduler or initialized metadata DB - these tests build DAG objects
directly and inspect their structure, they do not execute any task.
"""
from src.dags.dag_factory import (
    _select_features_schema_builder,
    _select_forecasting_stage_functions,
    build_dag,
)
from src.forecasting.clean_forecast import clean_forecast_data
from src.forecasting.evaluate_forecast import register_forecast_models_to_mlflow
from src.forecasting.features_forecast import engineer_forecast_features
from src.forecasting.train_forecast import train_forecast_models
from src.schemas.features import build_features_schema, build_forecast_features_schema
from src.utils.config import ProblemType, load_pipeline_orchestration_config


def test_select_forecasting_stage_functions_for_forecasting():
    """A forecasting problem_type returns the four forecasting stage functions."""
    functions = _select_forecasting_stage_functions(ProblemType.FORECASTING)
    assert functions is not None
    assert functions["clean"] is clean_forecast_data
    assert functions["features"] is engineer_forecast_features
    assert functions["train"] is train_forecast_models
    assert functions["register"] is register_forecast_models_to_mlflow


def test_select_forecasting_stage_functions_for_regression():
    """A non-forecasting problem_type returns None - dag_factory falls back to tabular functions."""
    assert _select_forecasting_stage_functions(ProblemType.REGRESSION) is None
    assert _select_forecasting_stage_functions(ProblemType.CLASSIFICATION) is None


def test_select_features_schema_builder_dispatch():
    """The schema builder dispatch picks the DatetimeIndex schema only for forecasting."""
    assert _select_features_schema_builder(ProblemType.FORECASTING) is build_forecast_features_schema
    assert _select_features_schema_builder(ProblemType.REGRESSION) is build_features_schema


def test_pjm_forecast_dag_builds_with_expected_task_ids():
    """The pjm_load_forecast DAG builds and exposes the same 9-stage task ID set as every other pipeline."""
    config = load_pipeline_orchestration_config("config/pjm_load_forecast", base_dir="config/base")
    dag = build_dag(config)
    task_ids = set(dag.task_ids)
    assert "04_clean_data" in task_ids
    assert "05_engineer_features" in task_ids
    assert "07_train_models" in task_ids
    assert "08_register_to_mlflow" in task_ids
    assert dag.dag_id == "pjm_load_forecast_pipeline"


def test_hospital_readmission_lagged_dag_still_builds_unchanged():
    """Regression check: an existing (non-forecasting) pipeline's DAG is unaffected by the dispatch change."""
    config = load_pipeline_orchestration_config(
        "config/hospital_readmission_lagged", base_dir="config/base",
    )
    dag = build_dag(config)
    assert dag.dag_id == "hospital_readmission_lagged_pipeline"
    assert "07_train_models" in dag.task_ids
