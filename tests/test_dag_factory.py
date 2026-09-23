"""Tests for the DAG factory's forecasting dispatch.

Airflow's DAG/PythonOperator can be constructed without a running
scheduler or initialized metadata DB - these tests build DAG objects
directly and inspect their structure, they do not execute any task.
"""
from pathlib import Path

from src.dags.dag_factory import (
    _select_features_schema_builder,
    _select_finance_stage_functions,
    _select_forecasting_stage_functions,
    build_dag,
)
from src.finance.clean_finance import clean_finance_data
from src.finance.evaluate_finance import register_finance_models_to_mlflow
from src.finance.features_finance import engineer_finance_features
from src.finance.train_finance import train_finance_models
from src.forecasting.clean_forecast import clean_forecast_data
from src.forecasting.evaluate_forecast import register_forecast_models_to_mlflow
from src.forecasting.features_forecast import engineer_forecast_features
from src.forecasting.train_forecast import train_forecast_models
from src.schemas.features import build_features_schema, build_finance_features_schema, build_forecast_features_schema
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
    """The schema builder dispatch picks the right schema per problem_type."""
    assert _select_features_schema_builder(ProblemType.FORECASTING) is build_forecast_features_schema
    assert _select_features_schema_builder(ProblemType.FINANCE) is build_finance_features_schema
    assert _select_features_schema_builder(ProblemType.REGRESSION) is build_features_schema


def test_select_finance_stage_functions_for_finance():
    """A finance problem_type returns the four finance stage functions."""
    functions = _select_finance_stage_functions(ProblemType.FINANCE)
    assert functions is not None
    assert functions["clean"] is clean_finance_data
    assert functions["features"] is engineer_finance_features
    assert functions["train"] is train_finance_models
    assert functions["register"] is register_finance_models_to_mlflow


def test_select_finance_stage_functions_for_regression():
    """A non-finance problem_type returns None - dag_factory falls back to tabular functions."""
    assert _select_finance_stage_functions(ProblemType.REGRESSION) is None
    assert _select_finance_stage_functions(ProblemType.CLASSIFICATION) is None
    assert _select_finance_stage_functions(ProblemType.FORECASTING) is None


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


def test_m6_returns_risk_dag_builds_with_expected_task_ids():
    """The m6_returns_risk DAG builds and exposes the same 9-stage task ID set as every other pipeline."""
    config = load_pipeline_orchestration_config("config/m6_returns_risk", base_dir="config/base")
    dag = build_dag(config)
    task_ids = set(dag.task_ids)
    assert "04_clean_data" in task_ids
    assert "05_engineer_features" in task_ids
    assert "07_train_models" in task_ids
    assert "08_register_to_mlflow" in task_ids
    assert dag.dag_id == "m6_returns_risk_pipeline"


def test_dispatch_wrapper_docstrings_are_dispatch_aware():
    """clean_wrapper/features_wrapper/train_wrapper's docstrings must not claim
    tabular-only behavior (encode/Box-Cox/VIF, R²/RMSE) now that all three
    dispatch to a different function per problem_type — profile_wrapper
    already sets the standard these three should match."""
    source = Path("src/dags/dag_factory.py").read_text()

    # The stale tabular-only phrasings must be gone...
    assert "Clean raw data: impute, drop bad cols, dedup." not in source
    assert "Engineer features: encode, Box-Cox, VIF, scale, split." not in source
    assert "Train all models and log R² + RMSE to MLflow." not in source

    # ...replaced by dispatch-aware language, mirroring profile_wrapper's own.
    assert "forecasting pipelines" in source
