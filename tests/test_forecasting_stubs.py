"""Tests for the forecasting stage stubs - Plans 2-4 replace each of these bodies.

These tests exist to (a) pin the exact call signature every stub must keep so
dag_factory.py's dispatch never has to special-case argument shape, and (b)
give plan 1 something concrete to turn from RED to GREEN.

Note: test_engineer_forecast_features_not_implemented was removed when Plan 2
implemented engineer_forecast_features. test_train_forecast_models_not_implemented
was removed the same way when Plan 3 implemented train_forecast_models — see
tests/test_train_forecast.py for its real behavior tests.
"""
import pytest

from src.forecasting.evaluate_forecast import register_forecast_models_to_mlflow


def test_register_forecast_models_not_implemented():
    with pytest.raises(NotImplementedError):
        register_forecast_models_to_mlflow(config_dir="config/pjm_load_forecast", run_id="2026-09-16")
