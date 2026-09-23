"""Tests for the finance stage stubs - later plans in this pipeline's
sequence replace each of these bodies.

These tests exist to (a) pin the exact call signature every stub must keep
so dag_factory.py's dispatch never has to special-case argument shape, and
(b) give this plan something concrete to turn from RED to GREEN.
"""
import pytest

from src.finance.clean_finance import clean_finance_data
from src.finance.evaluate_finance import register_finance_models_to_mlflow
from src.finance.features_finance import engineer_finance_features
from src.finance.train_finance import train_finance_models


def test_clean_finance_data_not_implemented():
    with pytest.raises(NotImplementedError):
        clean_finance_data("raw", "interim", "2026-09-22", config_dir="config/m6_returns_risk")


def test_engineer_finance_features_not_implemented():
    with pytest.raises(NotImplementedError):
        engineer_finance_features("interim", "features", "2026-09-22", config_dir="config/m6_returns_risk")


def test_train_finance_models_not_implemented():
    with pytest.raises(NotImplementedError):
        train_finance_models("features", "2026-09-22", config_dir="config/m6_returns_risk")


def test_register_finance_models_not_implemented():
    with pytest.raises(NotImplementedError):
        register_finance_models_to_mlflow(config_dir="config/m6_returns_risk", run_id="2026-09-22")
