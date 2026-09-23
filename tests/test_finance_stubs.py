"""Tests for the finance stage stubs - later plans in this pipeline's
sequence replace each of these bodies.

These tests exist to (a) pin the exact call signature every stub must keep
so dag_factory.py's dispatch never has to special-case argument shape, and
(b) give this plan something concrete to turn from RED to GREEN.

train_finance_models is implemented for real as of this plan (see
tests/test_train_finance.py) - its stub test was removed here.
"""
import pytest

from src.finance.evaluate_finance import register_finance_models_to_mlflow


def test_register_finance_models_not_implemented():
    with pytest.raises(NotImplementedError):
        register_finance_models_to_mlflow(config_dir="config/m6_returns_risk", run_id="2026-09-22")
