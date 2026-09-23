"""Tests for data validation schemas."""
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.schemas.features import build_features_schema, build_finance_features_schema, build_forecast_features_schema

# Use the same target column as pipeline.yaml for schema tests
features_schema = build_features_schema("Excess Readmission Ratio")


class TestFeaturesSchema:
    """Tests for feature matrix schema validation."""

    def test_valid_feature_matrix(self):
        """Test that valid feature matrix passes validation."""
        df = pd.DataFrame({
            "Excess Readmission Ratio": [0.95, 1.05, 0.88],
            "Facility Name_encoded": [0, 1, 2],
            "State_encoded": [0, 1, 2],
            "Measure Name_encoded": [0, 0, 0],
            "Facility ID": [1.0, 2.0, 3.0],
            "Number of Discharges": [150.0, 200.0, 100.0],
            "Predicted Readmission Rate": [0.12, 0.15, 0.10],
            "Expected Readmission Rate": [0.13, 0.14, 0.11],
            "Number of Readmissions": [18.0, 30.0, 10.0],
        })
        validated = features_schema.validate(df)
        assert len(validated) == 3

    def test_feature_matrix_missing_target(self):
        """Test that missing target column raises error."""
        df = pd.DataFrame({
            "State_encoded": [0, 1],
            "Facility Name_encoded": [1, 2],
            # Missing Excess Readmission Ratio (target, required)
        })
        with pytest.raises(SchemaError):
            features_schema.validate(df)

    def test_feature_matrix_with_nullable_columns(self):
        """Test that nullable encoded columns can have NaN values."""
        df = pd.DataFrame({
            "Excess Readmission Ratio": [0.95, 1.05],
            "Facility Name_encoded": [0, np.nan],  # Nullable
            "State_encoded": [0, 1],
            "Measure Name_encoded": [0, 0],
            "Facility ID": [1.0, 2.0],
            "Number of Discharges": [150.0, 200.0],
            "Predicted Readmission Rate": [0.12, 0.15],
            "Expected Readmission Rate": [0.13, 0.14],
            "Number of Readmissions": [18.0, np.nan],  # Nullable
        })
        validated = features_schema.validate(df)
        assert len(validated) == 2


class TestForecastFeaturesSchema:
    """Tests for the forecasting feature matrix schema (DatetimeIndex, not int index)."""

    def test_valid_forecast_feature_matrix(self):
        """A DatetimeIndex-indexed matrix with a numeric target passes validation."""
        schema = build_forecast_features_schema("PJME_MW")
        df = pd.DataFrame(
            {
                "PJME_MW": [30000.0, 31000.0, 29500.0],
                "lag_1h": [29800.0, 30000.0, 31000.0],
                "hour": [0, 1, 2],
            },
            index=pd.DatetimeIndex(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"], name="Datetime",
            ),
        )
        validated = schema.validate(df)
        assert len(validated) == 3

    def test_int_indexed_matrix_fails_forecast_schema(self):
        """A plain integer index is rejected - forecasting matrices must carry a DatetimeIndex."""
        schema = build_forecast_features_schema("PJME_MW")
        df = pd.DataFrame({"PJME_MW": [30000.0, 31000.0]})
        with pytest.raises(SchemaError):
            schema.validate(df)


class TestFinanceFeaturesSchema:
    """Tests for the finance feature matrix schema (long-format panel, plain
    integer index — unlike the forecasting schema's DatetimeIndex, since many
    rows share the same Date across different assets)."""

    def test_valid_finance_feature_matrix(self):
        """A plain-integer-indexed long-format matrix with a numeric target passes validation."""
        schema = build_finance_features_schema("log_return")
        df = pd.DataFrame({
            "log_return": [0.01, -0.02, 0.03],
            "Ticker": [0, 1, 0],
            "lag_1_return": [0.02, -0.01, 0.01],
        })
        validated = schema.validate(df)
        assert len(validated) == 3

    def test_finance_feature_matrix_missing_target(self):
        """Missing target column raises SchemaError."""
        schema = build_finance_features_schema("log_return")
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})
        with pytest.raises(SchemaError):
            schema.validate(df)
