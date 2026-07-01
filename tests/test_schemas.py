"""Tests for data validation schemas."""
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.schemas.features import build_features_schema
from src.schemas.raw import raw_schema

# Use the same target column as pipeline.yaml for schema tests
features_schema = build_features_schema("Excess Readmission Ratio")


class TestRawSchema:
    """Tests for raw data schema validation."""

    def test_valid_raw_data(self):
        """Test that valid raw Hospital Compare data passes validation."""
        df = pd.DataFrame({
            "Facility ID": [1, 2, 3],
            "Facility Name": ["Hospital A", "Hospital B", "Hospital C"],
            "State": ["NY", "CA", "TX"],
            "Measure Name": ["Pneumonia", "Pneumonia", "Pneumonia"],
            "Start Date": ["2023-01-01", "2023-01-01", "2023-01-01"],
            "End Date": ["2023-12-31", "2023-12-31", "2023-12-31"],
            "Number of Discharges": ["150", "200", "100"],
            "Excess Readmission Ratio": [0.95, 1.05, 0.88],
            "Predicted Readmission Rate": [0.12, 0.15, 0.10],
            "Expected Readmission Rate": [0.13, 0.14, 0.11],
            "Number of Readmissions": ["18", "30", "10"],
            "Footnote": [None, None, None],
        })
        validated = raw_schema.validate(df)
        assert len(validated) == 3

    def test_raw_data_with_too_few_to_report(self):
        """Test that 'Too Few to Report' text is accepted in numeric columns."""
        df = pd.DataFrame({
            "Facility ID": [1, 2],
            "Facility Name": ["Hospital A", "Hospital B"],
            "State": ["NY", "CA"],
            "Measure Name": ["Pneumonia", "Pneumonia"],
            "Number of Discharges": ["150", "Too Few to Report"],  # Should accept text
            "Excess Readmission Ratio": [0.95, 1.05],
            "Number of Readmissions": ["18", "Too Few to Report"],  # Should accept text
        })
        # Should not raise - str columns can contain any text
        validated = raw_schema.validate(df)
        assert len(validated) == 2

    def test_raw_data_missing_required_column(self):
        """Test that missing required columns raise validation error."""
        df = pd.DataFrame({
            # "Facility Name" is required by the schema — omitting it must fail
            "State": ["NY", "CA"],
        })
        with pytest.raises(SchemaError):
            raw_schema.validate(df)

    def test_raw_data_null_in_required_column(self):
        """Test that null values in non-nullable columns raise validation error."""
        df = pd.DataFrame({
            "Facility Name": ["Hospital A", None],  # nullable=False
            "State": ["NY", "CA"],
        })
        with pytest.raises(SchemaError):
            raw_schema.validate(df)

    def test_raw_data_nullable_columns(self):
        """Test that nullable columns can have NaN values."""
        df = pd.DataFrame({
            "Facility ID": [1, 2],
            "Facility Name": ["Hospital A", "Hospital B"],
            "State": ["NY", "CA"],
            "Excess Readmission Ratio": [0.95, np.nan],  # Nullable
            "Footnote": [None, "Some text"],  # Nullable
        })
        validated = raw_schema.validate(df)
        assert len(validated) == 2


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
