"""Tests for data validation schemas."""
import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.schemas.features import build_features_schema

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
