"""Pandera schema for feature matrix used in model training.

Different datasets produce different features. This schema validates only
the common structure and presence of the target column.
"""
from pandera import Column, DataFrameSchema, Index

features_schema = DataFrameSchema(
    columns={
        # Target variable - must be numeric, nullable to handle missing values
        "Excess Readmission Ratio": Column(float, nullable=True),
    },
    index=Index(int, nullable=False),
    strict=False,  # Allow any features beyond the target
    coerce=True,
)
