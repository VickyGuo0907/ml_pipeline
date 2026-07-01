"""Pandera schema factory for feature matrix validation."""
from pandera import Column, DataFrameSchema, Index


def build_features_schema(target_col: str) -> DataFrameSchema:
    """Build a feature validation schema driven by the pipeline target column.

    Args:
        target_col: Target column name from pipeline.yaml (e.g. 'Excess Readmission Ratio').

    Returns:
        DataFrameSchema that checks the target is a nullable float; strict=False
        allows any additional predictor columns without listing them explicitly.
    """
    return DataFrameSchema(
        columns={target_col: Column(float, nullable=True)},
        index=Index(int, nullable=False),
        strict=False,
        coerce=True,
    )
