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


def build_forecast_features_schema(target_col: str) -> DataFrameSchema:
    """Build a feature validation schema for a forecasting pipeline's feature matrix.

    Differs from build_features_schema in two ways: (1) the index must be a
    DatetimeIndex (observation timestamp), not a plain integer row index, since
    row order and spacing are meaningful; (2) coerce=False for strict column-type
    validation, vs coerce=True in the tabular schema. This stricter validation
    prevents a plain integer target column from silently coercing to float and
    masking type mismatches.

    Args:
        target_col: Target column name from pipeline.yaml (e.g. 'PJME_MW').

    Returns:
        DataFrameSchema that checks the target is a nullable float and the
        index is a non-null datetime; strict=False allows any additional
        lag/rolling/calendar predictor columns without listing them.
    """
    return DataFrameSchema(
        columns={target_col: Column(float, nullable=True)},
        index=Index("datetime64[ns]", nullable=False),
        strict=False,
        coerce=False,
    )
