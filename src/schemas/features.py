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

    The index dtype ("datetime64[ns]") is tz-naive and will reject a tz-aware
    index. This is intentional, not a gap: engineer_forecast_features always
    builds a naive pd.date_range/DatetimeIndex, and PJM's source data has no
    timezone information to begin with, so a tz-aware index never legitimately
    reaches this schema.

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


def build_finance_features_schema(target_col: str) -> DataFrameSchema:
    """Build a feature validation schema for a finance pipeline's feature matrix.

    Unlike build_forecast_features_schema's DatetimeIndex requirement, a
    finance feature matrix is long-format (one row per (Date, Ticker) pair —
    many rows share the same Date across different assets), so a plain
    integer row index is the correct shape here, same as the tabular schema.
    A distinct function (rather than reusing build_features_schema directly)
    keeps room to add finance-specific column checks (e.g. a required Ticker
    column) later without touching the tabular schema's contract.

    Args:
        target_col: Target column name from pipeline.yaml (e.g. 'log_return').

    Returns:
        DataFrameSchema that checks the target is a nullable float; strict=False
        allows any additional predictor columns without listing them explicitly.
        Note: all columns, including any additional columns like Ticker and Date,
        must already be numeric (label-encoded integers for Ticker, numeric
        representations for Date) because dag_factory.py's validate_features_wrapper
        enforces an all-numeric guard on the feature matrix before any pandera
        schema runs, and rejects any non-numeric columns first.
    """
    return DataFrameSchema(
        columns={target_col: Column(float, nullable=True)},
        index=Index(int, nullable=False),
        strict=False,
        coerce=True,
    )
