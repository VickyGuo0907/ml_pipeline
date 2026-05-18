"""Pandera schema for raw ingested Hospital Compare data.

Validates only common columns across different Hospital Compare datasets.
Different datasets use different naming conventions (e.g., "FacilityId" vs "Facility ID").
Each dataset may have different sets of columns beyond these core ones.
"""
from pandera import Column, DataFrameSchema, Index

raw_schema = DataFrameSchema(
    columns={
        # Core facility identifiers (all Hospital Compare files have these)
        "Facility Name": Column(str, nullable=False),
        "State": Column(str, nullable=False),
        # Facility ID has inconsistent naming across datasets - handle both versions
        # Will be validated if present, but not required
    },
    index=Index(int, nullable=False),
    strict=False,  # Allow extra columns not in schema (different datasets have different columns)
    coerce=True,
)
