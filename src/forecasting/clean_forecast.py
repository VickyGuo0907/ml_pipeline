"""Clean stage for time-series forecasting pipelines: parse, sort, and
gap-handle a continuous hourly series. Plan 2 implements this; this stub
only pins the call signature so src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def clean_forecast_data(
    raw_dir: str | Path,
    interim_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Parse, sort, and gap-handle the raw hourly series.

    Args:
        raw_dir: Base directory containing raw data.
        interim_dir: Output directory for cleaned data.
        run_id: Run identifier to locate/version data.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).

    Returns:
        Dictionary with cleaning statistics.

    Raises:
        NotImplementedError: Always - implemented in Plan 2.
    """
    raise NotImplementedError("clean_forecast_data is implemented in Plan 2")
