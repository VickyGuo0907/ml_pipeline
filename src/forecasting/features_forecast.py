"""Feature engineering stage for time-series forecasting pipelines:
lag/rolling/calendar/holiday features and the chronological train/test
split. Plan 2 implements this; this stub only pins the call signature so
src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def engineer_forecast_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Build lag/rolling/calendar/holiday features and split chronologically.

    Args:
        interim_dir: Directory containing cleaned interim data.
        features_dir: Output directory for feature matrices.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/pjm_load_forecast).

    Returns:
        Dictionary with feature matrix paths, shapes, and transform metadata.

    Raises:
        NotImplementedError: Always - implemented in Plan 2.
    """
    raise NotImplementedError("engineer_forecast_features is implemented in Plan 2")
