"""Feature engineering stage for finance returns/risk pipelines: the
cross-sectional GBM's small lag/volatility feature set, and the chronological
per-asset train/test split. A later plan implements this; this stub only
pins the call signature so src/dags/dag_factory.py can dispatch to it.
"""
from pathlib import Path
from typing import Any


def engineer_finance_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Build the cross-sectional GBM's feature set and split each asset chronologically.

    Args:
        interim_dir: Directory containing cleaned interim data.
        features_dir: Output directory for feature matrices.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).

    Returns:
        Dictionary with feature matrix paths, shapes, and transform metadata.
        The feature matrix must be entirely numeric (Ticker label-encoded as an int,
        Date as a numeric/period column, no raw strings or datetimes) because
        dag_factory.py's validate_features_wrapper enforces an all-numeric guard
        on every pipeline's feature matrix before any pandera schema runs.

    Raises:
        NotImplementedError: Always - implemented in a later plan.
    """
    raise NotImplementedError("engineer_finance_features is implemented in a later plan")
