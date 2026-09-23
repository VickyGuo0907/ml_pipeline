"""Clean stage for finance returns/risk pipelines: parse, sort, and
gap-handle a per-asset monthly return panel. A later plan implements this;
this stub only pins the call signature so src/dags/dag_factory.py can
dispatch to it.
"""
from pathlib import Path
from typing import Any


def clean_finance_data(
    raw_dir: str | Path,
    interim_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Parse, sort, and gap-handle the raw per-asset monthly return panel.

    Args:
        raw_dir: Base directory containing raw data.
        interim_dir: Output directory for cleaned data.
        run_id: Run identifier to locate/version data.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).

    Returns:
        Dictionary with cleaning statistics.

    Raises:
        NotImplementedError: Always - implemented in a later plan.
    """
    raise NotImplementedError("clean_finance_data is implemented in a later plan")
