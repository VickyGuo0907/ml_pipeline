"""Shared I/O helpers for pipeline stages: file reading/writing, path resolution, and manifests."""
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.yaml"


# ---------------------------------------------------------------------------
# File readers / writers
# ---------------------------------------------------------------------------

def read_csv(file_path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def read_parquet(file_path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(file_path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV."""
    df.to_csv(path, index=False)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet."""
    df.to_parquet(path, index=False)


# Maps file extension → reader. Add new formats here without touching callers.
READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": read_csv,
    ".parquet": read_parquet,
}

# Maps file extension → writer.
WRITERS: dict[str, Callable[[pd.DataFrame, Path], None]] = {
    ".csv": write_csv,
    ".parquet": write_parquet,
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_run_path(base_dir: str | Path, run_id: str) -> Path:
    """Return the versioned run directory: <base_dir>/<run_id>.

    Args:
        base_dir: Root data directory (e.g. raw_dir, interim_dir).
        run_id: Run identifier (e.g. '2026-06-30').

    Returns:
        Path to the run-specific subdirectory.
    """
    return Path(base_dir) / run_id


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Load manifest.yaml from a run directory.

    Args:
        run_dir: Versioned run directory containing manifest.yaml.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If manifest.yaml is absent from run_dir.
    """
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return yaml.safe_load(f) or {}


def write_manifest(run_dir: Path, data: dict[str, Any]) -> None:
    """Write manifest.yaml to a run directory.

    Args:
        run_dir: Output directory (created if absent).
        data: Manifest content to serialise.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.debug("Manifest written: %s", manifest_path)


def find_previous_run_id(base_dir: str | Path, current_run_id: str) -> str | None:
    """Find the most recent run directory strictly before current_run_id.

    Run directories are named so that lexicographic order matches recency
    (ISO dates like '2026-07-01' or Airflow logical dates both sort correctly
    this way). Non-directory entries are ignored.

    Args:
        base_dir: Directory containing one subdirectory per run_id (e.g. a features_dir).
        current_run_id: The run_id to find a predecessor for.

    Returns:
        The most recent run_id strictly before current_run_id, or None if
        base_dir doesn't exist or no earlier run directory is found.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    run_ids = sorted(
        p.name for p in base_path.iterdir()
        if p.is_dir() and p.name < current_run_id
    )
    return run_ids[-1] if run_ids else None
