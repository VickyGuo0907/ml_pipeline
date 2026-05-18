"""Data ingestion stage: move files from landing to raw with versioning."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity checking.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc)

    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def ingest_files(
    landing_dir: str | Path = "data/landing",
    raw_dir: str | Path = "data/raw",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Ingest CSV files from landing to raw directory with versioning.

    Moves all CSV files from landing directory to raw/<run_id>/ and creates
    manifest.yaml with file metadata (size, hash, timestamp).

    Args:
        landing_dir: Source directory with input files
        raw_dir: Target base directory for raw data
        run_id: Run identifier (defaults to ISO timestamp)

    Returns:
        Dictionary with run_id, file count, and manifest path

    Raises:
        FileNotFoundError: If landing directory doesn't exist
        ValueError: If no CSV files found in landing directory
    """
    landing_path = Path(landing_dir)
    raw_path = Path(raw_dir)

    if not landing_path.exists():
        raise FileNotFoundError(f"Landing directory not found: {landing_dir}")

    # Default run_id to ISO timestamp if not provided
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y-%m-%d")

    # Create versioned raw directory
    raw_run_dir = raw_path / run_id
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in landing directory
    csv_files = list(landing_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {landing_dir}")

    # Build manifest with file metadata
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "source_directory": str(landing_path),
        "files": {},
    }

    # Ingest each file
    for csv_file in csv_files:
        # Compute hash before moving
        file_hash = compute_file_hash(csv_file)
        file_size = csv_file.stat().st_size

        # Move file to raw directory
        target_path = raw_run_dir / csv_file.name
        shutil.copy2(csv_file, target_path)

        # Record in manifest
        manifest["files"][csv_file.name] = {
            "size_bytes": file_size,
            "hash_sha256": file_hash,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Write manifest.yaml
    manifest_path = raw_run_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    return {
        "run_id": run_id,
        "file_count": len(csv_files),
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_run_dir),
    }
