"""Data ingestion stage: move files from landing to raw with versioning."""
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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


def _copy_and_hash(src: Path, dest_dir: Path, algorithm: str = "sha256") -> tuple[int, str]:
    """Copy src into dest_dir and hash it in a single read pass.

    Hashing via compute_file_hash() and then copying via shutil.copy2() would
    each read the whole file independently — for a large source (e.g.
    HCAHPS-Hospital.csv at ~141MB) that's twice the necessary read I/O. This
    streams the file once, hashing each chunk as it's written to the
    destination, then copies stat metadata (mtime, permissions) to match
    copy2's behavior.

    Args:
        src: Source file path.
        dest_dir: Destination directory (versioned raw run dir).
        algorithm: Hash algorithm.

    Returns:
        (file_size_bytes, hex_digest).
    """
    hash_obj = hashlib.new(algorithm)
    dest = dest_dir / src.name
    size = 0
    with open(src, "rb") as fsrc, open(dest, "wb") as fdst:
        for chunk in iter(lambda: fsrc.read(8192), b""):
            hash_obj.update(chunk)
            fdst.write(chunk)
            size += len(chunk)
    shutil.copystat(src, dest)
    return size, hash_obj.hexdigest()


def _ingest_file(src: Path, dest_dir: Path, fmt: str) -> dict[str, Any]:
    """Copy and hash a single file into dest_dir, returning its manifest entry.

    Args:
        src: Source file path.
        dest_dir: Destination directory (versioned raw run dir).
        fmt: Format label recorded in the manifest (e.g. "csv", "parquet").

    Returns:
        Manifest entry dict with size, hash, format, and timestamp.
    """
    file_size, file_hash = _copy_and_hash(src, dest_dir)
    return {
        "size_bytes": file_size,
        "hash_sha256": file_hash,
        "format": fmt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Maps file extension → handler. Register new formats here without touching ingest_files.
HANDLERS: dict[str, Callable[[Path, Path], dict[str, Any]]] = {
    ".csv": lambda src, dest_dir: _ingest_file(src, dest_dir, "csv"),
    ".parquet": lambda src, dest_dir: _ingest_file(src, dest_dir, "parquet"),
}


def ingest_files(
    landing_dir: str | Path = "data/landing",
    raw_dir: str | Path = "data/raw",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Ingest files from landing to raw directory with versioning.

    Dispatches each file to a format-specific handler based on extension.
    Currently supports: CSV. To add a new format, implement a handler with
    signature (src: Path, dest_dir: Path) -> dict and register it in HANDLERS.

    Args:
        landing_dir: Source directory with input files
        raw_dir: Target base directory for raw data
        run_id: Run identifier (defaults to current UTC date)

    Returns:
        Dictionary with run_id, file_count, manifest_path, and raw_dir

    Raises:
        FileNotFoundError: If landing directory doesn't exist
        ValueError: If no supported files found in landing directory
    """
    landing_path = Path(landing_dir)
    raw_path = Path(raw_dir)

    if not landing_path.exists():
        raise FileNotFoundError(f"Landing directory not found: {landing_dir}")

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    raw_run_dir = raw_path / run_id
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    supported_files = sorted(f for f in landing_path.iterdir() if f.suffix.lower() in HANDLERS)
    if not supported_files:
        supported = sorted(HANDLERS)
        raise ValueError(
            f"No supported files found in {landing_dir}. Supported formats: {supported}"
        )

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(landing_path),
        "files": {},
    }

    for file in supported_files:
        handler = HANDLERS[file.suffix.lower()]
        manifest["files"][file.name] = handler(file, raw_run_dir)

    manifest_path = raw_run_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    return {
        "run_id": run_id,
        "file_count": len(supported_files),
        "manifest_path": str(manifest_path),
        "raw_dir": str(raw_run_dir),
    }
