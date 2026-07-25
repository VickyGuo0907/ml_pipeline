"""Fixed benchmark dataset management and statistical model comparison.

A benchmark is a frozen, manually-refreshed feature matrix used to compare a
newly trained model against whatever is currently in MLflow Production for
the same model name. See
docs/superpowers/specs/2026-07-03-champion-challenger-regression-check-design.md
for the full design rationale — in particular, why this exists instead of
comparing test_rmse across runs directly (the train/test split is re-drawn
every run, so it isn't a stable benchmark).
"""
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from src.ingest import compute_file_hash
from src.utils.io import write_manifest

logger = logging.getLogger(__name__)

CURRENT_POINTER_FILENAME = "current.yaml"
BENCHMARK_FILENAME = "benchmark.parquet"


def create_benchmark_snapshot(
    features_dir: str | Path,
    run_id: str,
    benchmark_dir: str | Path,
) -> dict[str, Any]:
    """Snapshot this run's train.parquet as the new benchmark version.

    Copies <features_dir>/<run_id>/train.parquet into
    <benchmark_dir>/<run_id>/benchmark.parquet, writes its manifest (checksum,
    source run_id, row count), and updates current.yaml to point at this
    version. train.parquet (not test.parquet) is used because a benchmark is
    a static evaluation set reused across many future runs, not itself a
    train/test split.

    Args:
        features_dir: Pipeline features directory (e.g. config.directories.features).
        run_id: This run's identifier — becomes the new benchmark version.
        benchmark_dir: Pipeline benchmark directory (e.g. config.directories.benchmark).

    Returns:
        Dict with version, row_count, and benchmark_path.

    Raises:
        FileNotFoundError: If this run's train.parquet doesn't exist.
    """
    train_path = Path(features_dir) / run_id / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    df = pd.read_parquet(train_path)

    version_dir = Path(benchmark_dir) / run_id
    version_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = version_dir / BENCHMARK_FILENAME
    df.to_parquet(benchmark_path, index=False)

    write_manifest(version_dir, {
        "version": run_id,
        "source_run_id": run_id,
        "source_path": str(train_path),
        "row_count": len(df),
        "hash_sha256": compute_file_hash(benchmark_path),
    })

    current_path = Path(benchmark_dir) / CURRENT_POINTER_FILENAME
    with open(current_path, "w") as f:
        yaml.dump({"version": run_id}, f)

    logger.info("Benchmark refreshed: version=%s rows=%d", run_id, len(df))
    return {"version": run_id, "row_count": len(df), "benchmark_path": str(benchmark_path)}


def load_current_benchmark(benchmark_dir: str | Path) -> pd.DataFrame | None:
    """Load the benchmark set currently pointed to by current.yaml.

    Args:
        benchmark_dir: Pipeline benchmark directory.

    Returns:
        The benchmark DataFrame, or None if no benchmark has been created yet.
    """
    current_path = Path(benchmark_dir) / CURRENT_POINTER_FILENAME
    if not current_path.exists():
        return None

    with open(current_path) as f:
        pointer = yaml.safe_load(f) or {}
    version = pointer.get("version")
    if not version:
        return None

    benchmark_path = Path(benchmark_dir) / version / BENCHMARK_FILENAME
    if not benchmark_path.exists():
        return None

    return pd.read_parquet(benchmark_path)


def bootstrap_metric_ci(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for any (y_true, y_pred) -> scalar metric.

    Generic resampling helper shared by the RMSE-specific champion/challenger
    check (bootstrap_rmse_ci below) and per-model test-metric CIs (RMSE and R²)
    computed in src/train.py.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions, same length and order as y_true.
        metric_fn: Scores one resample, e.g. sklearn's r2_score or an RMSE lambda.
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for a 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        (lower, upper) bound of the metric's confidence interval.
    """
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)

    values = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        values[i] = metric_fn(y_true_arr[idx], y_pred_arr[idx])

    alpha = 1 - confidence
    lower = float(np.percentile(values, 100 * (alpha / 2)))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_rmse_ci(
    y_true: pd.Series,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for RMSE via resampling.

    Thin wrapper over bootstrap_metric_ci, kept as its own function since it's
    the one used by the champion/challenger regression check in evaluate.py.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions, same length and order as y_true.
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for a 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        (lower, upper) bound of the RMSE confidence interval.
    """
    def _rmse(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    return bootstrap_metric_ci(y_true, y_pred, _rmse, n_iterations, confidence, random_state)
