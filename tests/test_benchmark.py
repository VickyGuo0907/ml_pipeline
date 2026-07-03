"""Tests for benchmark dataset management and bootstrap statistics."""
import numpy as np
import pandas as pd
import pytest
import yaml

from src.benchmark import (
    bootstrap_rmse_ci,
    create_benchmark_snapshot,
    load_current_benchmark,
)


class TestBootstrapRmseCi:
    """Tests for the bootstrap confidence interval calculation."""

    def test_perfect_predictions_give_zero_width_ci(self):
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_rmse_ci(y_true, y_pred, n_iterations=200)
        assert lower == pytest.approx(0.0, abs=1e-9)
        assert upper == pytest.approx(0.0, abs=1e-9)

    def test_worse_predictions_give_higher_ci(self):
        y_true = pd.Series(np.arange(100, dtype=float))
        good_pred = np.arange(100, dtype=float) + np.random.default_rng(1).normal(0, 0.1, 100)
        bad_pred = np.arange(100, dtype=float) + np.random.default_rng(1).normal(0, 5.0, 100)

        good_lower, good_upper = bootstrap_rmse_ci(y_true, good_pred, n_iterations=500)
        bad_lower, bad_upper = bootstrap_rmse_ci(y_true, bad_pred, n_iterations=500)

        assert bad_lower > good_upper  # non-overlapping, bad is clearly worse

    def test_is_reproducible_with_fixed_random_state(self):
        y_true = pd.Series(np.arange(50, dtype=float))
        y_pred = np.arange(50, dtype=float) + 1.0

        result_1 = bootstrap_rmse_ci(y_true, y_pred, n_iterations=300, random_state=7)
        result_2 = bootstrap_rmse_ci(y_true, y_pred, n_iterations=300, random_state=7)
        assert result_1 == result_2

    def test_lower_bound_never_exceeds_upper_bound(self):
        rng = np.random.default_rng(3)
        y_true = pd.Series(rng.random(80))
        y_pred = rng.random(80)
        lower, upper = bootstrap_rmse_ci(y_true, y_pred, n_iterations=300)
        assert lower <= upper


class TestBenchmarkSnapshot:
    """Tests for creating and loading versioned benchmark snapshots."""

    def test_create_benchmark_snapshot_writes_parquet_and_manifest(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-01"
        run_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [0.1, 0.2, 0.3]})
        df.to_parquet(run_dir / "train.parquet")

        benchmark_dir = tmp_path / "benchmark"
        result = create_benchmark_snapshot(features_dir, "2026-07-01", benchmark_dir)

        assert result["version"] == "2026-07-01"
        assert result["row_count"] == 3
        assert (benchmark_dir / "2026-07-01" / "benchmark.parquet").exists()
        assert (benchmark_dir / "2026-07-01" / "manifest.yaml").exists()

        manifest = yaml.safe_load((benchmark_dir / "2026-07-01" / "manifest.yaml").read_text())
        assert manifest["row_count"] == 3
        assert manifest["source_run_id"] == "2026-07-01"
        assert "hash_sha256" in manifest

    def test_create_benchmark_snapshot_updates_current_pointer(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-01"
        run_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1.0], "target": [0.1]}).to_parquet(run_dir / "train.parquet")

        benchmark_dir = tmp_path / "benchmark"
        create_benchmark_snapshot(features_dir, "2026-07-01", benchmark_dir)

        pointer = yaml.safe_load((benchmark_dir / "current.yaml").read_text())
        assert pointer["version"] == "2026-07-01"

    def test_create_benchmark_snapshot_raises_when_train_parquet_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_benchmark_snapshot(tmp_path / "features", "2026-07-01", tmp_path / "benchmark")

    def test_load_current_benchmark_returns_none_when_no_benchmark_exists(self, tmp_path):
        assert load_current_benchmark(tmp_path / "benchmark") is None

    def test_load_current_benchmark_round_trips_the_snapshot(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-01"
        run_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": [1.0, 2.0], "target": [0.1, 0.2]})
        df.to_parquet(run_dir / "train.parquet")

        benchmark_dir = tmp_path / "benchmark"
        create_benchmark_snapshot(features_dir, "2026-07-01", benchmark_dir)

        loaded = load_current_benchmark(benchmark_dir)
        assert loaded is not None
        assert len(loaded) == 2
        assert list(loaded.columns) == ["a", "target"]
