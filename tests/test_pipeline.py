"""Tests for ML pipeline stages."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.clean import clean_raw_data
from src.features import engineer_features
from src.ingest import compute_file_hash, ingest_files
from src.profile import profile_raw_files
from src.validate import validate_raw_files


class TestIngest:
    """Tests for data ingestion."""

    def test_ingest_files_creates_versioned_directory(self):
        """Test that ingest creates run_id-versioned directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            raw_dir = Path(tmpdir) / "raw"
            landing_dir.mkdir()

            # Create test CSV
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            csv_path = landing_dir / "test.csv"
            df.to_csv(csv_path, index=False)

            result = ingest_files(
                landing_dir=landing_dir,
                raw_dir=raw_dir,
                run_id="2026-05-18",
            )

            assert result["run_id"] == "2026-05-18"
            assert result["file_count"] == 1
            assert (raw_dir / "2026-05-18" / "test.csv").exists()
            assert (raw_dir / "2026-05-18" / "manifest.yaml").exists()

    def test_ingest_manifest_contains_file_metadata(self):
        """Test that manifest includes checksums and timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            raw_dir = Path(tmpdir) / "raw"
            landing_dir.mkdir()

            df = pd.DataFrame({"col1": [1, 2, 3]})
            csv_path = landing_dir / "test.csv"
            df.to_csv(csv_path, index=False)

            result = ingest_files(landing_dir, raw_dir, "2026-05-18")
            manifest_path = Path(result["manifest_path"])

            import yaml
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)

            assert "test.csv" in manifest["files"]
            assert "hash_sha256" in manifest["files"]["test.csv"]
            assert "size_bytes" in manifest["files"]["test.csv"]

    def test_ingest_raises_on_missing_landing_dir(self):
        """Test that ingest raises if landing directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                ingest_files(
                    landing_dir=Path(tmpdir) / "nonexistent",
                    raw_dir=Path(tmpdir) / "raw",
                )

    def test_ingest_raises_on_no_csv_files(self):
        """Test that ingest raises if no CSV files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            landing_dir.mkdir()

            with pytest.raises(ValueError, match="No CSV files"):
                ingest_files(landing_dir, Path(tmpdir) / "raw")

    def test_compute_file_hash(self):
        """Test file hashing for integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")

            hash1 = compute_file_hash(file_path)
            hash2 = compute_file_hash(file_path)

            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex is 64 chars


HEALTHCARE_CONFIG = "config/healthcare"


class TestClean:
    """Tests for data cleaning."""

    def test_clean_removes_high_missing_columns(self):
        """Test that clean removes columns with >50% missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interim_dir = Path(tmpdir) / "interim"
            raw_dir = Path(tmpdir) / "raw"
            interim_dir.mkdir()
            raw_dir.mkdir()

            run_id = "2026-05-18"

            df = pd.DataFrame({
                "col1": [1, 2, 3, 4, 5],
                "col2": [None, None, None, 4, 5],  # 60% missing
            })
            csv_path = raw_dir / run_id
            csv_path.mkdir(parents=True)
            df.to_csv(csv_path / "test.csv", index=False)

            import yaml
            manifest = {"files": {"test.csv": {}}}
            with open(csv_path / "manifest.yaml", "w") as f:
                yaml.dump(manifest, f)

            result = clean_raw_data(raw_dir, interim_dir, run_id, config_dir=HEALTHCARE_CONFIG)

            assert "test.csv" in result["files"]
            assert result["files"]["test.csv"]["cols_removed"] > 0

    def test_clean_removes_duplicates(self):
        """Test that clean removes duplicate rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interim_dir = Path(tmpdir) / "interim"
            raw_dir = Path(tmpdir) / "raw"
            interim_dir.mkdir()
            raw_dir.mkdir()

            run_id = "2026-05-18"

            df = pd.DataFrame({
                "col1": [1, 1, 2, 3],
                "col2": ["a", "a", "b", "c"],
            })
            csv_path = raw_dir / run_id
            csv_path.mkdir(parents=True)
            df.to_csv(csv_path / "test.csv", index=False)

            import yaml
            manifest = {"files": {"test.csv": {}}}
            with open(csv_path / "manifest.yaml", "w") as f:
                yaml.dump(manifest, f)

            result = clean_raw_data(raw_dir, interim_dir, run_id, config_dir=HEALTHCARE_CONFIG)

            assert result["files"]["test.csv"]["rows_removed"] > 0


class TestFeatures:
    """Tests for feature engineering."""

    def test_engineer_features_creates_train_test_split(self):
        """Test that features are split into train/test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interim_dir = Path(tmpdir) / "interim"
            features_dir = Path(tmpdir) / "features"
            config_dir = Path(tmpdir) / "config"

            interim_dir.mkdir()
            features_dir.mkdir()
            config_dir.mkdir()

            run_id = "2026-05-18"

            # Create cleaned data
            df = pd.DataFrame({
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": ["a", "b", "a", "b", "a"],
                "ExcessReadmissionRatio": [0.9, 1.0, 0.8, 1.1, 0.85],
            })
            csv_path = interim_dir / run_id
            csv_path.mkdir(parents=True)
            df.to_csv(csv_path / "test.csv", index=False)

            # Create manifest
            import yaml
            manifest = {"files": {"test.csv": {}}}
            with open(csv_path / "manifest.yaml", "w") as f:
                yaml.dump(manifest, f)

            # Create config files
            pipeline_config = """
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: ExcessReadmissionRatio
  type: continuous
problem_type: regression
train_test_split: 0.8
random_state: 42
"""
            features_config = """
encoding: {}
steps: []
nzv_threshold: 0.95
drop_columns: []
scale: true
"""
            models_config = """
models:
  - name: linear_baseline
    type: linear
    hyperparameters: {}
random_state: 42
train_test_split: 0.8
"""
            (config_dir / "pipeline.yaml").write_text(pipeline_config)
            (config_dir / "features.yaml").write_text(features_config)
            (config_dir / "models.yaml").write_text(models_config)

            result = engineer_features(interim_dir, features_dir, run_id, config_dir)

            assert result["train_shape"][0] > 0
            assert result["test_shape"][0] > 0
            assert result["train_shape"][0] + result["test_shape"][0] == len(df)
            assert (features_dir / run_id / "train.parquet").exists()
            assert (features_dir / run_id / "test.parquet").exists()


class TestIntegration:
    """End-to-end pipeline integration tests."""

    def test_ingest_to_clean_workflow(self):
        """Test workflow from ingest through clean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            raw_dir = Path(tmpdir) / "raw"
            interim_dir = Path(tmpdir) / "interim"

            landing_dir.mkdir()

            df = pd.DataFrame({
                "FacilityId": [1, 2, 3],
                "State": ["NY", "CA", "TX"],
                "ExcessReadmissionRatio": [0.95, 1.05, 0.88],
            })
            df.to_csv(landing_dir / "hospital_data.csv", index=False)

            run_id = "2026-05-18"

            ingest_result = ingest_files(landing_dir, raw_dir, run_id)
            assert ingest_result["file_count"] == 1

            clean_result = clean_raw_data(raw_dir, interim_dir, run_id, config_dir=HEALTHCARE_CONFIG)
            assert "hospital_data.csv" in clean_result["files"]
