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

    def test_ingest_raises_on_no_supported_files(self):
        """Test that ingest raises if no supported files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            landing_dir.mkdir()

            with pytest.raises(ValueError, match="No supported files"):
                ingest_files(landing_dir, Path(tmpdir) / "raw")

    def test_ingest_parquet_files(self):
        """Test that ingest handles Parquet files and records correct format in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            raw_dir = Path(tmpdir) / "raw"
            landing_dir.mkdir()

            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            df.to_parquet(landing_dir / "test.parquet", index=False)

            result = ingest_files(landing_dir=landing_dir, raw_dir=raw_dir, run_id="2026-05-18")

            assert result["file_count"] == 1
            assert (raw_dir / "2026-05-18" / "test.parquet").exists()

            import yaml
            with open(result["manifest_path"]) as f:
                manifest = yaml.safe_load(f)

            assert "test.parquet" in manifest["files"]
            assert manifest["files"]["test.parquet"]["format"] == "parquet"
            assert "hash_sha256" in manifest["files"]["test.parquet"]

    def test_ingest_mixed_csv_and_parquet(self):
        """Test that ingest handles a landing dir with both CSV and Parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            landing_dir = Path(tmpdir) / "landing"
            raw_dir = Path(tmpdir) / "raw"
            landing_dir.mkdir()

            df = pd.DataFrame({"col1": [1, 2]})
            df.to_csv(landing_dir / "data.csv", index=False)
            df.to_parquet(landing_dir / "data.parquet", index=False)

            result = ingest_files(landing_dir=landing_dir, raw_dir=raw_dir, run_id="2026-05-18")

            assert result["file_count"] == 2
            assert (raw_dir / "2026-05-18" / "data.csv").exists()
            assert (raw_dir / "2026-05-18" / "data.parquet").exists()

    def test_compute_file_hash(self):
        """Test file hashing for integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")

            hash1 = compute_file_hash(file_path)
            hash2 = compute_file_hash(file_path)

            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex is 64 chars


HEALTHCARE_CONFIG = "config/biomedical_clinical"

_PROFILE_PIPELINE_YAML = """\
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: score
  type: continuous
problem_type: regression
train_test_split: 0.8
random_state: 42
profiling:
  minimal: true
"""


def _make_profile_config(tmpdir: Path) -> Path:
    """Write a minimal pipeline.yaml for profile tests and return the config dir."""
    config_dir = tmpdir / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(_PROFILE_PIPELINE_YAML)
    return config_dir


class TestProfile:
    """Tests for the data profiling stage (Stage 3)."""

    def test_profile_creates_html_report_per_file(self):
        """One HTML report is generated for each file listed in the manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            raw_dir = _make_raw_dir(tmp, "2026-06-30", {"data.csv": df})
            reports_dir = tmp / "reports"
            config_dir = _make_profile_config(tmp)

            result = profile_raw_files(raw_dir, "2026-06-30", reports_dir=reports_dir, config_dir=config_dir)

            assert "data.csv" in result["reports"]
            report_path = Path(result["reports"]["data.csv"]["report_path"])
            assert report_path.exists()
            assert report_path.suffix == ".html"

    def test_profile_report_name_includes_run_id_and_stem(self):
        """Report filename is <run_id>_<file_stem>_profile.html."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            df = pd.DataFrame({"x": [1, 2]})
            raw_dir = _make_raw_dir(tmp, "2026-06-30", {"my_data.csv": df})
            reports_dir = tmp / "reports"
            config_dir = _make_profile_config(tmp)

            result = profile_raw_files(raw_dir, "2026-06-30", reports_dir=reports_dir, config_dir=config_dir)

            report_path = Path(result["reports"]["my_data.csv"]["report_path"])
            assert report_path.name == "2026-06-30_my_data_profile.html"

    def test_profile_returns_row_and_column_counts(self):
        """Result dict contains correct row and column counts for each file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
            raw_dir = _make_raw_dir(tmp, "2026-06-30", {"sample.csv": df})
            reports_dir = tmp / "reports"
            config_dir = _make_profile_config(tmp)

            result = profile_raw_files(raw_dir, "2026-06-30", reports_dir=reports_dir, config_dir=config_dir)

            assert result["reports"]["sample.csv"]["rows"] == 3
            assert result["reports"]["sample.csv"]["columns"] == 3

    def test_profile_handles_multiple_files(self):
        """A separate HTML report is generated for each file in the manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            df1 = pd.DataFrame({"x": [1, 2]})
            df2 = pd.DataFrame({"y": [3, 4]})
            raw_dir = _make_raw_dir(
                tmp, "2026-06-30", {"file_a.csv": df1, "file_b.csv": df2}
            )
            reports_dir = tmp / "reports"
            config_dir = _make_profile_config(tmp)

            result = profile_raw_files(raw_dir, "2026-06-30", reports_dir=reports_dir, config_dir=config_dir)

            assert "file_a.csv" in result["reports"]
            assert "file_b.csv" in result["reports"]
            assert Path(result["reports"]["file_a.csv"]["report_path"]).exists()
            assert Path(result["reports"]["file_b.csv"]["report_path"]).exists()

    def test_profile_raises_on_missing_manifest(self):
        """Raises FileNotFoundError when manifest.yaml is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = _make_profile_config(tmp)
            with pytest.raises(FileNotFoundError, match="Manifest not found"):
                profile_raw_files(Path(tmpdir) / "raw", "2026-06-30", reports_dir=tmpdir, config_dir=config_dir)

_VALIDATE_PIPELINE_YAML = """\
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: score
  type: continuous
problem_type: regression
train_test_split: 0.8
random_state: 42
validation:
  required_columns:
    - name
    - state
  numeric_bounds:
    score:
      min: 0.0
      max: 100.0
  min_rows: 1
"""

_VALIDATE_CLEANING_YAML = """\
steps: []
impute_strategy: median
drop_column_patterns: []
"""

# pipeline.yaml variant that declares sentinel strings under validation (their correct home)
_VALIDATE_PIPELINE_YAML_WITH_SENTINELS = """\
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: score
  type: continuous
problem_type: regression
train_test_split: 0.8
random_state: 42
validation:
  sentinel_values:
    - "Not Available"
    - "Too Few to Report"
  required_columns:
    - name
    - state
  numeric_bounds:
    score:
      min: 0.0
      max: 100.0
  min_rows: 1
"""


def _make_raw_dir(tmpdir: Path, run_id: str, files: dict[str, pd.DataFrame]) -> Path:
    """Write DataFrames to a versioned raw directory with a manifest."""
    raw_run = tmpdir / "raw" / run_id
    raw_run.mkdir(parents=True)
    manifest: dict = {"files": {}}
    for fname, df in files.items():
        if fname.endswith(".parquet"):
            df.to_parquet(raw_run / fname, index=False)
        else:
            df.to_csv(raw_run / fname, index=False)
        manifest["files"][fname] = {}
    import yaml
    with open(raw_run / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    return tmpdir / "raw"


class TestValidate:
    """Tests for raw data validation (Stage 2)."""

    def _write_config(self, config_dir: Path, pipeline_yaml: str = _VALIDATE_PIPELINE_YAML) -> None:
        """Write pipeline.yaml and a minimal cleaning.yaml to config_dir."""
        (config_dir / "pipeline.yaml").write_text(pipeline_yaml)
        (config_dir / "cleaning.yaml").write_text(_VALIDATE_CLEANING_YAML)

    def test_validate_passes_valid_data(self):
        """Valid data with required columns and in-range values passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            df = pd.DataFrame({"name": ["A", "B"], "state": ["NY", "CA"], "score": [50.0, 75.0]})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"data.csv": df})

            result = validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

            assert "data.csv" in result["validated_files"]
            assert result["validated_files"]["data.csv"]["status"] == "passed"
            assert result["failed_files"] == []

    def test_validate_fails_missing_required_column(self):
        """File missing a required column raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            df = pd.DataFrame({"name": ["A", "B"]})  # missing 'state'
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"data.csv": df})

            with pytest.raises(RuntimeError, match="Validation failed"):
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

    def test_validate_fails_out_of_range_value(self):
        """Numeric value outside declared bounds raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            df = pd.DataFrame({"name": ["A"], "state": ["NY"], "score": [999.0]})  # score > 100
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"data.csv": df})

            with pytest.raises(RuntimeError, match="Validation failed"):
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

    def test_validate_fails_empty_file(self):
        """File with zero rows raises RuntimeError (below min_rows=1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            df = pd.DataFrame({"name": pd.Series([], dtype=str), "state": pd.Series([], dtype=str)})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"data.csv": df})

            with pytest.raises(RuntimeError, match="Validation failed"):
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

    def test_validate_collects_errors_across_all_files(self):
        """All files are checked before raising — multi-file failure report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            good = pd.DataFrame({"name": ["A"], "state": ["NY"], "score": [50.0]})
            bad1 = pd.DataFrame({"name": ["B"]})           # missing 'state'
            bad2 = pd.DataFrame({"name": ["C"], "state": ["TX"], "score": [200.0]})  # out of range
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {
                "good.csv": good, "bad1.csv": bad1, "bad2.csv": bad2,
            })

            with pytest.raises(RuntimeError) as exc_info:
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

            # Both bad files reported, good file not mentioned in error
            error_msg = str(exc_info.value)
            assert "bad1.csv" in error_msg
            assert "bad2.csv" in error_msg
            assert "2 file(s)" in error_msg

    def test_validate_numeric_bounds_optional_when_column_absent(self):
        """Numeric bounds column absent from file does not cause failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            # score is in numeric_bounds but not required — file without it should pass
            df = pd.DataFrame({"name": ["A", "B"], "state": ["NY", "CA"]})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"data.csv": df})

            result = validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)
            assert result["failed_files"] == []

    def test_validate_passes_cms_sentinel_strings_in_bounds_column(self):
        """Sentinel strings declared in pipeline.yaml are treated as NaN, not invalid values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir, pipeline_yaml=_VALIDATE_PIPELINE_YAML_WITH_SENTINELS)

            df = pd.DataFrame({
                "name": ["A", "B", "C"],
                "state": ["NY", "CA", "TX"],
                "score": ["50.0", "Not Available", "Too Few to Report"],
            })
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"data.csv": df})

            result = validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)
            assert result["failed_files"] == []


# Pipeline YAML with per_file_schemas: readmissions file requires "measure_col", HCAHPS requires "survey_col"
_PER_FILE_PIPELINE_YAML = """\
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: score
  type: continuous
problem_type: regression
train_test_split: 0.8
random_state: 42
validation:
  required_columns:
    - "shared_col"
  min_rows: 1
  per_file_schemas:
    - file_pattern: "readmissions"
      required_columns:
        - "shared_col"
        - "measure_col"
      numeric_bounds:
        measure_col:
          min: 0.0
          max: 10.0
      min_rows: 2
    - file_pattern: "hcahps"
      required_columns:
        - "shared_col"
        - "survey_col"
      min_rows: 1
"""


class TestPerFileSchema:
    """Tests for per-file schema enforcement in Stage 2."""

    def _write_config(self, config_dir: Path) -> None:
        """Write pipeline.yaml with per_file_schemas and a minimal cleaning.yaml."""
        (config_dir / "pipeline.yaml").write_text(_PER_FILE_PIPELINE_YAML)
        (config_dir / "cleaning.yaml").write_text(_VALIDATE_CLEANING_YAML)

    def test_per_file_schema_passes_matching_file_with_correct_columns(self):
        """File matching a per-file pattern passes when all per-file required columns are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            df = pd.DataFrame({
                "shared_col": ["A", "B"],
                "measure_col": [1.0, 2.0],
            })
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"readmissions_2024.csv": df})

            result = validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)
            assert result["failed_files"] == []

    def test_per_file_schema_fails_when_per_file_column_missing(self):
        """File matching a per-file pattern fails when a per-file required column is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            # 'measure_col' is required by the 'readmissions' per-file schema but absent here
            df = pd.DataFrame({"shared_col": ["A", "B"]})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"readmissions_2024.csv": df})

            with pytest.raises(RuntimeError, match="Validation failed"):
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

    def test_per_file_schema_falls_back_to_global_for_unmatched_file(self):
        """File not matching any per-file pattern is validated against the global schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            # 'other_data.csv' matches no per-file pattern — global only requires 'shared_col'
            df = pd.DataFrame({"shared_col": ["X"]})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"other_data.csv": df})

            result = validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)
            assert result["failed_files"] == []

    def test_per_file_schema_min_rows_enforced_per_file(self):
        """Per-file min_rows (2) is enforced independently from global min_rows (1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            # readmissions per-file schema requires min_rows=2 but file only has 1 row
            df = pd.DataFrame({"shared_col": ["A"], "measure_col": [1.0]})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"readmissions_2024.csv": df})

            with pytest.raises(RuntimeError, match="Validation failed"):
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

    def test_per_file_schema_numeric_bounds_enforced(self):
        """Per-file numeric bounds are enforced; value outside range fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            # measure_col is bounded 0–10 by the per-file schema; 99.0 is out of range
            df = pd.DataFrame({"shared_col": ["A", "B"], "measure_col": [1.0, 99.0]})
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {"readmissions_2024.csv": df})

            with pytest.raises(RuntimeError, match="Validation failed"):
                validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)

    def test_two_files_each_validated_against_own_schema(self):
        """Two files with different schemas are each validated against their matched per-file rule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            config_dir.mkdir()
            self._write_config(config_dir)

            readmissions = pd.DataFrame({
                "shared_col": ["A", "B"],
                "measure_col": [1.0, 2.0],
            })
            hcahps = pd.DataFrame({
                "shared_col": ["X"],
                "survey_col": ["Q1"],
            })
            raw_dir = _make_raw_dir(tmp, "2026-06-29", {
                "readmissions_2024.csv": readmissions,
                "hcahps_survey.csv": hcahps,
            })

            result = validate_raw_files(raw_dir, "2026-06-29", config_dir=config_dir)
            assert result["failed_files"] == []
            assert "readmissions_2024.csv" in result["validated_files"]
            assert "hcahps_survey.csv" in result["validated_files"]


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


_PIVOT_JOIN_PIPELINE_YAML = """\
sources:
  - name: test
    path: data/landing
    format: csv
target:
  name: Excess Readmission Ratio
  type: continuous
problem_type: regression
train_test_split: 0.8
random_state: 42
"""

_PIVOT_JOIN_FEATURES_YAML = """\
join_strategy:
  enabled: true
  id_column: "Facility ID"
  spine:
    file_pattern: "readmissions"
    measure_column: "Measure Name"
    measure_value: "READM-30-PN-HRRP"
  pivots:
    - file_pattern: "hcahps"
      measure_column: "HCAHPS Question"
      measure_filter: "linear mean score"
      value_column: "HCAHPS Linear Mean Value"
      strip_suffix: " - linear mean score"
encoding:
  "State": "frequency"
steps: []
nzv_threshold: 0.95
drop_columns:
  - "Facility ID"
  - "Measure Name"
scale: true
"""

_PIVOT_JOIN_MODELS_YAML = """\
models:
  - name: linear_baseline
    type: linear
    hyperparameters: {}
random_state: 42
train_test_split: 0.8
"""


class TestPivotJoin:
    """Tests for pivot-join feature assembly (multi-source long-format files)."""

    def _setup_config(self, config_dir: Path) -> None:
        """Write minimal configs for pivot-join tests."""
        (config_dir / "pipeline.yaml").write_text(_PIVOT_JOIN_PIPELINE_YAML)
        (config_dir / "features.yaml").write_text(_PIVOT_JOIN_FEATURES_YAML)
        (config_dir / "models.yaml").write_text(_PIVOT_JOIN_MODELS_YAML)

    def _make_interim(self, interim_dir: Path, run_id: str) -> Path:
        """Create spine + pivot CSV files with a manifest."""
        run_path = interim_dir / run_id
        run_path.mkdir(parents=True)

        spine = pd.DataFrame({
            "Facility ID": [1001, 1002, 1003, 1004, 1005],
            "Facility Name": ["A", "B", "C", "D", "E"],
            "State": ["NY", "CA", "TX", "FL", "WA"],
            "Measure Name": ["READM-30-PN-HRRP"] * 5,
            "Excess Readmission Ratio": [0.95, 1.05, 0.88, 1.12, 0.97],
            "Number of Discharges": [200, 350, 180, 420, 150],
        })
        spine.to_csv(run_path / "readmissions.csv", index=False)

        hcahps_rows = []
        questions = ["Nurse communication - linear mean score", "Doctor communication - linear mean score"]
        values = {"Nurse communication - linear mean score": [82.0, 78.5, 85.0, 75.0, 88.0],
                  "Doctor communication - linear mean score": [79.0, 81.0, 77.5, 83.0, 80.0]}
        for fid in [1001, 1002, 1003, 1004, 1005]:
            for q in questions:
                hcahps_rows.append({"Facility ID": fid, "HCAHPS Question": q,
                                     "HCAHPS Linear Mean Value": values[q][[1001,1002,1003,1004,1005].index(fid)]})
            hcahps_rows.append({"Facility ID": fid,
                                 "HCAHPS Question": "Patients who reported nurses communicated",
                                 "HCAHPS Linear Mean Value": None})
        pd.DataFrame(hcahps_rows).to_csv(run_path / "hcahps.csv", index=False)

        import yaml
        with open(run_path / "manifest.yaml", "w") as f:
            yaml.dump({"files": {"readmissions.csv": {}, "hcahps.csv": {}}}, f)

        return interim_dir

    def test_pivot_join_produces_wide_matrix(self):
        """Pivot-join produces one row per hospital with HCAHPS columns appended."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            features_dir = tmp / "features"
            config_dir.mkdir(); features_dir.mkdir()
            self._setup_config(config_dir)
            interim_dir = self._make_interim(tmp / "interim", "2026-06-29")

            result = engineer_features(interim_dir, features_dir, "2026-06-29", config_dir)

            assert result["train_shape"][0] + result["test_shape"][0] == 5
            train_df = pd.read_parquet(features_dir / "2026-06-29" / "train.parquet")
            assert "Nurse communication" in train_df.columns
            assert "Doctor communication" in train_df.columns
            assert "Facility ID" not in train_df.columns

    def test_pivot_join_filters_spine_to_measure(self):
        """Spine file is filtered to the configured measure value only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            features_dir = tmp / "features"
            config_dir.mkdir(); features_dir.mkdir()
            self._setup_config(config_dir)

            run_path = (tmp / "interim" / "2026-06-29")
            run_path.mkdir(parents=True)
            spine = pd.DataFrame({
                "Facility ID": [1001, 1001, 1002, 1002],
                "State": ["NY", "NY", "CA", "CA"],
                "Measure Name": ["READM-30-PN-HRRP", "READM-30-HF-HRRP", "READM-30-PN-HRRP", "READM-30-HF-HRRP"],
                "Excess Readmission Ratio": [0.95, 1.10, 1.05, 0.90],
                "Number of Discharges": [200, 300, 350, 280],
            })
            spine.to_csv(run_path / "readmissions.csv", index=False)
            import yaml
            with open(run_path / "manifest.yaml", "w") as f:
                yaml.dump({"files": {"readmissions.csv": {}}}, f)

            result = engineer_features(tmp / "interim", features_dir, "2026-06-29", config_dir)
            assert result["train_shape"][0] + result["test_shape"][0] == 2


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
