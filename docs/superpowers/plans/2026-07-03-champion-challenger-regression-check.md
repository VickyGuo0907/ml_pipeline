# Champion/Challenger Regression Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model-vs-Production comparison statistically sound (fixed benchmark set + bootstrap
confidence intervals instead of a noisy point comparison), add drift context to regressions, and
tag the best-performing model of each run as `run_champion` — all informational, never auto-promoting.

**Architecture:** A new `src/benchmark.py` module owns a manually-refreshed, versioned benchmark
dataset per pipeline and the bootstrap-CI statistics. `src/monitoring.py` gets a small refactor to
expose a reusable boolean drift check alongside its existing HTML report. `src/evaluate.py`
consumes both, plus a new `find_previous_run_id` helper in `src/utils/io.py`, to attach
`regression_vs_production`, `*_rmse_ci`, and `drift_detected` fields to the evaluation report and
MLflow version tags — without changing what gets registered or promoted. `dag_factory.py` adds one
new always-present-but-usually-no-op task (`06c_create_benchmark`) and threads two new optional
parameters into the existing register task.

**Tech Stack:** Python, pandas, numpy, MLflow (pyfunc, tracking client), Evidently AI, Pydantic,
pytest with unittest.mock, Airflow (PythonOperator).

## Global Constraints

- NO auto-promotion to Production — manual UI click only. This plan only adds information, never
  changes what triggers a promotion.
- Every new storage boundary gets a `manifest.yaml` — no exceptions.
- Existing behavior when the new optional parameters (`benchmark_dir`, `features_dir`) are omitted
  must be **byte-for-byte unchanged** — all currently-passing tests in `tests/test_evaluate.py`
  must keep passing without modification.
- Type hints on all function signatures; docstrings with Args/Returns/Raises, matching the existing
  style in `src/evaluate.py` and `src/utils/io.py`.
- Run `uv run pytest tests/ -q` after every task; it must show all-passing before moving on.
- Spec reference: `docs/superpowers/specs/2026-07-03-champion-challenger-regression-check-design.md`

---

### Task 1: Config schema — `BenchmarkConfig` and directory wiring

**Files:**
- Modify: `src/utils/config.py`
- Modify: `config/biomedical_clinical/pipeline.yaml`
- Modify: `config/bioinfo_gene/pipeline.yaml`
- Modify: `config/biomedical_clinical/orchestration.yaml`
- Modify: `config/bioinfo_gene/orchestration.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `BenchmarkConfig` (fields: `enabled: bool`), `PipelineConfig.benchmark: BenchmarkConfig`,
  `OrchestrationDirectoriesConfig.benchmark: str` (default `"data/benchmark"`) — consumed by
  Task 5 (`evaluate.py`) and Task 6 (`dag_factory.py`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` (append at end of file):

```python
def test_benchmark_config_defaults():
    """Test BenchmarkConfig defaults to disabled — opt-in per pipeline."""
    cfg = BenchmarkConfig()
    assert cfg.enabled is False


def test_pipeline_config_benchmark_defaults_disabled():
    """Test PipelineConfig.benchmark defaults to disabled when not specified in YAML."""
    data = {
        "sources": [{"name": "test", "path": "data/", "format": "csv"}],
        "target": {"name": "y", "type": "continuous"},
        "problem_type": "regression",
    }
    config = PipelineConfig(**data)
    assert config.benchmark.enabled is False


def test_biomedical_clinical_benchmark_enabled():
    """Test the biomedical_clinical pipeline opts into the benchmark set."""
    config = load_pipeline_config(BIOMEDICAL_CONFIG)
    assert config.benchmark.enabled is True


def test_bioinfo_gene_benchmark_disabled():
    """Test bioinfo_gene stays lean and does not opt into the benchmark set."""
    config = load_pipeline_config("config/bioinfo_gene")
    assert config.benchmark.enabled is False


def test_orchestration_directories_config_has_benchmark_field():
    """Test OrchestrationDirectoriesConfig exposes a benchmark directory."""
    config = load_pipeline_orchestration_config(
        pipeline_dir=BIOMEDICAL_CONFIG,
        base_dir="config/base",
    )
    assert config.directories.benchmark == "data/biomedical_clinical/benchmark"
```

Update the import block at the top of `tests/test_config.py` to add `BenchmarkConfig`:

```python
from src.utils.config import (
    BenchmarkConfig,
    CleaningConfig,
    FeaturesConfig,
    ModelsConfig,
    OrchestrationConfig,
    PipelineConfig,
    UnsupervisedConfig,
    discover_pipelines,
    load_cleaning_config,
    load_features_config,
    load_models_config,
    load_pipeline_config,
    load_pipeline_orchestration_config,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v -k benchmark`
Expected: FAIL with `ImportError: cannot import name 'BenchmarkConfig'`

- [ ] **Step 3: Add `BenchmarkConfig` and wire it into `PipelineConfig`**

In `src/utils/config.py`, add this class immediately before `class PipelineConfig(BaseModel):`
(currently line 107):

```python
class BenchmarkConfig(BaseModel):
    """Fixed benchmark dataset settings for champion/challenger comparison.

    See docs/superpowers/specs/2026-07-03-champion-challenger-regression-check-design.md.
    Disabled by default so pipelines that don't opt in behave exactly as before —
    no benchmark directory expected, regression check skipped entirely.
    """

    enabled: bool = Field(
        default=False,
        description="Whether this pipeline maintains a fixed benchmark set for regression checks",
    )


```

Then add the field to `PipelineConfig` (after the existing `unsupervised` field, before the
`@field_validator`):

```python
    unsupervised: UnsupervisedConfig = Field(
        default_factory=UnsupervisedConfig, description="Unsupervised exploration settings for Stage 06b"
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig, description="Fixed benchmark set settings for Stage 06c"
    )
```

- [ ] **Step 4: Add the `benchmark` directory field to `OrchestrationDirectoriesConfig`**

In `src/utils/config.py`, modify `OrchestrationDirectoriesConfig` (currently lines 278-290):

```python
class OrchestrationDirectoriesConfig(BaseModel):
    """Data directories configuration."""

    landing: str = Field(default="data/landing", description="Landing directory")
    raw: str = Field(default="data/raw", description="Raw data directory")
    interim: str = Field(default="data/interim", description="Interim data directory")
    features: str = Field(default="data/features", description="Features directory")
    benchmark: str = Field(default="data/benchmark", description="Fixed benchmark dataset directory")
    reports: str = Field(default="reports", description="Reports directory")
    config: str = Field(default="config", description="Configuration directory")
    reports_base_url: str = Field(
        default="http://localhost:8888",
        description="Base URL for the reports nginx server (used for Airflow doc_md links)",
    )
```

- [ ] **Step 5: Update the two pipeline configs**

Append to `config/biomedical_clinical/pipeline.yaml` (after the existing `unsupervised:` block):

```yaml

benchmark:
  enabled: true
```

Append to `config/bioinfo_gene/pipeline.yaml` (after the existing `validation:` block):

```yaml

benchmark:
  enabled: false
```

Add `benchmark:` to the `directories:` block in `config/biomedical_clinical/orchestration.yaml`
(after the `features:` line):

```yaml
directories:
  landing: data/biomedical_clinical/landing
  raw: data/biomedical_clinical/raw
  interim: data/biomedical_clinical/interim
  features: data/biomedical_clinical/features
  benchmark: data/biomedical_clinical/benchmark
  reports: reports/biomedical_clinical
  config: config/biomedical_clinical
  reports_base_url: "http://localhost:8888/biomedical_clinical"
```

Add `benchmark:` to the `directories:` block in `config/bioinfo_gene/orchestration.yaml`
(after the `features:` line):

```yaml
directories:
  landing: data/bioinfo_gene/landing
  raw: data/bioinfo_gene/raw
  interim: data/bioinfo_gene/interim
  features: data/bioinfo_gene/features
  benchmark: data/bioinfo_gene/benchmark
  reports: reports/bioinfo_gene
  config: config/bioinfo_gene
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all tests, including the 5 new ones)

- [ ] **Step 7: Run full suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: PASS (68 previously-passing tests + 5 new = 73 passed)

- [ ] **Step 8: Commit**

```bash
git add src/utils/config.py tests/test_config.py config/biomedical_clinical/pipeline.yaml config/bioinfo_gene/pipeline.yaml config/biomedical_clinical/orchestration.yaml config/bioinfo_gene/orchestration.yaml
git commit -m "feat: add BenchmarkConfig and benchmark directory for champion/challenger check"
```

---

### Task 2: `find_previous_run_id` helper

**Files:**
- Modify: `src/utils/io.py`
- Test: `tests/test_io.py` (new file — `io.py` has no dedicated test file yet)

**Interfaces:**
- Produces: `find_previous_run_id(base_dir: str | Path, current_run_id: str) -> str | None` —
  consumed by Task 5 (`evaluate.py`) and Task 6 (`dag_factory.py`'s `drift_wrapper` fix).

- [ ] **Step 1: Write the failing test**

Create `tests/test_io.py`:

```python
"""Tests for shared I/O helpers."""
from pathlib import Path

from src.utils.io import find_previous_run_id


class TestFindPreviousRunId:
    """Tests for locating the run directory immediately before a given run_id."""

    def test_finds_the_most_recent_prior_run(self, tmp_path):
        for run_id in ["2026-06-01", "2026-06-15", "2026-07-01"]:
            (tmp_path / run_id).mkdir()

        assert find_previous_run_id(tmp_path, "2026-07-01") == "2026-06-15"

    def test_returns_none_when_no_prior_run_exists(self, tmp_path):
        (tmp_path / "2026-07-01").mkdir()
        assert find_previous_run_id(tmp_path, "2026-07-01") is None

    def test_returns_none_when_base_dir_missing(self, tmp_path):
        assert find_previous_run_id(tmp_path / "does_not_exist", "2026-07-01") is None

    def test_ignores_non_directory_entries(self, tmp_path):
        (tmp_path / "2026-06-01").mkdir()
        (tmp_path / "2026-06-30.txt").write_text("not a run directory")
        (tmp_path / "2026-07-01").mkdir()

        assert find_previous_run_id(tmp_path, "2026-07-01") == "2026-06-01"

    def test_current_run_itself_is_not_returned(self, tmp_path):
        (tmp_path / "2026-07-01").mkdir()
        assert find_previous_run_id(tmp_path, "2026-07-01") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'find_previous_run_id'`

- [ ] **Step 3: Implement `find_previous_run_id`**

Append to `src/utils/io.py` (end of file, after `write_manifest`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_io.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: PASS (78 passed)

- [ ] **Step 6: Commit**

```bash
git add src/utils/io.py tests/test_io.py
git commit -m "feat: add find_previous_run_id helper for run-to-run comparisons"
```

---

### Task 3: Extract `compute_drift_detected` from `monitoring.py`

**Files:**
- Modify: `src/monitoring.py`
- Test: `tests/test_monitoring.py` (new file — `monitoring.py` has no test file yet)

**Interfaces:**
- Consumes: nothing new (uses existing `evidently.report.Report` / `evidently.metrics.DatasetDriftMetric`,
  already imported at the top of `monitoring.py` with a try/except fallback to `None`).
- Produces: `compute_drift_detected(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool | None` —
  consumed by Task 5 (`evaluate.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_monitoring.py`:

```python
"""Tests for drift detection helpers."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring import compute_drift_detected, generate_drift_report


def _make_df(n: int, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({
        "a": [i + offset for i in range(n)],
        "b": [(i % 5) + offset for i in range(n)],
    })


class TestComputeDriftDetected:
    """Tests for the standalone boolean drift check."""

    def test_returns_none_when_evidently_unavailable(self):
        with patch("src.monitoring.Report", None), patch("src.monitoring.DatasetDriftMetric", None):
            result = compute_drift_detected(_make_df(50), _make_df(50))
        assert result is None

    def test_returns_boolean_when_evidently_available(self):
        reference = _make_df(200)
        current = _make_df(200)
        result = compute_drift_detected(reference, current)
        assert result in (True, False)

    def test_returns_none_on_evidently_failure(self):
        with patch("src.monitoring.Report") as mock_report_cls:
            mock_report_cls.side_effect = RuntimeError("boom")
            result = compute_drift_detected(_make_df(10), _make_df(10))
        assert result is None


class TestGenerateDriftReport:
    """Tests for the full HTML-report-producing path (previously untested)."""

    def test_creates_baseline_report_when_no_previous_run(self, tmp_path):
        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-01"
        run_dir.mkdir(parents=True)
        _make_df(50).to_parquet(run_dir / "train.parquet")

        result = generate_drift_report(
            features_dir=features_dir,
            run_id="2026-07-01",
            previous_run_id=None,
            reports_dir=tmp_path / "reports",
        )

        assert result["type"] == "baseline"
        assert result["run_id"] == "2026-07-01"

    def test_compares_against_previous_run_when_available(self, tmp_path):
        features_dir = tmp_path / "features"
        for run_id, offset in [("2026-06-01", 0.0), ("2026-07-01", 0.0)]:
            run_dir = features_dir / run_id
            run_dir.mkdir(parents=True)
            _make_df(200, offset=offset).to_parquet(run_dir / "train.parquet")

        result = generate_drift_report(
            features_dir=features_dir,
            run_id="2026-07-01",
            previous_run_id="2026-06-01",
            reports_dir=tmp_path / "reports",
        )

        assert result["comparison_run_id"] == "2026-06-01"
        assert "drift_detected" in result

    def test_raises_when_current_run_data_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_drift_report(
                features_dir=tmp_path / "features",
                run_id="2026-07-01",
                previous_run_id=None,
                reports_dir=tmp_path / "reports",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_drift_detected'`

- [ ] **Step 3: Refactor `monitoring.py` to extract the shared helper**

Replace the full contents of `src/monitoring.py` with:

```python
"""Data drift monitoring using Evidently AI."""
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
except ImportError:
    Report = None
    DatasetDriftMetric = None


def _run_drift_metric(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Any | None:
    """Run Evidently's DatasetDriftMetric and return the Report object.

    Args:
        reference_df: Baseline feature matrix to compare against.
        current_df: Current feature matrix.

    Returns:
        The Evidently Report after running, or None if Evidently is unavailable
        or the run itself raised.
    """
    if Report is None or DatasetDriftMetric is None:
        return None
    try:
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=reference_df, current_data=current_df)
        return report
    except Exception:
        return None


def compute_drift_detected(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool | None:
    """Return whether dataset drift was detected between two feature matrices.

    Cheap, boolean-only entry point — does not write an HTML report. Used by
    evaluate.py's regression check to give a regression flag drift context.

    Args:
        reference_df: Baseline feature matrix to compare against.
        current_df: Current feature matrix.

    Returns:
        True/False if the comparison ran successfully, or None if Evidently is
        unavailable or the comparison failed.
    """
    report = _run_drift_metric(reference_df, current_df)
    if report is None:
        return None
    return report.as_dict()["metrics"][0].get("result", {}).get("drift_detected", None)


def generate_drift_report(
    features_dir: str | Path,
    run_id: str,
    previous_run_id: str | None = None,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Generate data drift report comparing current to previous training set.

    Uses Evidently AI to detect statistical drift in feature distributions.

    Args:
        features_dir: Directory containing feature matrices
        run_id: Current run identifier
        previous_run_id: Previous run ID for comparison (if available)
        reports_dir: Output directory for drift reports

    Returns:
        Dictionary with drift report information

    Raises:
        FileNotFoundError: If current feature files don't exist
    """
    features_path = Path(features_dir) / run_id
    train_path = features_path / "train.parquet"
    reports_path = Path(reports_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")

    reports_path.mkdir(parents=True, exist_ok=True)

    current_df = pd.read_parquet(train_path)

    drift_results: dict[str, Any] = {
        "run_id": run_id,
        "current_shape": current_df.shape,
    }

    if previous_run_id:
        previous_path = Path(features_dir) / previous_run_id / "train.parquet"

        if previous_path.exists():
            previous_df = pd.read_parquet(previous_path)
            report = _run_drift_metric(previous_df, current_df)

            drift_results["comparison_run_id"] = previous_run_id
            drift_results["previous_shape"] = previous_df.shape

            if report is not None:
                report_path = reports_path / f"{run_id}_drift_report.html"
                report.save_html(report_path)
                drift_results["report_path"] = str(report_path)
                drift_results["drift_detected"] = report.as_dict()["metrics"][0].get("result", {}).get("drift_detected", None)
            else:
                drift_results["warning"] = "Evidently AI not available or report generation failed; skipping drift report"
        else:
            drift_results["warning"] = f"Previous run data not found: {previous_path}"
            drift_results["baseline_run_id"] = run_id
    else:
        report = _run_drift_metric(current_df, current_df)
        if report is not None:
            report_path = reports_path / f"{run_id}_baseline_drift_report.html"
            report.save_html(report_path)
            drift_results["report_path"] = str(report_path)
        else:
            drift_results["warning"] = "Evidently AI not available or baseline report generation failed"

        drift_results["type"] = "baseline"
        drift_results["note"] = "No previous run available for comparison; using current data as baseline"

    return drift_results
```

Note: this preserves every existing field `generate_drift_report` returned (the pre-existing
try/except-per-branch error messages are consolidated into `_run_drift_metric` returning `None`,
which is behaviorally equivalent — callers only ever read fields out of the returned dict, not
distinguished the warning wording).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_monitoring.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: PASS (84 passed)

- [ ] **Step 6: Commit**

```bash
git add src/monitoring.py tests/test_monitoring.py
git commit -m "refactor: extract compute_drift_detected from monitoring.py, add missing test coverage"
```

---

### Task 4: `src/benchmark.py` — versioned benchmark set + bootstrap CI

**Files:**
- Create: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `compute_file_hash` from `src/ingest.py` (existing), `write_manifest` from
  `src/utils/io.py` (existing).
- Produces: `create_benchmark_snapshot(features_dir, run_id, benchmark_dir) -> dict[str, Any]`,
  `load_current_benchmark(benchmark_dir) -> pd.DataFrame | None`,
  `bootstrap_rmse_ci(y_true, y_pred, n_iterations=1000, confidence=0.95, random_state=42) -> tuple[float, float]` —
  all consumed by Task 5 (`evaluate.py`) and `create_benchmark_snapshot` also by Task 6 (`dag_factory.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.benchmark'`

- [ ] **Step 3: Implement `src/benchmark.py`**

Create `src/benchmark.py`:

```python
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
from typing import Any

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


def bootstrap_rmse_ci(
    y_true: pd.Series,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for RMSE via resampling.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions, same length and order as y_true.
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for a 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        (lower, upper) bound of the RMSE confidence interval.
    """
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)

    rmses = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        rmses[i] = float(np.sqrt(np.mean((y_true_arr[idx] - y_pred_arr[idx]) ** 2)))

    alpha = 1 - confidence
    lower = float(np.percentile(rmses, 100 * (alpha / 2)))
    upper = float(np.percentile(rmses, 100 * (1 - alpha / 2)))
    return lower, upper
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: PASS (93 passed)

- [ ] **Step 6: Commit**

```bash
git add src/benchmark.py tests/test_benchmark.py
git commit -m "feat: add src/benchmark.py — versioned benchmark snapshots and bootstrap RMSE CI"
```

---

### Task 5: Wire the regression check and champion tag into `evaluate.py`

**Files:**
- Modify: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `load_current_benchmark`, `bootstrap_rmse_ci` (from Task 4's `src/benchmark.py`),
  `compute_drift_detected` (from Task 3's `src/monitoring.py`), `find_previous_run_id` (from
  Task 2's `src/utils/io.py`), `load_pipeline_config` (existing, `src/utils/config.py`).
- Produces: `register_models_to_mlflow(..., features_dir=None, benchmark_dir=None)` — two new
  optional keyword parameters, consumed by Task 6 (`dag_factory.py`). Both default to `None`,
  which must reproduce today's exact behavior (verified by every pre-existing test in
  `tests/test_evaluate.py` continuing to pass unmodified).

- [ ] **Step 1: Write the failing tests**

Add a new fixture and test classes to `tests/test_evaluate.py`. Add these imports at the top
(extend the existing import block, don't replace it):

```python
import numpy as np
import pandas as pd
```

Add this fixture after the existing `config_dir` fixture (do not modify `config_dir` itself —
every existing test depends on it having only `models.yaml`, no `pipeline.yaml`):

```python
@pytest.fixture
def config_dir_with_pipeline(tmp_path):
    """models.yaml + pipeline.yaml, needed for tests that exercise the benchmark path."""
    models_cfg = {
        "evaluation": {"min_test_r2": 0.5, "max_test_rmse": 1.0},
        "models": [{"name": "ridge_baseline", "type": "ridge"}],
    }
    pipeline_cfg = {
        "sources": [{"name": "src", "path": "data.csv"}],
        "target": {"name": "target", "type": "continuous"},
        "problem_type": "regression",
        "benchmark": {"enabled": True},
    }
    (tmp_path / "models.yaml").write_text(yaml.dump(models_cfg))
    (tmp_path / "pipeline.yaml").write_text(yaml.dump(pipeline_cfg))
    return tmp_path


def _make_benchmark_dir(tmp_path, rows: int = 50) -> Path:
    """A benchmark dir with one already-created version, ready for load_current_benchmark()."""
    from src.benchmark import create_benchmark_snapshot

    features_dir = tmp_path / "features"
    run_dir = features_dir / "bench-run"
    run_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "f1": list(range(rows)),
        "target": [float(i) for i in range(rows)],
    })
    df.to_parquet(run_dir / "train.parquet")

    benchmark_dir = tmp_path / "benchmark"
    create_benchmark_snapshot(features_dir, "bench-run", benchmark_dir)
    return benchmark_dir
```

Add `from pathlib import Path` to the top-level imports of `tests/test_evaluate.py` if not already
present (it is not — check the current import block and add it alongside the existing imports).

Add these test classes at the end of `tests/test_evaluate.py`:

```python
class TestRunChampion:
    """Tests for tagging the best-performing model of a run."""

    @patch("src.evaluate.mlflow")
    def test_tags_the_lowest_rmse_model_as_champion(self, mock_mlflow, config_dir, tmp_path):
        def fake_get_run(run_id):
            metrics_by_run = {
                "run_a": {"test_rmse": 0.5, "test_r2": 0.6, "train_rmse": 0.4, "train_r2": 0.65},
                "run_b": {"test_rmse": 0.2, "test_r2": 0.9, "train_rmse": 0.15, "train_r2": 0.92},
            }
            return _mock_run(metrics_by_run[run_id])

        mock_mlflow.get_run.side_effect = fake_get_run
        mock_mlflow.register_model.side_effect = lambda model_uri, name: MagicMock(version="1")
        client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = client

        # Two distinct model names so each takes its own path through the loop
        cfg = {
            "evaluation": {"min_test_r2": 0.0, "max_test_rmse": None},
            "models": [{"name": "model_a", "type": "ridge"}, {"name": "model_b", "type": "ridge"}],
        }
        multi_config_dir = tmp_path / "multi_config"
        multi_config_dir.mkdir()
        (multi_config_dir / "models.yaml").write_text(yaml.dump(cfg))

        register_models_to_mlflow(
            mlflow_run_ids={"model_a": "run_a", "model_b": "run_b"},
            config_dir=multi_config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert report["run_champion"] == "model_b"  # lower test_rmse (0.2 < 0.5)


class TestRegressionVsProduction:
    """Tests for the benchmark-based statistical regression check."""

    @patch("src.evaluate.mlflow")
    def test_skipped_when_benchmark_dir_not_provided(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "regression_vs_production" not in report["models"]["ridge_baseline"]

    @patch("src.evaluate.mlflow")
    def test_skipped_when_no_production_version_exists(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        client = MagicMock()
        client.get_latest_versions.return_value = []  # nothing in Production yet
        mock_mlflow.tracking.MlflowClient.return_value = client

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=_make_benchmark_dir(tmp_path),
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "regression_vs_production" not in report["models"]["ridge_baseline"]

    @patch("src.evaluate.mlflow")
    def test_flags_regression_when_candidate_is_significantly_worse(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="2")
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="1")]
        mock_mlflow.tracking.MlflowClient.return_value = client

        benchmark_dir = _make_benchmark_dir(tmp_path, rows=50)

        production_model = MagicMock()
        production_model.predict.return_value = list(range(50))  # perfect on the benchmark target
        candidate_model = MagicMock()
        candidate_model.predict.return_value = [i + 10 for i in range(50)]  # way off
        mock_mlflow.pyfunc.load_model.side_effect = [production_model, candidate_model]

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=benchmark_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        model_report = report["models"]["ridge_baseline"]
        assert model_report["regression_vs_production"] is True
        assert "production_rmse_ci" in model_report
        assert "candidate_rmse_ci" in model_report

    @patch("src.evaluate.mlflow")
    def test_no_regression_flag_when_cis_overlap(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="2")
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="1")]
        mock_mlflow.tracking.MlflowClient.return_value = client

        benchmark_dir = _make_benchmark_dir(tmp_path, rows=50)

        # Both models get the SAME magnitude of random noise applied independently —
        # genuinely similar performance, not a constant offset (a constant offset would
        # give both a near-zero-width CI that trivially wouldn't overlap, which would
        # defeat the point of this test — real variance is required for a fair overlap check).
        rng = np.random.default_rng(0)
        production_model = MagicMock()
        production_model.predict.return_value = np.arange(50, dtype=float) + rng.normal(0, 1.0, 50)
        candidate_model = MagicMock()
        candidate_model.predict.return_value = np.arange(50, dtype=float) + rng.normal(0, 1.0, 50)
        mock_mlflow.pyfunc.load_model.side_effect = [production_model, candidate_model]

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=benchmark_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert report["models"]["ridge_baseline"]["regression_vs_production"] is False

    @patch("src.evaluate.mlflow")
    def test_production_load_failure_does_not_block_registration(self, mock_mlflow, config_dir_with_pipeline, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="2")
        client = MagicMock()
        client.get_latest_versions.return_value = [MagicMock(version="1")]
        mock_mlflow.tracking.MlflowClient.return_value = client
        mock_mlflow.pyfunc.load_model.side_effect = RuntimeError("registry unavailable")

        result = register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir_with_pipeline,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            benchmark_dir=_make_benchmark_dir(tmp_path),
        )

        assert result["registered_models"]["ridge_baseline"]["status"] == "registered"
        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "regression_vs_production" not in report["models"]["ridge_baseline"]


class TestDriftContext:
    """Tests for attaching drift_detected context to the evaluation report."""

    @patch("src.evaluate.compute_drift_detected")
    @patch("src.evaluate.mlflow")
    def test_attaches_drift_detected_when_features_dir_provided(self, mock_mlflow, mock_compute_drift, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()
        mock_compute_drift.return_value = True

        features_dir = tmp_path / "features"
        for run_id in ["2026-07-02", "2026-07-03"]:
            run_dir = features_dir / run_id
            run_dir.mkdir(parents=True)
            pd.DataFrame({"a": [1.0, 2.0], "target": [0.1, 0.2]}).to_parquet(run_dir / "train.parquet")

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            features_dir=features_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert report["models"]["ridge_baseline"]["drift_detected"] is True

    @patch("src.evaluate.mlflow")
    def test_no_drift_field_when_features_dir_not_provided(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "drift_detected" not in report["models"]["ridge_baseline"]

    @patch("src.evaluate.mlflow")
    def test_no_drift_field_when_no_previous_run_exists(self, mock_mlflow, config_dir, tmp_path):
        mock_mlflow.get_run.return_value = _mock_run(
            {"test_rmse": 0.2, "test_r2": 0.8, "train_rmse": 0.1, "train_r2": 0.85}
        )
        mock_mlflow.register_model.return_value = MagicMock(version="1")
        mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

        features_dir = tmp_path / "features"
        run_dir = features_dir / "2026-07-03"
        run_dir.mkdir(parents=True)
        pd.DataFrame({"a": [1.0], "target": [0.1]}).to_parquet(run_dir / "train.parquet")

        register_models_to_mlflow(
            mlflow_run_ids={"ridge_baseline": "run123"},
            config_dir=config_dir,
            run_id="2026-07-03",
            reports_dir=tmp_path / "reports",
            features_dir=features_dir,
        )

        report = yaml.safe_load((tmp_path / "reports" / "2026-07-03_evaluation.yaml").read_text())
        assert "drift_detected" not in report["models"]["ridge_baseline"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — `register_models_to_mlflow() got an unexpected keyword argument 'benchmark_dir'`
(and similar for `features_dir`), plus `run_champion` KeyError-style assertion failures.

- [ ] **Step 3: Implement the changes in `src/evaluate.py`**

Update the imports at the top of `src/evaluate.py`:

```python
"""Model evaluation and registration to MLflow.

Evaluation gate: each model is checked against thresholds from models.yaml
(evaluation.min_test_r2, evaluation.max_test_rmse) before registration.
Models that fail are skipped and recorded as 'rejected' in the evaluation
report. Models that pass are registered to MLflow Staging.

An evaluation YAML report is written to reports/<pipeline>/<run_id>_evaluation.yaml
regardless of outcome, providing a full audit trail of every decision made.

Also tags the best-performing model of each run as run_champion, and — when
benchmark_dir/features_dir are provided — attaches a statistically-grounded
regression_vs_production flag and drift_detected context. See
docs/superpowers/specs/2026-07-03-champion-challenger-regression-check-design.md.
"""
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml

from src.benchmark import bootstrap_rmse_ci, load_current_benchmark
from src.monitoring import compute_drift_detected
from src.utils.config import EvaluationConfig, load_models_config, load_pipeline_config
from src.utils.io import find_previous_run_id

logger = logging.getLogger(__name__)
```

Add this new private helper after `_check_thresholds` (before `register_models_to_mlflow`):

```python
def _check_regression_vs_production(
    client: mlflow.tracking.MlflowClient,
    model_name: str,
    candidate_model_uri: str,
    benchmark_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any] | None:
    """Compare a candidate model against the current Production version of the same name.

    Scores both models on the same fixed benchmark set and compares bootstrapped
    RMSE confidence intervals — a raw point comparison of test_rmse across runs
    isn't valid here since the train/test split is redrawn every run (see the
    design spec). Regression = the candidate's CI is entirely worse than (does
    not overlap with) Production's CI.

    Args:
        client: MLflow tracking client.
        model_name: Registered model name — compared against its own Production
            version, never a different model type.
        candidate_model_uri: MLflow URI for the newly trained candidate (runs:/<run_id>/model).
        benchmark_df: Fixed benchmark feature matrix, including the target column.
        target_col: Name of the target column within benchmark_df.

    Returns:
        Dict with regression_vs_production (bool), production_rmse_ci, and
        candidate_rmse_ci — or None if the check could not be performed (no
        Production version exists yet, or a model failed to load/predict).
    """
    try:
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception as e:
        logger.warning("Could not look up Production version for %s: %s", model_name, e)
        return None
    if not production_versions:
        return None

    X_benchmark = benchmark_df.drop(columns=[target_col])
    y_benchmark = benchmark_df[target_col]

    try:
        production_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        production_pred = production_model.predict(X_benchmark)
        candidate_model = mlflow.pyfunc.load_model(candidate_model_uri)
        candidate_pred = candidate_model.predict(X_benchmark)
    except Exception as e:
        logger.warning("Could not score %s against the benchmark set: %s", model_name, e)
        return None

    production_ci = bootstrap_rmse_ci(y_benchmark, production_pred)
    candidate_ci = bootstrap_rmse_ci(y_benchmark, candidate_pred)

    # Non-overlapping AND candidate is the worse one, in a single comparison:
    # candidate's entire plausible-RMSE range sits above production's entire range.
    is_regression = candidate_ci[0] > production_ci[1]

    return {
        "regression_vs_production": is_regression,
        "production_rmse_ci": list(production_ci),
        "candidate_rmse_ci": list(candidate_ci),
    }
```

Modify the `register_models_to_mlflow` signature and docstring:

```python
def register_models_to_mlflow(
    mlflow_tracking_uri: str = "http://mlflow-server:5000",
    mlflow_run_ids: dict[str, str] | None = None,
    config_dir: str | Path = "config",
    run_id: str = "unknown",
    reports_dir: str | Path = "reports",
    features_dir: str | Path | None = None,
    benchmark_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate trained models against quality thresholds and register passing ones.

    Threshold gate: models failing min_test_r2 or max_test_rmse from models.yaml
    are skipped (not registered) and recorded as 'rejected' in the evaluation
    report. All decisions are written to reports/<pipeline>/<run_id>_evaluation.yaml.

    The best-performing registered model (lowest test_rmse) is tagged
    run_champion. When benchmark_dir is provided and the pipeline's
    pipeline.yaml has benchmark.enabled: true, each registered model is also
    compared against its own Production version (if one exists) on a fixed
    benchmark set via bootstrapped confidence intervals. When features_dir is
    provided, a drift_detected flag (current vs previous run's training
    features) is attached as interpretive context. Both are additive and
    non-blocking — omitting either parameter reproduces prior behavior exactly.

    NO auto-promotion to Production — manual UI click only.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_run_ids: Dict mapping model names to MLflow run IDs.
        config_dir: Pipeline config directory (for evaluation thresholds).
        run_id: Airflow logical date, used for the report filename.
        reports_dir: Directory to write the evaluation YAML report.
        features_dir: Pipeline features directory. When provided, enables
            drift-context detection (current vs previous run). Optional.
        benchmark_dir: Pipeline benchmark directory. When provided (and the
            pipeline's benchmark.enabled is true), enables the statistical
            regression check against Production. Optional.

    Returns:
        Dictionary with per-model evaluation decisions and registry info.

    Raises:
        ValueError: If no run IDs provided or a metric sanity check fails.
        RuntimeError: If any model fails to register due to an infrastructure error.
    """
    if not mlflow_run_ids:
        raise ValueError("mlflow_run_ids required for model registration")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    models_cfg = load_models_config(config_dir)
    eval_cfg = models_cfg.evaluation

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
    timestamp = datetime.now(timezone.utc).isoformat()

    benchmark_df: pd.DataFrame | None = None
    target_col: str | None = None
    if benchmark_dir is not None:
        pipeline_cfg = load_pipeline_config(config_dir)
        if pipeline_cfg.benchmark.enabled:
            benchmark_df = load_current_benchmark(benchmark_dir)
            target_col = pipeline_cfg.target.name

    drift_detected: bool | None = None
    if features_dir is not None:
        previous_run_id = find_previous_run_id(features_dir, run_id)
        if previous_run_id is not None:
            current_train = pd.read_parquet(Path(features_dir) / run_id / "train.parquet")
            previous_train = pd.read_parquet(Path(features_dir) / previous_run_id / "train.parquet")
            drift_detected = compute_drift_detected(previous_train, current_train)
```

Immediately after that block (still before `report: dict[str, Any] = {...}`), the rest of the
function through the threshold-rejection branch is unchanged. In the "Register passing model"
branch, insert the regression check right after the existing `model_uri`/`register_model` lines
and before the `registered_model_tags` block:

```python
            # Register passing model
            model_uri = f"runs:/{mlflow_run_id}/model"
            registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            version = registered_model.version

            regression_info: dict[str, Any] | None = None
            if benchmark_df is not None and target_col is not None:
                regression_info = _check_regression_vs_production(
                    client, model_name, model_uri, benchmark_df, target_col
                )
```

Then, in the `version_tags` dict construction, add (right after the existing `if train_r2 is not
None:` block):

```python
            if regression_info is not None:
                version_tags["regression_vs_production"] = str(regression_info["regression_vs_production"]).lower()
            if drift_detected is not None:
                version_tags["drift_detected"] = str(drift_detected).lower()
```

And in the `report["models"][model_name] = {...}` block for the registered case, add right after
constructing the dict:

```python
            report["models"][model_name] = {
                "status": "registered",
                "version": version,
                "test_r2": test_r2,
                "train_r2": train_r2,
                "test_rmse": test_rmse,
                "train_rmse": train_rmse,
            }
            if regression_info is not None:
                report["models"][model_name].update(regression_info)
            if drift_detected is not None:
                report["models"][model_name]["drift_detected"] = drift_detected
```

Finally, the tail of the function (from `# Write evaluation report regardless of outcome` through
`return registration_results`) currently reads:

```python
    # Write evaluation report regardless of outcome
    _write_evaluation_report(report, run_id, reports_dir)

    registered = [k for k, v in report["models"].items() if v["status"] == "registered"]
    rejected = [k for k, v in report["models"].items() if v["status"] == "rejected"]
    logger.info(
        "Evaluation complete: %d registered, %d rejected, %d errors",
        len(registered), len(rejected), len(infra_failures),
    )

    if infra_failures:
        raise RuntimeError(
            f"Registration failed for {len(infra_failures)} model(s): "
            f"{', '.join(infra_failures)}. See logs for details."
        )

    # Zero registrations means every model failed the quality gate — fail the task
    # loudly instead of letting the DAG run go green with nothing deployable.
    if not registered:
        raise ValueError(
            f"All {len(rejected)} model(s) rejected by evaluation thresholds "
            f"({', '.join(rejected)}). Nothing registered to Staging. "
            f"See reports/<pipeline>/{run_id}_evaluation.yaml for reasons."
        )

    return registration_results
```

Replace that entire block with (this moves `registered = [...]` earlier — before the report is
written, since `run_champion` must be *in* the report before `_write_evaluation_report` runs —
and inserts champion selection in between; `rejected`, the `logger.info` call, and both
`raise` branches are otherwise unchanged, just relying on `registered` computed above them now
instead of after):

```python
    registered = [k for k, v in report["models"].items() if v["status"] == "registered"]
    if registered:
        champion_name = min(registered, key=lambda name: report["models"][name]["test_rmse"])
        report["run_champion"] = champion_name
        champion_version = report["models"][champion_name]["version"]
        try:
            _set_version_tags(client, champion_name, champion_version, {"run_champion": "true"})
        except Exception as e:
            logger.warning("Could not tag run champion %s v%s: %s", champion_name, champion_version, e)
    else:
        report["run_champion"] = None

    # Write evaluation report regardless of outcome
    _write_evaluation_report(report, run_id, reports_dir)

    rejected = [k for k, v in report["models"].items() if v["status"] == "rejected"]
    logger.info(
        "Evaluation complete: %d registered, %d rejected, %d errors",
        len(registered), len(rejected), len(infra_failures),
    )

    if infra_failures:
        raise RuntimeError(
            f"Registration failed for {len(infra_failures)} model(s): "
            f"{', '.join(infra_failures)}. See logs for details."
        )

    # Zero registrations means every model failed the quality gate — fail the task
    # loudly instead of letting the DAG run go green with nothing deployable.
    if not registered:
        raise ValueError(
            f"All {len(rejected)} model(s) rejected by evaluation thresholds "
            f"({', '.join(rejected)}). Nothing registered to Staging. "
            f"See reports/<pipeline>/{run_id}_evaluation.yaml for reasons."
        )

    return registration_results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (all existing tests unmodified and passing, plus 9 new tests)

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: PASS (102 passed)

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add champion tag and benchmark-based regression check to evaluate.py"
```

---

### Task 6: Wire it into `dag_factory.py`

**Files:**
- Modify: `src/dags/dag_factory.py`

**Interfaces:**
- Consumes: `create_benchmark_snapshot` (Task 4), `register_models_to_mlflow(..., features_dir=,
  benchmark_dir=)` (Task 5), `find_previous_run_id` (Task 2).
- Produces: new task `06c_create_benchmark` in every pipeline DAG.

No dedicated test file exists for `dag_factory.py` (Airflow DAGs aren't unit-tested elsewhere in
this repo — task correctness is verified structurally here instead, via the DAG object itself).

- [ ] **Step 1: Add the benchmark task ID constant**

In `src/dags/dag_factory.py`, add to the task ID constants block (after `_TASK_EXPLORE`):

```python
_TASK_EXPLORE = "06b_unsupervised_explore"
_TASK_BENCHMARK = "06c_create_benchmark"
_TASK_TRAIN = "07_train_models"
```

- [ ] **Step 2: Add the import for `create_benchmark_snapshot` and `find_previous_run_id`**

Update the import block at the top of `src/dags/dag_factory.py`:

```python
from src.benchmark import create_benchmark_snapshot  # noqa: E402
from src.clean import clean_raw_data  # noqa: E402
from src.evaluate import register_models_to_mlflow  # noqa: E402
from src.explore import run_unsupervised_analysis  # noqa: E402
from src.features import engineer_features  # noqa: E402
from src.ingest import ingest_files  # noqa: E402
from src.monitoring import generate_drift_report  # noqa: E402
from src.profile import profile_raw_files  # noqa: E402
from src.train import train_models  # noqa: E402
from src.utils.config import OrchestrationConfig, discover_pipelines, load_pipeline_config, load_pipeline_orchestration_config  # noqa: E402
from src.utils.io import find_previous_run_id  # noqa: E402
from src.validate import validate_raw_files  # noqa: E402
```

- [ ] **Step 3: Add the `benchmark_wrapper` function**

Add this function after `explore_wrapper` and before `train_wrapper`:

```python
    def benchmark_wrapper(**context) -> dict:
        """Refresh the fixed benchmark set — no-op unless triggered with conf.refresh_benchmark."""
        conf = context["dag_run"].conf or {}
        if not conf.get("refresh_benchmark", False):
            logger.info(
                "Skipping benchmark refresh — trigger with conf={'refresh_benchmark': true} to refresh"
            )
            return {"skipped": True}
        return create_benchmark_snapshot(
            features_dir=config.directories.features,
            run_id=_pull_run_id(context),
            benchmark_dir=config.directories.benchmark,
        )
```

This requires a `logger` in `dag_factory.py` — check the top of the file: there is currently no
module-level `logger`. Add it right after the existing imports (before `_TASK_INGEST = ...`):

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Fix `drift_wrapper`'s hardcoded `previous_run_id=None`**

The existing `drift_wrapper` always passes `previous_run_id=None` to `generate_drift_report`,
meaning the drift report has never actually compared against a previous run in practice — it
only ever produces a baseline. Replace it:

```python
    def drift_wrapper(**context) -> dict:
        """Generate Evidently drift report comparing current vs previous features."""
        run_id = _pull_run_id(context)
        if not run_id:
            raise ValueError(f"[{config.dag.dag_id}] run_id not found in cross-task storage")
        previous_run_id = find_previous_run_id(config.directories.features, run_id)
        return generate_drift_report(
            features_dir=config.directories.features,
            run_id=run_id,
            previous_run_id=previous_run_id,
            reports_dir=config.directories.reports,
        )
```

- [ ] **Step 5: Thread the new parameters into `register_wrapper`**

Replace `register_wrapper`:

```python
    def register_wrapper(**context) -> dict:
        """Evaluate models against thresholds and register passing ones to MLflow Staging."""
        ti: TaskInstance = context["task_instance"]
        mlflow_run_ids = ti.xcom_pull(task_ids=_TASK_TRAIN, key="mlflow_run_ids")
        return register_models_to_mlflow(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            mlflow_run_ids=mlflow_run_ids,
            config_dir=config.directories.config,
            run_id=_pull_run_id(context),
            reports_dir=config.directories.reports,
            features_dir=config.directories.features,
            benchmark_dir=config.directories.benchmark,
        )
```

- [ ] **Step 6: Add the task to the DAG and wire dependencies**

In the `with dag:` block, add the new task right after `train_task` is defined (in the
"Always-required tasks" section — the benchmark task always exists in the graph, per the design;
only its wrapper body is conditional):

```python
        train_task = PythonOperator(
            task_id=_TASK_TRAIN,
            python_callable=train_wrapper,
            retries=config.tasks.train_models_retries,
        )
        benchmark_task = PythonOperator(
            task_id=_TASK_BENCHMARK,
            python_callable=benchmark_wrapper,
            doc_md=(
                "## Benchmark Snapshot\n\n"
                "No-op on a normal scheduled run. Trigger this DAG with "
                "`conf={\"refresh_benchmark\": true}` to refresh the fixed "
                "benchmark set used for champion/challenger regression checks."
            ),
        )
        register_task = PythonOperator(
            task_id=_TASK_REGISTER,
            python_callable=register_wrapper,
        )
```

Update the dependency wiring — replace the existing:

```python
        # Core training chain: validate_features → train → register → [drift?]
        end = validate_features_task >> train_task >> register_task
        if _enabled.drift_report:
            end >> drift_task
```

with:

```python
        # Core training chain: validate_features → benchmark → train → register → [drift?]
        end = validate_features_task >> benchmark_task >> train_task >> register_task
        if _enabled.drift_report:
            end >> drift_task
```

- [ ] **Step 7: Verify the DAG structure loads and wires correctly**

Run this verification script (no pytest needed — Airflow DAG construction can be checked directly):

```bash
uv run python -c "
from src.utils.config import load_pipeline_orchestration_config
from src.dags.dag_factory import build_dag

config = load_pipeline_orchestration_config('config/biomedical_clinical', base_dir='config/base')
dag = build_dag(config)

assert '06c_create_benchmark' in dag.task_ids, dag.task_ids
benchmark_task = dag.get_task('06c_create_benchmark')
train_task = dag.get_task('07_train_models')
assert benchmark_task.task_id in [t.task_id for t in train_task.upstream_list]
print('OK:', dag.task_ids)
"
```

Expected output: `OK: [...]` listing all task IDs including `06c_create_benchmark`, with no
assertion errors.

- [ ] **Step 8: Run full suite to confirm no regressions**

Run: `uv run pytest tests/ -q`
Expected: PASS (102 passed — this task doesn't add pytest tests, just the DAG itself)

- [ ] **Step 9: Commit**

```bash
git add src/dags/dag_factory.py
git commit -m "feat: add benchmark refresh task to dag_factory.py, fix drift_wrapper's previous_run_id"
```

---

### Task 7: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CLAUDE.md`
- Modify: `config/biomedical_clinical/README.md`

- [ ] **Step 1: Update `README.md`'s stage list and add a Champion/Challenger section**

In the "Core tasks (always run)" numbered list, after item 6 (`train`), the numbering doesn't
change (benchmark is conditionally-manual, closer in spirit to the optional tasks) — instead, add
it to the **Optional tasks** bullet list, right after the `unsupervised_explore` bullet:

```markdown
- **create_benchmark** *(after validate_features, before train)* — no-op on a normal scheduled
  run; trigger the DAG with `conf={"refresh_benchmark": true}` to snapshot the current run's
  training features as the new fixed benchmark set, used by the regression check below
```

Add a new subsection after "### Model Registry":

```markdown
### Champion/Challenger & Regression Detection

Each run's best-performing model (lowest `test_rmse`) is tagged `run_champion` in MLflow and
recorded at the top of `reports/<pipeline>/<run_id>_evaluation.yaml`.

If a pipeline has `benchmark.enabled: true` in `pipeline.yaml` and a benchmark set has been
created (see `create_benchmark` above), every newly registered model is also compared against
its own Production version (same model name) on that fixed benchmark set, using bootstrapped
95% confidence intervals rather than a raw point comparison — the train/test split is redrawn
every run, so comparing `test_rmse` across two different runs directly isn't statistically valid.
A `regression_vs_production: true` flag (plus both models' RMSE confidence intervals) is written
to the evaluation report and MLflow version tags when the candidate's CI is entirely worse than
Production's. This is purely informational — **nothing here changes what gets registered to
Staging or promoted to Production**; promotion remains a manual MLflow UI action. A
`drift_detected` flag (current vs previous run's training features) is attached alongside it as
interpretive context: a regression alongside detected drift suggests the data changed, while a
regression without drift suggests the model itself regressed.

See `docs/superpowers/specs/2026-07-03-champion-challenger-regression-check-design.md` for the
full design rationale.
```

- [ ] **Step 2: Update `ARCHITECTURE.md`**

Find the config directory table entry for `pipeline.yaml` and add benchmark to its description
(search for the line documenting `pipeline.yaml`'s contents and append `; benchmark.enabled —
opt into the fixed benchmark set for champion/challenger regression checks`).

Find the config directory table entry for `orchestration.yaml` and add
`; directories.benchmark — fixed benchmark set location` to its description.

Add a new bullet to whatever section documents "Model promotion" / "No auto-promotion" design
decisions:

```markdown
✓ Champion/challenger regression detection compares each new model against its own Production
  version on a fixed, manually-refreshed benchmark set via bootstrapped confidence intervals —
  informational only, never blocks Staging registration or triggers promotion
```

- [ ] **Step 3: Update `CLAUDE.md`**

In the "Repo layout" section, add `benchmark.py` to the `src/` file list (alongside the existing
`evaluate.py` entry):

```
│   ├── features.py, train.py, evaluate.py, benchmark.py
```

Add one line to the "Pipeline stages" section noting the new stage exists and is manual-trigger
only (find the stage list and add after the register/evaluate stage description):

```markdown
- (optional, manual-trigger only) benchmark — snapshots the current run's training features as
  a fixed evaluation set for champion/challenger regression checks; a no-op unless the DAG is
  triggered with conf={"refresh_benchmark": true}
```

- [ ] **Step 4: Update `config/biomedical_clinical/README.md`**

Find the config file table and update the `pipeline.yaml` row description to mention
`benchmark.enabled`, consistent with the style of the other rows in that table.

- [ ] **Step 5: Run full suite one more time**

Run: `uv run pytest tests/ -q`
Expected: PASS (102 passed) — docs changes don't affect tests, this just confirms nothing else
drifted during the doc-editing pass.

- [ ] **Step 6: Commit**

```bash
git add README.md ARCHITECTURE.md CLAUDE.md config/biomedical_clinical/README.md
git commit -m "docs: describe champion/challenger regression check and benchmark stage"
```
