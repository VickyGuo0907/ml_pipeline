# Hospital Readmission Lagged Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, independent pipeline (`hospital_readmission_lagged`) that predicts a hospital's **2025** pneumonia excess readmission ratio from its **2024** quality-measure predictors, without touching the existing `biomedical_clinical` (contemporaneous 2024→2024) pipeline.

**Architecture:** The two pipelines share all of `src/` and differ only in `config/<pipeline>/` contents and landing-zone files (per `docs/superpowers/specs`-style locked decisions in `CLAUDE.md`). The only shared-code change is a generic "direct-join" capability for wide (one-row-per-hospital) sources, plus a correctness fix so sparse pivot-source value columns aren't corrupted by cross-measure imputation before they reach the pivot step. Everything else is new YAML config plus a landing-zone staging script that lives outside `src/`.

**Tech Stack:** pandas, pydantic, pandera, Apache Airflow (DAG auto-discovery via `src/dags/dag_factory.py`), pytest.

## Global Constraints

- Do NOT modify `config/biomedical_clinical/*` — it stays the contemporaneous baseline, byte-for-byte behavior preserved.
- Do NOT add hospital-specific branching to `src/`. The only shared-code change is the generic `direct_joins` join type and the `protect_columns` imputation fix — both must work identically for any pipeline, not just this one.
- Do NOT introduce Kubernetes, Feast, Optuna, Ray Tune, MinIO, or streaming components (out of scope per `CLAUDE.md`).
- Do NOT auto-promote models to Production.
- Do NOT skip `manifest.yaml` at storage boundaries.
- Use `uv run pytest` (not bare `pytest`) so the project's pinned dependency set is used.
- Landing zone for the new pipeline (`data/hospital_readmission_lagged/landing/`) must contain **only** the 6 files named in `data/hospitial_readmission_lagged/PIPELINE_HANDOFF.md` — never `Unplanned_Hospital_Visits-Hospital.csv` (it carries a pneumonia readmission rate — leakage) and never the raw CMS dump wholesale.
- Type hints on all function signatures; pandera schemas validate every Parquet boundary; run IDs stay ISO-date strings.

---

### Task 1: Shared config models — `direct_joins` join type + `protect_columns` semantics

**Files:**
- Modify: `src/utils/config.py:179-195` (insert `JoinDirectConfig`, extend `JoinStrategyConfig`, broaden `protect_columns` docstring)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `JoinDirectConfig(BaseModel)` with field `file_pattern: str`. `JoinStrategyConfig.direct_joins: list[JoinDirectConfig]` (new field, default `[]`). These are consumed by Task 2 (`src/features.py`) and Task 4 (`config/hospital_readmission_lagged/features.yaml`).
- Produces: `CleaningConfig.protect_columns` gains an additional meaning (exempt from Stage-4 imputation, not just the high-missing drop) — no signature change, only behavior consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

First, add `JoinDirectConfig` and `JoinStrategyConfig` to the existing top-of-file import block in `tests/test_config.py` (matching this file's established convention of importing config classes at module level, not inline), so:

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

becomes:

```python
from src.utils.config import (
    BenchmarkConfig,
    CleaningConfig,
    FeaturesConfig,
    JoinDirectConfig,
    JoinStrategyConfig,
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

Then add to `tests/test_config.py` (after `test_cleaning_config_defaults`, around line 166):

```python
def test_join_strategy_config_direct_joins_default_empty():
    """Test JoinStrategyConfig.direct_joins defaults to an empty list."""
    config = JoinStrategyConfig()
    assert config.direct_joins == []


def test_join_strategy_config_accepts_direct_joins():
    """Test JoinStrategyConfig parses a direct_joins list of file patterns."""
    config = JoinStrategyConfig(
        enabled=True,
        id_column="Facility ID",
        direct_joins=[{"file_pattern": "Hospital_General_Information"}],
    )
    assert len(config.direct_joins) == 1
    assert isinstance(config.direct_joins[0], JoinDirectConfig)
    assert config.direct_joins[0].file_pattern == "Hospital_General_Information"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k direct_joins -v`
Expected: FAIL — `ImportError: cannot import name 'JoinDirectConfig'` (or `AttributeError: 'JoinStrategyConfig' object has no attribute 'direct_joins'`)

- [ ] **Step 3: Implement the config model**

In `src/utils/config.py`, the existing block at lines 179–195 reads:

```python
class JoinPivotConfig(BaseModel):
    """Config for a side file that gets filtered, pivoted wide, then joined to the spine."""

    file_pattern: str = Field(description="Substring matched against filename to identify this pivot file")
    measure_column: str = Field(description="Column whose distinct values become column headers after pivot")
    measure_filter: str = Field(description="Substring used to filter measure_column rows before pivoting")
    value_column: str = Field(description="Column containing numeric values to fill the pivot table")
    strip_suffix: str = Field(default="", description="Suffix stripped from measure names when naming pivot columns")


class JoinStrategyConfig(BaseModel):
    """Multi-source pivot-join config for building wide feature matrices from long-format files."""

    enabled: bool = Field(default=False, description="Enable pivot-join assembly; False falls back to naive concat")
    id_column: str = Field(default="Facility ID", description="Column used as join key across all sources")
    spine: JoinSpineConfig | None = Field(default=None, description="Primary file that provides the target and row count")
    pivots: list[JoinPivotConfig] = Field(default_factory=list, description="Side files to pivot wide and left-join onto the spine")
```

Replace it with:

```python
class JoinPivotConfig(BaseModel):
    """Config for a side file that gets filtered, pivoted wide, then joined to the spine."""

    file_pattern: str = Field(description="Substring matched against filename to identify this pivot file")
    measure_column: str = Field(description="Column whose distinct values become column headers after pivot")
    measure_filter: str = Field(description="Substring used to filter measure_column rows before pivoting")
    value_column: str = Field(description="Column containing numeric values to fill the pivot table")
    strip_suffix: str = Field(default="", description="Suffix stripped from measure names when naming pivot columns")


class JoinDirectConfig(BaseModel):
    """Config for a side file that's already wide (one row per id_column) and gets left-joined
    onto the spine as-is — no pivot. Used for sources like a hospital directory file where each
    row is already one hospital with its own columns.
    """

    file_pattern: str = Field(description="Substring matched against filename to identify this direct-join file")


class JoinStrategyConfig(BaseModel):
    """Multi-source pivot-join config for building wide feature matrices from long-format files."""

    enabled: bool = Field(default=False, description="Enable pivot-join assembly; False falls back to naive concat")
    id_column: str = Field(default="Facility ID", description="Column used as join key across all sources")
    spine: JoinSpineConfig | None = Field(default=None, description="Primary file that provides the target and row count")
    pivots: list[JoinPivotConfig] = Field(default_factory=list, description="Side files to pivot wide and left-join onto the spine")
    direct_joins: list[JoinDirectConfig] = Field(
        default_factory=list,
        description="Side files already wide (one row per id_column) that are left-joined directly onto the spine, no pivoting",
    )
```

Also update the `protect_columns` field docstring in `CleaningConfig` (around line 165) from:

```python
    protect_columns: list[str] = Field(
        default_factory=list,
        description="Columns excluded from the high-missing-value drop (useful for sparse pivot-join columns)",
    )
```

to:

```python
    protect_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns excluded from the high-missing-value drop AND from Stage 4 imputation — "
            "useful for sparse pivot-join value columns that mix multiple measure types and must "
            "be imputed per-measure after the Stage 5 pivot, not globally here."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -k direct_joins -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full config test suite to check for regressions**

Run: `uv run pytest tests/test_config.py -v`
Expected: All PASS (no existing test references `JoinDirectConfig` or relies on the old `protect_columns` docstring text)

- [ ] **Step 6: Commit**

```bash
git add src/utils/config.py tests/test_config.py
git commit -m "feat: add direct_joins config for wide-source joins, broaden protect_columns semantics"
```

---

### Task 2: Direct-join assembly in feature engineering

**Files:**
- Modify: `src/features.py:91-126` (`_pivot_join_sources`)
- Test: `tests/test_pipeline.py` (new `TestDirectJoin` class)

**Interfaces:**
- Consumes: `JoinStrategyConfig.direct_joins: list[JoinDirectConfig]` from Task 1.
- Produces: `_pivot_join_sources` now also left-joins any file matching a `direct_joins` pattern, and de-duplicates overlapping column names during merge (spine's column wins) instead of letting pandas suffix them `_x`/`_y`. No public signature change — `engineer_features` callers are unaffected.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`, after the `TestPivotJoin` class (after line 841, before `class TestIntegration:`):

```python
_DIRECT_JOIN_FEATURES_YAML = """\
join_strategy:
  enabled: true
  id_column: "Facility ID"
  spine:
    file_pattern: "readmissions"
    measure_column: "Measure Name"
    measure_value: "READM-30-PN-HRRP"
  direct_joins:
    - file_pattern: "hospital_info"
encoding:
  "State": "frequency"
nzv_threshold: 0.95
drop_columns:
  - "Facility ID"
  - "Measure Name"
scale: true
"""


class TestDirectJoin:
    """Tests for direct-join feature assembly (wide sources merged without pivoting)."""

    def _setup_config(self, config_dir: Path) -> None:
        """Write minimal configs for direct-join tests."""
        (config_dir / "pipeline.yaml").write_text(_PIVOT_JOIN_PIPELINE_YAML)
        (config_dir / "features.yaml").write_text(_DIRECT_JOIN_FEATURES_YAML)
        (config_dir / "models.yaml").write_text(_PIVOT_JOIN_MODELS_YAML)

    def _make_interim(self, interim_dir: Path, run_id: str, hospital_info: pd.DataFrame) -> Path:
        """Create spine + direct-join CSV files with a manifest."""
        run_path = interim_dir / run_id
        run_path.mkdir(parents=True)

        spine = pd.DataFrame({
            "Facility ID": [1001, 1002, 1003],
            "State": ["NY", "CA", "TX"],
            "Measure Name": ["READM-30-PN-HRRP"] * 3,
            "Excess Readmission Ratio": [0.95, 1.05, 0.88],
        })
        spine.to_csv(run_path / "readmissions.csv", index=False)
        hospital_info.to_csv(run_path / "hospital_info.csv", index=False)

        import yaml
        with open(run_path / "manifest.yaml", "w") as f:
            yaml.dump({"files": {"readmissions.csv": {}, "hospital_info.csv": {}}}, f)

        return interim_dir

    def test_direct_join_appends_wide_source_without_pivot(self):
        """A direct_joins source is left-joined as-is, no pivot applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            features_dir = tmp / "features"
            config_dir.mkdir(); features_dir.mkdir()
            self._setup_config(config_dir)

            hospital_info = pd.DataFrame({
                "Facility ID": [1001, 1002, 1003],
                "Hospital Type": ["Acute Care Hospitals", "Critical Access Hospitals", "Acute Care Hospitals"],
                "Hospital overall rating": [3, 4, 2],
            })
            self._make_interim(tmp / "interim", "2026-07-10", hospital_info)

            result = engineer_features(tmp / "interim", features_dir, "2026-07-10", config_dir)

            train_df = pd.read_parquet(features_dir / "2026-07-10" / "train.parquet")
            test_df = pd.read_parquet(features_dir / "2026-07-10" / "test.parquet")
            combined = pd.concat([train_df, test_df])
            assert "Hospital overall rating" in combined.columns
            assert result["train_shape"][0] + result["test_shape"][0] == 3

    def test_direct_join_drops_overlapping_columns_instead_of_suffixing(self):
        """Columns already present from the spine are not duplicated with _x/_y suffixes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_dir = tmp / "config"
            features_dir = tmp / "features"
            config_dir.mkdir(); features_dir.mkdir()
            self._setup_config(config_dir)

            # hospital_info also carries a "State" column that would collide with the spine's.
            hospital_info = pd.DataFrame({
                "Facility ID": [1001, 1002, 1003],
                "State": ["ny", "ca", "tx"],
                "Hospital overall rating": [3, 4, 2],
            })
            self._make_interim(tmp / "interim", "2026-07-10", hospital_info)

            engineer_features(tmp / "interim", features_dir, "2026-07-10", config_dir)

            train_df = pd.read_parquet(features_dir / "2026-07-10" / "train.parquet")
            test_df = pd.read_parquet(features_dir / "2026-07-10" / "test.parquet")
            combined = pd.concat([train_df, test_df])
            assert "State_x" not in combined.columns
            assert "State_y" not in combined.columns
            assert "State" in combined.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k TestDirectJoin -v`
Expected: FAIL — `Hospital_General_Information`-style file is silently ignored (no `direct_joins` handling yet), so `"Hospital overall rating" in combined.columns` assertion fails.

- [ ] **Step 3: Implement direct-join handling in `_pivot_join_sources`**

In `src/features.py`, the pivot loop (lines 91–115) currently ends with:

```python
        for pivot_cfg in join_config.pivots:
            if pivot_cfg.file_pattern in f.name:
                if pivot_cfg.measure_column in df.columns:
                    mask = df[pivot_cfg.measure_column].str.contains(
                        pivot_cfg.measure_filter, na=False, regex=False
                    )
                    df = df[mask].copy()
                    if pivot_cfg.strip_suffix:
                        df[pivot_cfg.measure_column] = df[pivot_cfg.measure_column].str.replace(
                            pivot_cfg.strip_suffix, "", regex=False
                        )
                    df[pivot_cfg.value_column] = pd.to_numeric(df[pivot_cfg.value_column], errors="coerce")
                    wide = df.pivot_table(
                        index=id_col,
                        columns=pivot_cfg.measure_column,
                        values=pivot_cfg.value_column,
                        aggfunc="first",
                    ).reset_index()
                    wide.columns.name = None
                    dupes = wide[id_col].duplicated().sum()
                    if dupes:
                        logger.warning("Pivot '%s' has %d duplicate %s after pivot", f.name, dupes, id_col)
                    logger.info("Pivot '%s' → %d rows × %d cols", f.name, len(wide), len(wide.columns))
                    side_dfs.append(wide)
                break

    if spine_df is None:
```

Insert a new loop for `direct_joins` between the `pivots` loop and the `if spine_df is None:` check:

```python
        for pivot_cfg in join_config.pivots:
            if pivot_cfg.file_pattern in f.name:
                if pivot_cfg.measure_column in df.columns:
                    mask = df[pivot_cfg.measure_column].str.contains(
                        pivot_cfg.measure_filter, na=False, regex=False
                    )
                    df = df[mask].copy()
                    if pivot_cfg.strip_suffix:
                        df[pivot_cfg.measure_column] = df[pivot_cfg.measure_column].str.replace(
                            pivot_cfg.strip_suffix, "", regex=False
                        )
                    df[pivot_cfg.value_column] = pd.to_numeric(df[pivot_cfg.value_column], errors="coerce")
                    wide = df.pivot_table(
                        index=id_col,
                        columns=pivot_cfg.measure_column,
                        values=pivot_cfg.value_column,
                        aggfunc="first",
                    ).reset_index()
                    wide.columns.name = None
                    dupes = wide[id_col].duplicated().sum()
                    if dupes:
                        logger.warning("Pivot '%s' has %d duplicate %s after pivot", f.name, dupes, id_col)
                    logger.info("Pivot '%s' → %d rows × %d cols", f.name, len(wide), len(wide.columns))
                    side_dfs.append(wide)
                break

        for direct_cfg in join_config.direct_joins:
            if direct_cfg.file_pattern in f.name:
                dupes = df[id_col].duplicated().sum()
                if dupes:
                    logger.warning(
                        "Direct-join '%s' has %d duplicate %s — deduplicating", f.name, dupes, id_col
                    )
                    df = df.drop_duplicates(subset=[id_col], keep="first")
                logger.info("Direct-join '%s' loaded: %d rows × %d cols", f.name, len(df), len(df.columns))
                side_dfs.append(df)
                break

    if spine_df is None:
```

Then update the merge loop (lines 121–123), currently:

```python
    result = spine_df
    for side_df in side_dfs:
        result = result.merge(side_df, on=id_col, how="left")
```

to drop overlapping columns before each merge so a direct-joined wide source never collides with columns the spine (or an earlier side source) already contributed:

```python
    result = spine_df
    for side_df in side_dfs:
        overlap = [c for c in side_df.columns if c != id_col and c in result.columns]
        if overlap:
            logger.warning(
                "Dropping %d column(s) already present before merge: %s", len(overlap), overlap
            )
            side_df = side_df.drop(columns=overlap)
        result = result.merge(side_df, on=id_col, how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -k "TestDirectJoin or TestPivotJoin" -v`
Expected: All PASS (both new `TestDirectJoin` tests and the pre-existing `TestPivotJoin` tests, confirming no regression to the pivot path)

- [ ] **Step 5: Run the full pipeline test suite**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_pipeline.py
git commit -m "feat: support direct-join wide sources in feature engineering"
```

---

### Task 3: Protect sparse pivot-source columns from premature cross-measure imputation

**Files:**
- Modify: `src/clean.py:60-92` (`_clean_single_file`)
- Test: `tests/test_pipeline.py` (`TestClean` class)

**Interfaces:**
- Consumes: `CleaningConfig.protect_columns` (existing field, broadened semantics from Task 1).
- Produces: `_clean_single_file` no longer runs Stage-4 imputation over `protect_columns`; their pre-imputation values (post sentinel-replacement) pass through unchanged to the interim Parquet/CSV, so a later per-measure pivot (Task 2) or per-column median fill (already in `src/features.py`) imputes them correctly instead of a single global median mixing unrelated measure types.

**Why this matters:** the new lagged pipeline pivots `Timely_and_Effective_Care-Hospital.csv`, `Complications_and_Deaths-Hospital.csv`, and `Healthcare_Associated_Infections-Hospital.csv` on `Measure ID`, with a single `Score` column holding values for ~20-36 unrelated measures per file (verified against the actual CMS files: e.g. `Timely_and_Effective_Care-Hospital.csv`'s `Score` column is ~57% missing/non-numeric across 26 distinct measure types after sentinel replacement). Without this fix, Stage 4's `median_impute` would fill all of those NaNs with one global median computed across every measure type mixed together — a materially wrong number for any individual measure — before the file ever reaches the per-measure pivot in Stage 5.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`, inside `class TestClean:` (after `test_clean_removes_duplicates`, before the closing of the class at line 634):

```python
    def test_clean_skips_imputation_for_protected_columns(self):
        """protect_columns are left as NaN through Stage 4, not globally median-imputed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            interim_dir = Path(tmpdir) / "interim"
            raw_dir = Path(tmpdir) / "raw"
            config_dir = Path(tmpdir) / "config"
            interim_dir.mkdir()
            raw_dir.mkdir()
            config_dir.mkdir()

            (config_dir / "pipeline.yaml").write_text(
                'sources:\n  - name: test\n    path: data/landing\n    format: csv\n'
                'target:\n  name: y\n  type: continuous\nproblem_type: regression\n'
            )
            (config_dir / "cleaning.yaml").write_text(
                'impute_strategy: median\nprotect_columns:\n  - "Score"\n'
            )

            run_id = "2026-07-10"
            df = pd.DataFrame({
                "Facility ID": [1, 2, 3, 4],
                "Measure ID": ["PSI_03", "PSI_04", "MORT_30_PN", "MORT_30_PN"],
                "Score": [1.2, None, 0.08, None],
            })
            csv_path = raw_dir / run_id
            csv_path.mkdir(parents=True)
            df.to_csv(csv_path / "test.csv", index=False)

            import yaml
            with open(csv_path / "manifest.yaml", "w") as f:
                yaml.dump({"files": {"test.csv": {}}}, f)

            clean_raw_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

            cleaned = pd.read_csv(interim_dir / run_id / "test.csv")
            assert cleaned["Score"].isna().sum() == 2  # untouched, not globally median-filled
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k test_clean_skips_imputation_for_protected_columns -v`
Expected: FAIL — `assert 0 == 2` (both NaNs got filled with the global median across `PSI_03`/`PSI_04`/`MORT_30_PN` mixed together)

- [ ] **Step 3: Implement the fix**

In `src/clean.py`, `_clean_single_file` (lines 60–92) currently reads:

```python
def _clean_single_file(
    df: pd.DataFrame,
    cleaning_config: Any,
    sentinels: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply cleaning transformations to a DataFrame.

    Args:
        df: Raw DataFrame loaded by the caller.
        cleaning_config: Validated CleaningConfig.
        sentinels: Strings to replace with NaN (from pipeline.yaml validation.sentinel_values).

    Returns:
        Tuple of (cleaned DataFrame, stats dict).
    """
    initial_shape = df.shape

    df = _apply_type_coercion(df, sentinels)
    df = _drop_high_missing(df, threshold=0.5, protect=cleaning_config.protect_columns)
    df, pattern_dropped = drop_pattern_columns(df, cleaning_config.drop_column_patterns)

    impute_fn = IMPUTE_REGISTRY.get(cleaning_config.impute_strategy, IMPUTE_REGISTRY["median"])
    df = impute_fn(df)

    df = df.drop_duplicates(subset=cleaning_config.duplicates_subset)

    return df, {
        "initial_shape": initial_shape,
        "final_shape": df.shape,
        "rows_removed": initial_shape[0] - df.shape[0],
        "cols_removed": initial_shape[1] - df.shape[1],
        "pattern_dropped": pattern_dropped,
    }
```

Replace the imputation block so `protect_columns` are held out and restored around it:

```python
def _clean_single_file(
    df: pd.DataFrame,
    cleaning_config: Any,
    sentinels: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply cleaning transformations to a DataFrame.

    Args:
        df: Raw DataFrame loaded by the caller.
        cleaning_config: Validated CleaningConfig.
        sentinels: Strings to replace with NaN (from pipeline.yaml validation.sentinel_values).

    Returns:
        Tuple of (cleaned DataFrame, stats dict).
    """
    initial_shape = df.shape

    df = _apply_type_coercion(df, sentinels)
    df = _drop_high_missing(df, threshold=0.5, protect=cleaning_config.protect_columns)
    df, pattern_dropped = drop_pattern_columns(df, cleaning_config.drop_column_patterns)

    impute_fn = IMPUTE_REGISTRY.get(cleaning_config.impute_strategy, IMPUTE_REGISTRY["median"])
    protected = [c for c in cleaning_config.protect_columns if c in df.columns]
    held_out = df[protected].copy() if protected else None
    df = impute_fn(df)
    if held_out is not None:
        df[protected] = held_out

    df = df.drop_duplicates(subset=cleaning_config.duplicates_subset)

    return df, {
        "initial_shape": initial_shape,
        "final_shape": df.shape,
        "rows_removed": initial_shape[0] - df.shape[0],
        "cols_removed": initial_shape[1] - df.shape[1],
        "pattern_dropped": pattern_dropped,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -k test_clean_skips_imputation_for_protected_columns -v`
Expected: PASS

- [ ] **Step 5: Run the full pipeline and pivot-join suites to check for regressions**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: All PASS — `biomedical_clinical`'s existing `protect_columns: ["HCAHPS Linear Mean Value"]` behavior is numerically unchanged (that column is only ever non-null for the rows Stage 5 already isolates by question type, so deferring its imputation to the post-pivot per-column median fill in `src/features.py` produces the same values, just one stage later).

- [ ] **Step 6: Commit**

```bash
git add src/clean.py tests/test_pipeline.py
git commit -m "fix: exempt protect_columns from Stage 4 imputation to avoid cross-measure contamination"
```

---

### Task 4: New pipeline config directory — `config/hospital_readmission_lagged/`

**Files:**
- Create: `config/hospital_readmission_lagged/pipeline.yaml`
- Create: `config/hospital_readmission_lagged/cleaning.yaml`
- Create: `config/hospital_readmission_lagged/features.yaml`
- Create: `config/hospital_readmission_lagged/models.yaml`
- Create: `config/hospital_readmission_lagged/orchestration.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `JoinDirectConfig`/`direct_joins` (Task 1/2), `protect_columns` imputation exemption (Task 1/3).
- Produces: a loadable pipeline config directory — `load_pipeline_config`, `load_cleaning_config`, `load_features_config`, `load_models_config`, `load_pipeline_orchestration_config` all succeed against `config/hospital_readmission_lagged`. Consumed by Task 5 (staging script targets `data/hospital_readmission_lagged/landing`), Task 6 (DAG discovery), and Task 7 (end-to-end run).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (after `test_bioinfo_gene_benchmark_disabled`, around line 206):

```python
LAGGED_CONFIG = "config/hospital_readmission_lagged"


def test_load_hospital_readmission_lagged_pipeline_config():
    """Test the lagged pipeline targets the 2025 HRRP file with a PN filter downstream."""
    config = load_pipeline_config(LAGGED_CONFIG)
    assert config.target.name == "Excess Readmission Ratio"
    assert config.pipeline_type == "hospital_readmission_lagged"
    assert config.train_test_split == 0.80
    patterns = [pf.file_pattern for pf in config.validation.per_file_schemas]
    assert "FY_2025_Hospital_Readmissions_Reduction_Program" in patterns
    assert "Hospital_General_Information" in patterns


def test_load_hospital_readmission_lagged_features_config():
    """Test the lagged pipeline's features.yaml wires the 2025 spine + 2024 pivots/direct-join."""
    config = load_features_config(LAGGED_CONFIG)
    assert config.join_strategy.enabled is True
    assert config.join_strategy.spine.file_pattern == "FY_2025_Hospital_Readmissions_Reduction_Program"
    assert config.join_strategy.spine.measure_value == "READM-30-PN-HRRP"
    pivot_patterns = [p.file_pattern for p in config.join_strategy.pivots]
    assert "HCAHPS" in pivot_patterns
    assert "Timely_and_Effective_Care" in pivot_patterns
    assert "Complications_and_Deaths" in pivot_patterns
    assert "Healthcare_Associated_Infections" in pivot_patterns
    direct_patterns = [d.file_pattern for d in config.join_strategy.direct_joins]
    assert "Hospital_General_Information" in direct_patterns
    assert "Number of Readmissions" in config.drop_columns
    assert "Predicted Readmission Rate" in config.drop_columns
    assert "Expected Readmission Rate" in config.drop_columns


def test_load_hospital_readmission_lagged_cleaning_config():
    """Test the lagged pipeline protects sparse pivot-source Score/HCAHPS columns from imputation."""
    config = load_cleaning_config(LAGGED_CONFIG)
    assert "Score" in config.protect_columns
    assert "HCAHPS Linear Mean Value" in config.protect_columns


def test_load_hospital_readmission_lagged_models_config():
    """Test the lagged pipeline's model ladder matches the capstone validation plan (2+ supervised)."""
    config = load_models_config(LAGGED_CONFIG)
    model_names = [m.name for m in config.models]
    assert "elastic_net" in model_names
    assert "lightgbm_gbm" in model_names
    assert len(model_names) >= 2


def test_hospital_readmission_lagged_orchestration_config():
    """Test the lagged pipeline's DAG id and directories are wired independently of biomedical_clinical."""
    config = load_pipeline_orchestration_config(
        pipeline_dir=LAGGED_CONFIG,
        base_dir="config/base",
    )
    assert config.dag.dag_id == "hospital_readmission_lagged_pipeline"
    assert config.directories.landing == "data/hospital_readmission_lagged/landing"
    assert config.directories.config == "config/hospital_readmission_lagged"
    assert config.tasks.retries == 1  # inherited from base


def test_discover_pipelines_includes_hospital_readmission_lagged():
    """Test discover_pipelines picks up the new pipeline directory alongside the existing two."""
    pipelines = discover_pipelines("config")
    names = [p.name for p in pipelines]
    assert "hospital_readmission_lagged" in names
    assert "biomedical_clinical" in names
    assert "bioinfo_gene" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k hospital_readmission_lagged -v`
Expected: FAIL — `FileNotFoundError: Config file not found: config/hospital_readmission_lagged/pipeline.yaml`

- [ ] **Step 3: Create `config/hospital_readmission_lagged/pipeline.yaml`**

```yaml
# Hospital Readmission Lagged Pipeline
# Workflow: 2024 CMS Hospital Compare quality measures -> 2025 pneumonia excess readmission ratio
# Companion to biomedical_clinical (contemporaneous 2024->2024 baseline); see
# data/hospitial_readmission_lagged/PIPELINE_HANDOFF.md for the full capstone design rationale.

pipeline_type: hospital_readmission_lagged

sources:
  - name: "hospital_readmission_lagged"
    path: "data/hospital_readmission_lagged/landing"
    format: "csv"

# Target comes from the 2025 HRRP file — one period ahead of every predictor file below.
target:
  name: "Excess Readmission Ratio"
  type: "continuous"

problem_type: "regression"

# 80/20 split, held out at the hospital level per the capstone validation plan.
train_test_split: 0.80

random_state: 42

validation:
  sentinel_values:
    - "Not Available"
    - "Too Few to Report"
    - "Not Applicable"

  # Global fallback: Facility ID is the join key present in every file.
  required_columns:
    - "Facility ID"
  min_rows: 1

  per_file_schemas:
    - file_pattern: "FY_2025_Hospital_Readmissions_Reduction_Program"
      required_columns:
        - "Facility ID"
        - "Measure Name"       # needed for the PN filter in the feature stage
      numeric_bounds:
        "Excess Readmission Ratio":
          min: 0.0
          max: 5.0             # CMS-defined ratio; values >5 indicate a data error
      min_rows: 1000

    - file_pattern: "HCAHPS"
      required_columns:
        - "Facility ID"
        - "HCAHPS Question"    # needed for the pivot in the feature stage
      numeric_bounds:
        "HCAHPS Linear Mean Value":
          min: 0.0
          max: 100.0           # satisfaction score; nullable (~89% NaN by design)
      min_rows: 1000

    - file_pattern: "Timely_and_Effective_Care"
      required_columns:
        - "Facility ID"
        - "Measure ID"         # needed for the pivot in the feature stage
      min_rows: 1000

    - file_pattern: "Complications_and_Deaths"
      required_columns:
        - "Facility ID"
        - "Measure ID"         # needed for the pivot in the feature stage
      min_rows: 1000

    - file_pattern: "Healthcare_Associated_Infections"
      required_columns:
        - "Facility ID"
        - "Measure ID"         # needed for the pivot in the feature stage
      min_rows: 1000

    - file_pattern: "Hospital_General_Information"
      required_columns:
        - "Facility ID"
      min_rows: 1000

profiling:
  minimal: false

unsupervised:
  enabled: true
  pca:
    enabled: true
  clustering:
    algorithm: "kmeans"
    max_k: 10

benchmark:
  enabled: true
```

- [ ] **Step 4: Create `config/hospital_readmission_lagged/cleaning.yaml`**

```yaml
# Hospital Readmission Lagged Pipeline — cleaning config
#
# Every source file here carries the same CMS hospital-directory metadata (Facility Name,
# Address, City/Town, ZIP Code, County/Parish, Telephone Number, Footnote, Start/End Date).
# None of it is a usable predictor, and leaving it in would collide with the spine's own
# copies when Hospital_General_Information is direct-joined in Stage 5.
impute_strategy: "median"

drop_column_patterns:
  - "Footnote"
  - "Start Date"
  - "End Date"
  - "Facility Name"
  - "Address"
  - "City/Town"
  - "ZIP Code"
  - "County/Parish"
  - "Telephone Number"

# HCAHPS Linear Mean Value and Score are shared value columns that mix many unrelated
# measure types before the Stage 5 pivot filters/splits them apart. Protecting them here
# means they're neither dropped for high-missing nor globally median-imputed across
# unrelated measures — Stage 5's per-measure pivot, then the per-column median fill in
# src/features.py, impute them correctly instead.
protect_columns:
  - "HCAHPS Linear Mean Value"
  - "Score"

# Columns to check for exact duplicates (null = check all columns)
duplicates_subset: null
```

- [ ] **Step 5: Create `config/hospital_readmission_lagged/features.yaml`**

```yaml
# Hospital Readmission Lagged Pipeline — feature engineering config
#
# Spine: 2025 HRRP file, filtered to the pneumonia measure (the target year).
# Pivots: 4 long-format 2024 predictor files, pivoted wide on their own measure column.
# Direct-join: Hospital_General_Information (2024) is already one row per hospital —
# no pivot needed, just a left-join (see src/features.py's direct_joins support).
join_strategy:
  enabled: true
  id_column: "Facility ID"
  spine:
    file_pattern: "FY_2025_Hospital_Readmissions_Reduction_Program"
    measure_column: "Measure Name"
    measure_value: "READM-30-PN-HRRP"
  pivots:
    - file_pattern: "HCAHPS"
      measure_column: "HCAHPS Question"
      measure_filter: "linear mean score"
      value_column: "HCAHPS Linear Mean Value"
      strip_suffix: " - linear mean score"
    - file_pattern: "Timely_and_Effective_Care"
      measure_column: "Measure ID"
      measure_filter: ""       # keep all 26 process-of-care measures
      value_column: "Score"
      strip_suffix: ""
    - file_pattern: "Complications_and_Deaths"
      measure_column: "Measure ID"
      measure_filter: ""       # keep all 19 measures, including MORT_30_PN (pneumonia mortality)
      value_column: "Score"
      strip_suffix: ""
    - file_pattern: "Healthcare_Associated_Infections"
      measure_column: "Measure ID"
      # Each of the 6 HAI types reports 6 sub-metrics (numerator, eligible cases, CI
      # bounds, device-days, SIR); only the standardized infection ratio (_SIR) is a
      # non-redundant summary — the rest are the inputs used to compute it.
      measure_filter: "_SIR"
      value_column: "Score"
      strip_suffix: ""
  direct_joins:
    - file_pattern: "Hospital_General_Information"

# Encoding applied after the pivot/direct-join assembly
encoding:
  "State": "frequency"
  "Hospital Type": "frequency"
  "Hospital Ownership": "frequency"
  "Emergency Services": "label"
  "Meets criteria for promoting interoperability of EHRs": "label"
  "Meets criteria for birthing friendly designation": "label"

# Box-Cox transform on the target, matching biomedical_clinical (same underlying distribution)
boxcox_target: true

# VIF disabled: same rationale as biomedical_clinical — the pivoted quality measures are
# correlated by design (many are components of the same CMS composite score), and the
# capstone validation plan doesn't call for VIF pruning.
vif_threshold: null

# Near-zero variance filter (removes features where >=95% of values are identical)
nzv_threshold: 0.95

# Drop modeling-specific columns not needed in the feature matrix. Metadata columns are
# already removed at the clean stage; these are the join key, the spine's filter column,
# and the leakage columns the ratio is directly derived from (excess = predicted/expected).
drop_columns:
  - "Facility ID"
  - "Measure Name"
  - "Number of Readmissions"
  - "Predicted Readmission Rate"
  - "Expected Readmission Rate"

scale: true
```

- [ ] **Step 6: Create `config/hospital_readmission_lagged/models.yaml`**

```yaml
# Stage 07 — model training config for hospital_readmission_lagged pipeline
#
# Validation plan requires 1 unsupervised (see pipeline.yaml's unsupervised block) + 2+
# supervised models: an elastic-net baseline and LightGBM, per the capstone proposal, plus
# a random forest for the linear-vs-nonlinear contrast already established in
# biomedical_clinical.
#
# Evaluation thresholds are left disabled (null): this is a brand-new lagged design with
# no prior run to calibrate against, and the proposal explicitly treats a low-but-honest
# R² as an acceptable, publishable result — don't gate registration on a guessed threshold
# before a first real run establishes the baseline. Tighten these once real metrics exist.

random_state: 42
train_test_split: 0.80

evaluation:
  min_test_r2: null
  max_test_rmse: null

models:
  - name: "elastic_net"
    type: "elastic_net"
    hyperparameters:
      alpha: 0.01
      l1_ratio: 0.5
      max_iter: 2000

  - name: "random_forest"
    type: "random_forest"
    hyperparameters:
      n_estimators: 100
      max_features: "sqrt"   # feature count isn't fixed ahead of time like biomedical_clinical's 10
      random_state: 42
      n_jobs: -1

  - name: "lightgbm_gbm"
    type: "gbm"
    hyperparameters:
      n_estimators: 100
      learning_rate: 0.05
      max_depth: 7
      num_leaves: 31
      min_child_samples: 20
      subsample: 0.8
      colsample_bytree: 0.8
      reg_alpha: 0.0
      reg_lambda: 1.0
      random_state: 42
      n_jobs: -1
```

- [ ] **Step 7: Create `config/hospital_readmission_lagged/orchestration.yaml`**

```yaml
dag:
  dag_id: hospital_readmission_lagged_pipeline
  owner: data-eng
  description: Lagged hospital readmission prediction — 2024 quality measures → 2025 pneumonia excess readmission ratio
  schedule: "@monthly"
  catchup: false
  tags:
    - ml
    - biomedical
    - clinical
    - lagged

tasks:
  retries: 1
  retry_delay_minutes: 5
  train_models_retries: 0
  enabled:
    profile: true             # ydata-profiling HTML per source file
    unsupervised_explore: true    # PCA + k-means hospital segmentation, required by the validation plan
    drift_report: true        # Evidently drift vs previous training set

directories:
  landing: data/hospital_readmission_lagged/landing
  raw: data/hospital_readmission_lagged/raw
  interim: data/hospital_readmission_lagged/interim
  features: data/hospital_readmission_lagged/features
  benchmark: data/hospital_readmission_lagged/benchmark
  reports: reports/hospital_readmission_lagged
  config: config/hospital_readmission_lagged
  reports_base_url: "http://localhost:8888/hospital_readmission_lagged"
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All PASS, including the 6 new tests and every pre-existing `biomedical_clinical`/`bioinfo_gene` test (unchanged).

- [ ] **Step 9: Commit**

```bash
git add config/hospital_readmission_lagged/ tests/test_config.py
git commit -m "feat: add hospital_readmission_lagged pipeline config (2024 predictors → 2025 PN ratio)"
```

---

### Task 5: Landing-zone staging script (outside `src/`)

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/stage_hospital_readmission_lagged_landing.py`
- Test: `tests/test_stage_hospital_readmission_lagged_landing.py`

**Interfaces:**
- Produces: `stage_landing(source_2024: Path, source_2025: Path, dest: Path) -> dict[str, Any]` — copies the 5 predictor files from `source_2024` and the 1 target file from `source_2025` into `dest`, creating `dest` if needed. Returns a dict of `{filename: dest_path}`. Raises `FileNotFoundError` if any of the 6 required source files is missing.
- Consumes: nothing from earlier tasks (pure file I/O); its output (`data/hospital_readmission_lagged/landing/`) is consumed by Task 7's end-to-end run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_hospital_readmission_lagged_landing.py`:

```python
"""Tests for the hospital_readmission_lagged landing-zone staging script."""
import tempfile
from pathlib import Path

import pytest

from scripts.stage_hospital_readmission_lagged_landing import (
    PREDICTOR_FILES_2024,
    TARGET_FILE_2025,
    stage_landing,
)


def _touch(path: Path, content: str = "Facility ID,Value\n1,1\n") -> None:
    """Write a minimal CSV so copy2 has real bytes to move."""
    path.write_text(content)


class TestStageLanding:
    """Tests for stage_landing copying the 6 required files into the landing zone."""

    def test_stage_landing_copies_all_six_files(self):
        """All 5 predictor files (2024) and the 1 target file (2025) land in dest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_2024 = tmp / "hospitals_2024-10-30"
            source_2025 = tmp / "hospitals_2025-11-26"
            dest = tmp / "landing"
            source_2024.mkdir()
            source_2025.mkdir()

            for name in PREDICTOR_FILES_2024:
                _touch(source_2024 / name)
            _touch(source_2025 / TARGET_FILE_2025)

            result = stage_landing(source_2024, source_2025, dest)

            assert len(result) == 6
            for name in [*PREDICTOR_FILES_2024, TARGET_FILE_2025]:
                assert (dest / name).exists()

    def test_stage_landing_raises_on_missing_source_file(self):
        """A missing required file raises FileNotFoundError naming the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_2024 = tmp / "hospitals_2024-10-30"
            source_2025 = tmp / "hospitals_2025-11-26"
            dest = tmp / "landing"
            source_2024.mkdir()
            source_2025.mkdir()

            # Omit one predictor file
            for name in PREDICTOR_FILES_2024[1:]:
                _touch(source_2024 / name)
            _touch(source_2025 / TARGET_FILE_2025)

            with pytest.raises(FileNotFoundError, match=PREDICTOR_FILES_2024[0]):
                stage_landing(source_2024, source_2025, dest)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_stage_hospital_readmission_lagged_landing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Create `scripts/__init__.py`**

```python
"""Standalone, dataset-specific prep scripts that run before the generic pipeline.

Nothing here is imported by src/ — these scripts stage landing zones and are the only
place a pipeline's specific file/year/refresh choices are allowed to live, per
CLAUDE.md's architecture rule that src/ stays dataset-agnostic.
"""
```

- [ ] **Step 4: Create `scripts/stage_hospital_readmission_lagged_landing.py`**

```python
"""Stage the hospital_readmission_lagged landing zone from the raw CMS quarterly dumps.

Copies the 5 predictor files from the 2024-10-30 refresh and the 1 target file from the
2025-11-26 refresh into data/hospital_readmission_lagged/landing/. Run this once before
triggering the hospital_readmission_lagged_pipeline DAG (or any time the source refresh
changes) — see data/hospitial_readmission_lagged/PIPELINE_HANDOFF.md for why these two
specific refreshes were chosen and why Unplanned_Hospital_Visits-Hospital.csv is
deliberately excluded (it carries a pneumonia readmission rate — leakage).
"""
import argparse
import shutil
from pathlib import Path
from typing import Any

# 2024 quarterly refresh used for every predictor — verified in PIPELINE_HANDOFF.md that
# the HRRP target is identical across all four 2024 refreshes, so the choice of refresh
# only matters for these predictor files, not for reproducing the target.
DEFAULT_SOURCE_2024 = Path("data/hospitial_readmission_lagged/hospitals_annual_2024/hospitals_2024-10-30")
# 2025 refresh that provides the target — one year ahead of the predictors above.
DEFAULT_SOURCE_2025 = Path("data/hospitial_readmission_lagged/hospitals_annual_2025/hospitals_2025-11-26")
DEFAULT_DEST = Path("data/hospital_readmission_lagged/landing")

PREDICTOR_FILES_2024: list[str] = [
    "HCAHPS-Hospital.csv",
    "Timely_and_Effective_Care-Hospital.csv",
    "Complications_and_Deaths-Hospital.csv",
    "Healthcare_Associated_Infections-Hospital.csv",
    "Hospital_General_Information.csv",
]
TARGET_FILE_2025: str = "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv"


def stage_landing(source_2024: Path, source_2025: Path, dest: Path) -> dict[str, Any]:
    """Copy the 6 required lagged-pipeline files into the landing zone.

    Args:
        source_2024: Directory containing the 2024 quarterly refresh (predictors).
        source_2025: Directory containing the 2025 quarterly refresh (target).
        dest: Landing-zone directory to copy into (created if absent).

    Returns:
        Dict mapping each staged filename to its destination path.

    Raises:
        FileNotFoundError: If any of the 6 required source files is missing.
    """
    missing = [
        str(source_2024 / name) for name in PREDICTOR_FILES_2024 if not (source_2024 / name).exists()
    ]
    if not (source_2025 / TARGET_FILE_2025).exists():
        missing.append(str(source_2025 / TARGET_FILE_2025))
    if missing:
        raise FileNotFoundError(f"Missing required source file(s): {missing}")

    dest.mkdir(parents=True, exist_ok=True)

    staged: dict[str, Any] = {}
    for name in PREDICTOR_FILES_2024:
        dest_path = dest / name
        shutil.copy2(source_2024 / name, dest_path)
        staged[name] = dest_path

    dest_path = dest / TARGET_FILE_2025
    shutil.copy2(source_2025 / TARGET_FILE_2025, dest_path)
    staged[TARGET_FILE_2025] = dest_path

    extra = [
        p.name for p in dest.iterdir()
        if p.name not in staged and not p.name.startswith(".")
    ]
    if extra:
        print(
            f"Warning: {dest} contains {len(extra)} file(s) not managed by this script: {extra}. "
            "Landing zone must contain only the 6 staged files so file_pattern matches stay unambiguous."
        )

    return staged


def main() -> None:
    """CLI entry point: stage the landing zone from the default (or overridden) paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-2024", type=Path, default=DEFAULT_SOURCE_2024)
    parser.add_argument("--source-2025", type=Path, default=DEFAULT_SOURCE_2025)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    staged = stage_landing(args.source_2024, args.source_2025, args.dest)
    print(f"Staged {len(staged)} file(s) into {args.dest}:")
    for name, path in staged.items():
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_stage_hospital_readmission_lagged_landing.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/ tests/test_stage_hospital_readmission_lagged_landing.py
git commit -m "feat: add landing-zone staging script for hospital_readmission_lagged"
```

---

### Task 6: Confirm DAG auto-discovery registers the new pipeline

**Files:**
- Test: `tests/test_config.py` (already added `test_discover_pipelines_includes_hospital_readmission_lagged` in Task 4, Step 1)

**Interfaces:**
- Consumes: `config/hospital_readmission_lagged/orchestration.yaml` (Task 4), `discover_pipelines`/`load_pipeline_orchestration_config` (existing, unchanged).
- Produces: confirmation that `src/dags/dag_factory.py` needs zero code changes — its module-level loop (`for _pipeline_dir in discover_pipelines("config"): ...`) already picks up any directory containing `orchestration.yaml`.

`src/dags/dag_factory.py`'s own docstring states "Adding a new pipeline requires only a new `config/<pipeline>/` directory with an `orchestration.yaml` — no changes to this file," and Task 4 already added a directly-testable regression (`test_discover_pipelines_includes_hospital_readmission_lagged`) plus `test_hospital_readmission_lagged_orchestration_config` proving the merged config resolves to the right `dag_id` and directories. This task is the verification pass confirming that promise holds for a third pipeline, without editing `dag_factory.py`.

- [ ] **Step 1: Run the discovery and orchestration tests**

Run: `uv run pytest tests/test_config.py -k "discover_pipelines or orchestration" -v`
Expected: PASS — `discover_pipelines("config")` returns `biomedical_clinical`, `bioinfo_gene`, and `hospital_readmission_lagged`; `load_pipeline_orchestration_config` resolves `hospital_readmission_lagged_pipeline` as the `dag_id`.

- [ ] **Step 2: Import-check the DAG factory module directly (requires Airflow installed)**

Run: `uv run python -c "from src.dags import dag_factory; print(sorted(k for k in vars(dag_factory) if k.endswith('_pipeline')))"`
Expected output includes all three: `['bioinfo_gene_pipeline', 'biomedical_clinical_pipeline', 'hospital_readmission_lagged_pipeline']`

If this environment doesn't have `apache-airflow` installed (it's a heavy dependency usually only present inside the Airflow Docker container), skip this step and rely on Step 1's config-level proof — `dag_factory.py` itself has no pipeline-specific logic, so passing Step 1 is sufficient evidence the DAG will register once the module imports successfully inside the Airflow container.

- [ ] **Step 3: No commit needed** — this task is verification-only; Task 4 already committed the tests it depends on.

---

### Task 7: End-to-end local run — verify the cross-year join and leakage exclusion

**Files:**
- Test: `tests/test_pipeline.py` (new `TestHospitalReadmissionLaggedIntegration` class)

**Interfaces:**
- Consumes: `stage_landing` (Task 5), `ingest_files`, `validate_raw_files`, `clean_raw_data`, `engineer_features` (all existing, unchanged signatures), `config/hospital_readmission_lagged/` (Task 4).
- Produces: a repeatable, real-data integration test proving the cross-year join and leakage exclusion work end-to-end — skipped automatically (not failed) when the local CMS data dump isn't present, since `data/` is gitignored and this data only exists on the machine that downloaded it.

**Important — a number worth flagging before executing this task:** `data/hospitial_readmission_lagged/PIPELINE_HANDOFF.md` states a verified matched sample of "3,063 hospitals with a usable pneumonia ratio in both years" (2024 AND 2025). That two-year figure is **not** the same quantity this pipeline produces: this pipeline's target comes from the 2025 HRRP file only (2024 files are predictors, not required to also have a valid PN ratio). Running `pd.read_csv` directly against the actual 2025-11-26 file during plan research found **2,731** rows with `Measure Name == "READM-30-PN-HRRP"` and a non-null `Excess Readmission Ratio` — that is the correct expected order of magnitude for this pipeline's spine row count (before the left-joins, which don't drop spine rows). The assertion below uses a generous range (2,000–3,200) rather than hard-coding either number, and prints the actual count so this can be reconciled against the capstone write-up. Surface the printed number to the user after running this task — don't silently assume either figure is "the" right one.

- [ ] **Step 1: Write the integration test**

Add to `tests/test_pipeline.py`, after `class TestIntegration:` (end of file):

```python
_LAGGED_SOURCE_2024 = Path("data/hospitial_readmission_lagged/hospitals_annual_2024/hospitals_2024-10-30")
_LAGGED_SOURCE_2025 = Path("data/hospitial_readmission_lagged/hospitals_annual_2025/hospitals_2025-11-26")
_LAGGED_CONFIG = Path("config/hospital_readmission_lagged")

_LAGGED_DATA_AVAILABLE = (
    (_LAGGED_SOURCE_2024 / "Hospital_General_Information.csv").exists()
    and (_LAGGED_SOURCE_2025 / "FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv").exists()
)


@pytest.mark.skipif(
    not _LAGGED_DATA_AVAILABLE,
    reason="requires the local CMS hospitial_readmission_lagged raw data dump (gitignored, not checked in)",
)
class TestHospitalReadmissionLaggedIntegration:
    """End-to-end run against the real CMS data: staging -> ingest -> validate -> clean -> features."""

    def test_full_run_produces_leakage_free_matched_feature_matrix(self):
        """The 2024->2025 join produces a plausible row count with no leakage columns."""
        from scripts.stage_hospital_readmission_lagged_landing import stage_landing

        landing_dir = Path("data/hospital_readmission_lagged/landing")
        stage_landing(_LAGGED_SOURCE_2024, _LAGGED_SOURCE_2025, landing_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_id = "test-e2e"

            ingest_files(landing_dir=landing_dir, raw_dir=tmp / "raw", run_id=run_id)
            validate_raw_files(raw_dir=tmp / "raw", run_id=run_id, config_dir=_LAGGED_CONFIG)
            clean_raw_data(
                raw_dir=tmp / "raw", interim_dir=tmp / "interim", run_id=run_id, config_dir=_LAGGED_CONFIG
            )
            result = engineer_features(
                interim_dir=tmp / "interim",
                features_dir=tmp / "features",
                run_id=run_id,
                config_dir=_LAGGED_CONFIG,
            )

            train_df = pd.read_parquet(tmp / "features" / run_id / "train.parquet")
            test_df = pd.read_parquet(tmp / "features" / run_id / "test.parquet")
            combined = pd.concat([train_df, test_df])
            total_rows = result["train_shape"][0] + result["test_shape"][0]

            print(f"\nhospital_readmission_lagged E2E: matched {total_rows} hospitals "
                  f"({result['train_shape'][0]} train / {result['test_shape'][0]} test), "
                  f"{combined.shape[1]} columns")

            # Loose bound: proves the cross-year join actually matched a large, plausible
            # slice of hospitals rather than 0, all ~5,394, or some other structurally wrong
            # number. See this task's docstring note above for why an exact figure isn't
            # hard-coded here.
            assert 2000 <= total_rows <= 3200

            leakage_columns = {
                "Number of Readmissions", "Predicted Readmission Rate", "Expected Readmission Rate",
                "Facility ID", "Measure Name",
            }
            assert not leakage_columns & set(combined.columns)

            # Sanity: the direct-joined Hospital_General_Information source actually contributed.
            assert "Hospital Type" in combined.columns
            assert "Excess Readmission Ratio" in combined.columns
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_pipeline.py -k TestHospitalReadmissionLaggedIntegration -v -s`
Expected: PASS, with the printed match count visible (`-s` disables output capture so the `print` line shows). If the local data dump isn't present, this SKIPS rather than fails — that's expected and correct in any environment without the gitignored `data/hospitial_readmission_lagged/` dump.

- [ ] **Step 3: Run the complete test suite one last time**

Run: `uv run pytest -v`
Expected: All PASS (or SKIPPED for the one data-dependent test on machines without the dump), confirming Tasks 1–7 didn't regress `biomedical_clinical` or `bioinfo_gene`.

- [ ] **Step 4: Report the actual matched-hospital count to the user**

Read the printed `total_rows` from Step 2's output and tell the user what it was, alongside the discrepancy noted above (2,731 vs. the handoff doc's 3,063 two-year figure) so they can decide which number belongs in the capstone write-up.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: add end-to-end integration test for hospital_readmission_lagged"
```

---

## After this plan

Not in scope here (left for later, per `PIPELINE_HANDOFF.md`'s own "optional/Week 5" framing):
- Staging the optional `FY_2024_...HRRP...Hospital.csv` file into `biomedical_clinical` for a same-year persistence-check comparison.
- Per-quarter predictor snapshots as a drift-monitoring sensitivity analysis.
- Actually triggering `hospital_readmission_lagged_pipeline` inside the Airflow UI/docker-compose stack (Task 6 verifies discovery without requiring the stack to be running; triggering the real DAG is a manual follow-up once Docker services are up).
