# Champion/Challenger Regression Check — Design Spec

**Date:** 2026-07-03
**Status:** Approved for planning (see conversation history for the brainstorm that produced this)

## Context

The pipeline trains multiple candidate models per run (`ols_baseline`, `elastic_net`, `ridge_l2`,
`lasso_l1`, `random_forest`, `lightgbm_gbm` for `biomedical_clinical`; a leaner set for
`bioinfo_gene`). `src/evaluate.py` already gates registration on fixed thresholds
(`min_test_r2`, `max_test_rmse` from `models.yaml`), but nothing compares a newly trained
candidate against whatever is currently deployed to Production. A model that regresses in
accuracy would still register to Staging with no signal beyond the raw numbers sitting in an
evaluation report nobody is watching.

The initial idea was a direct point comparison: reject/warn if `new_test_rmse > production_test_rmse`.
Investigation surfaced why that's not sound as designed:

- `test_rmse` is computed on a train/test split that is **re-drawn every run** from whatever data
  currently exists (`features.py` uses a fixed `random_state`, but that only makes the split
  deterministic for a *given* dataset — it does not make the held-out rows stable across runs
  when the underlying data changes, e.g. a CMS quarterly refresh). Comparing `test_rmse` across
  two different runs compares performance on two different samples, not the same benchmark.
- A single point estimate has no notion of statistical significance — with ~500-900 test rows,
  a small RMSE delta could easily be sampling noise rather than a real regression.
- A regression and a data-drift event look identical in the raw numbers, but call for different
  human responses (retrain vs investigate a bug).

This spec addresses those three gaps. It explicitly does **not** attempt the larger
production-readiness roadmap discussed alongside it (a live ground-truth feedback loop from
`serve.py`, shadow/canary deployment) — that's a separate, larger piece of work.

## Non-goals

- No change to promotion behavior. Promotion to Production remains a manual MLflow UI action —
  this spec only makes the information available to the human clicking that button richer and
  more trustworthy. Nothing here auto-promotes or auto-blocks Staging registration.
- No live/online monitoring of served predictions. This is all offline, at training-run time.
- No support for classification metrics (AUC/F1) in this pass — `problem_type: classification`
  exists in `PipelineConfig` but nothing in the pipeline currently exercises it. Bootstrap CI
  computation here is RMSE-only; extending to a configurable primary metric is a follow-up if a
  classification pipeline is ever added.

## Components

### 1. Fixed benchmark set (`src/benchmark.py`, new)

A benchmark is a frozen feature matrix + target, versioned the same way every other pipeline
artifact is versioned in this repo (dated directory + `manifest.yaml` with a checksum):

```
data/<pipeline>/benchmark/<version>/benchmark.parquet
data/<pipeline>/benchmark/<version>/manifest.yaml     # checksum, source run_id, created_at, row_count
data/<pipeline>/benchmark/current.yaml                # {"version": "<version>"} — pointer to the active benchmark
```

`<version>` uses the same run_id format as everywhere else (ISO date or Airflow logical_date).

**Refresh is manual, not automatic.** A new DAG task, `create_benchmark`, exists in every
pipeline's DAG (added in `dag_factory.py`) but is a no-op unless the DAG run's `conf` explicitly
requests it:

```python
def create_benchmark_wrapper(**context) -> dict:
    conf = context["dag_run"].conf or {}
    if not conf.get("refresh_benchmark", False):
        logger.info("Skipping benchmark refresh — trigger with conf={'refresh_benchmark': true} to refresh")
        return {"skipped": True}
    # snapshot this run's post-validate_features feature matrix as the new benchmark
    ...
```

This keeps a normal scheduled run untouched (no implicit resampling of the benchmark), keeps
"one DAG, task groups" intact (no second DAG), and makes refreshing the golden set a deliberate,
auditable act (it shows up as a DAG run with `conf={"refresh_benchmark": true}` in Airflow's
history).

Config addition — a small block in `pipeline.yaml` (not a new file; per the existing "config
depth: light" principle, a 1-2 field concern doesn't warrant its own YAML):

```yaml
benchmark:
  enabled: true
```

`BenchmarkConfig` (new, in `src/utils/config.py`):
```python
class BenchmarkConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether this pipeline maintains a fixed benchmark set")
```
Defaults to `False` so pipelines that never opt in behave exactly as they do today (no benchmark
directory expected, regression check skipped entirely — see Component 2 error handling).

Core functions in `src/benchmark.py`:
```python
def create_benchmark_snapshot(features_dir, run_id, benchmark_dir, config_dir) -> dict:
    """Copy this run's train.parquet (features + target) into data/<pipeline>/benchmark/<run_id>/,
    write its manifest, and update current.yaml to point at it."""

def load_current_benchmark(benchmark_dir) -> pd.DataFrame | None:
    """Read current.yaml, load that version's benchmark.parquet. Returns None if no benchmark exists yet."""
```

**Source file: `train.parquet`, not `test.parquet`.** A benchmark is a static evaluation set
reused across many future runs — it isn't itself a train/test split, so "held out" doesn't apply
to it the way it does during training. `train.parquet` is larger and more representative of the
full population; `test.parquet` exists only to validate that one run's models, and is an
implementation detail of training, not a natural benchmark source.

### 2. Statistical comparison (`src/evaluate.py`)

At registration time, for a model that already passed the existing threshold gate:

1. Load the current benchmark set for this pipeline via `load_current_benchmark()`. If none
   exists (benchmark disabled, or never refreshed), skip this whole component — register exactly
   as today, no regression field added to the report.
2. Score the new candidate model on the benchmark features.
3. Look up the current Production version of the *same model name* (`client.get_latest_versions(model_name, stages=["Production"])`).
   If none exists (nothing in Production yet for this model type), skip — nothing to compare
   against.
4. Load the Production model via `mlflow.pyfunc.load_model(f"models:/{model_name}/Production")`
   and score it on the *same* benchmark rows.
5. Bootstrap both: resample the benchmark set with replacement ~1000 times, compute RMSE each
   iteration, take the 2.5th/97.5th percentiles as a 95% CI for each model.
6. **Regression = the two CIs don't overlap AND the new model's mean RMSE is the worse one.**
   Overlapping CIs → no flag, even if the point estimate is technically higher — the difference
   isn't distinguishable from noise at this sample size.

New function in `src/benchmark.py`:
```python
def bootstrap_rmse_ci(y_true: pd.Series, y_pred: np.ndarray, n_iterations: int = 1000, confidence: float = 0.95) -> tuple[float, float]:
    """Returns (lower, upper) bound of the RMSE confidence interval via bootstrap resampling."""
```

### 3. Drift context (`src/monitoring.py`, refactor)

Rather than reordering `register_task`/`drift_task` (register currently runs before drift in the
DAG), extract the core Evidently comparison into a shared, cheap helper that returns a boolean
instead of writing an HTML report:

```python
def compute_drift_detected(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool | None:
    """Runs DatasetDriftMetric and returns drift_detected, or None if Evidently unavailable/failed."""
```

`generate_drift_report()` (existing, unchanged behavior) calls this internally and additionally
writes the HTML report. `evaluate.py`'s regression check calls the same helper directly —
comparing this run's training features against the *previous* run's — to attach
`drift_detected: true | false | null` to the regression entry as context. This does not change
the pass/fail decision from Component 2; it's purely interpretive context for the human reading
the report ("regression + drift" suggests the data changed; "regression + no drift" suggests the
model itself is worse).

### 4. Champion tag (unchanged from original design)

After all models in a run are evaluated, the one with the lowest `test_rmse` **from this run's
own train/test split** is tagged `run_champion: "true"` on its registered version, and
`champion_model: <name>` is recorded at the top of the evaluation report. This comparison is
same-run/same-split across the 6 candidates, so it does not have the cross-run comparability
problem that motivated Components 1-3, and needs none of the new benchmark machinery.

## Data flow

```
validate_features_task
    ↓
[create_benchmark_task]  — no-op unless conf.refresh_benchmark == true
    ↓
train_task  → trains all models on this run's split, logs test_rmse/test_r2 per model
    ↓
register_task (evaluate.py):
    for each model that passed the threshold gate:
        - determine run_champion (same-run comparison, all candidates)
        - load_current_benchmark()
            → if present: score candidate + Production (same name) on it,
              bootstrap CI both, flag regression if CIs don't overlap
        - compute_drift_detected(this run's train features, previous run's)
              → attach as context, does not affect the regression flag
        - register to Staging; write tags + evaluation report entry
```

## Evaluation report schema (additions)

```yaml
run_champion: lightgbm_gbm
models:
  lightgbm_gbm:
    status: registered
    test_r2: 0.041
    test_rmse: 0.052
    regression_vs_production: true
    production_rmse_ci: [0.041, 0.048]
    candidate_rmse_ci: [0.052, 0.061]
    drift_detected: true
  ridge_l2:
    status: registered
    test_r2: 0.018
    test_rmse: 0.061
    regression_vs_production: null   # no benchmark configured, or no Production version yet — check skipped
```

New MLflow version tags (alongside the existing `test_rmse`, `test_r2`, etc.): `run_champion`,
`regression_vs_production`, `drift_detected` (all as strings, matching the existing tag
convention).

## Error handling

- Benchmark disabled or missing (`current.yaml` absent) → Component 2 skipped entirely, no
  exception, `regression_vs_production: null` in the report.
- No Production version exists for a given model name → same as above, skipped, not an error.
- Loading the Production pyfunc model fails (registry unavailable, artifact missing) → caught,
  logged as a warning, `regression_vs_production: null`, registration of the *new* candidate
  proceeds unaffected. A failure to evaluate the champion must never block registering the
  challenger.
- `compute_drift_detected` returns `None` (Evidently unavailable, or fewer than 2 runs of history)
  → `drift_detected: null` in the report, not an error.
- `create_benchmark_wrapper` running with `refresh_benchmark: true` but no prior benchmark exists
  → treated as first-time creation, not an update; no manifest to diff against.

## Testing

- `tests/test_benchmark.py` (new): `bootstrap_rmse_ci` math on synthetic data (known distribution,
  check the CI bounds are sane and repeatable with a fixed seed); `create_benchmark_snapshot` /
  `load_current_benchmark` manifest and versioning logic in isolation (tmp_path fixtures, no
  MLflow involved).
- `tests/test_evaluate.py` (extend existing mocked-MLflow suite): no benchmark configured (skip),
  benchmark exists but no Production version (skip), CIs overlap (no flag), CIs don't overlap with
  candidate worse (flag + report fields populated), Production pyfunc load fails (warning, not a
  hard failure, candidate still registers).
- `tests/test_monitoring.py` (new, since `compute_drift_detected` is newly extracted and
  currently untested): drift detected true/false cases with synthetic reference/current
  DataFrames, Evidently unavailable → returns `None`.

## Deferred (explicitly out of scope for this pass)

- Bootstrap iteration count (1000) and confidence level (95%) are hardcoded, not exposed as
  config. Fine as constants for now; promote to `models.yaml` if a pipeline ever needs to tune
  them.
- A configurable "primary comparison metric" (RMSE vs R² vs a classification metric) — deferred
  until a non-regression `problem_type` pipeline actually exists (see Non-goals).
