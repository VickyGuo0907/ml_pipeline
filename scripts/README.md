# scripts/

Standalone, pipeline-specific prep scripts that run *before* the generic pipeline (ingest → ... → drift_report). Nothing here is imported by `src/`, and nothing in `src/` imports anything here — this is the one place a pipeline's specific file/year/refresh choices are allowed to live, per the architecture rule that `src/` stays dataset-agnostic (see the root `CLAUDE.md`).

## When a pipeline needs a script here

Most pipelines don't — if a landing zone is just "drop these files in and they never change" (like `biomedical_clinical`), there's no script, just files sitting in `data/<pipeline>/landing/` with a `.gitkeep`.

Add a script here only when populating that landing zone is itself nontrivial: selecting a specific dated refresh out of several, pulling files from more than one source location, or any other one-time decision that isn't expressible as pipeline YAML config. `hospital_readmission_lagged` needs one because its landing zone is assembled from two different dated CMS snapshots (2024 predictors + 2025 target) living in two different source directories.

## Convention

One script per pipeline that needs this, named `stage_<pipeline_name>_landing.py`. Each one follows the same shape — copy it when adding a new one:

1. **Module docstring** — what it stages and *why* these specific files/dates/sources were chosen (not just what the code does; the code already says that).
2. **Constants** at module level for the specific files, dates, and paths involved (`PREDICTOR_FILES_2024`, `TARGET_FILE_2025`, `DEFAULT_SOURCE_2024`, etc. in the existing example) — never buried inside functions, so the pipeline-specific choices are visible at a glance.
3. **One importable, testable core function** (e.g. `stage_landing(...)`) that does the actual work and can be called directly from a test — no side effects beyond the file copy itself, no `print()` inside it.
4. **A thin CLI wrapper** — `main()` with `argparse` (defaults from the module constants, overridable by flag) plus an `if __name__ == "__main__":` guard. This is where `print()` output belongs.

Matching test file: `tests/test_stage_<pipeline_name>_landing.py`, testing the core function against real temp directories (not mocks).

Run any script directly with `uv run python -m scripts.stage_<pipeline_name>_landing`.

## What NOT to do here

Don't build a shared base class or a generic "staging framework." There's no evidence yet that two pipelines' staging needs look alike — one file vs. many, one source dir vs. several, one dated refresh vs. cross-year joins. Keep each script independent until a real, repeated pattern shows up across three or more of them; forcing a common interface on a sample size of one is guessing at a shape that doesn't exist yet.
