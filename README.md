# ML Pipeline POC

A common, flexible ML pipeline template for tabular regression problems — ingestion, validation,
feature engineering, training, evaluation, and serving, all driven by YAML config rather than
code changes. The goal is a reusable pipeline, not a one-off script: add a new dataset by
dropping a `config/<pipeline>/` directory, and `dag_factory.py` auto-discovers it and registers
a new Airflow DAG with no Python changes required.

Two demo pipelines exercise this in the repo today:

- **`biomedical_clinical`** — CMS Hospital Compare hospital readmission data (the primary
  walkthrough in this README, since it has the richest config: multi-source pivot-joins, Box-Cox,
  VIF pruning, six model types)
- **`bioinfo_gene`** — a second, differently-shaped dataset (gene expression), proving the same
  codebase adapts to a new domain through config alone, with its own optional-task toggles
  (profiling on, clustering and drift off)

CMS Hospital Compare is one demo dataset used to present the pipeline, not the point of the
project — the point is that swapping it for a different dataset takes a new config directory,
not a new pipeline.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- `uv` package manager

### Setup

1. **Clone and configure:**
   ```bash
   cp .env.example .env
   ```

2. **Start all services:**
   ```bash
   docker compose up -d
   ```

3. **Verify services are healthy:**
   ```bash
   docker compose ps
   ```

   All 8 services should show `Healthy` or `Running`:
   - Airflow Postgres (5432)
   - MLflow Postgres (5433)
   - Airflow Webserver (8080)
   - Airflow Scheduler
   - Airflow Triggerer
   - MLflow Server (5001)
   - FastAPI Server (8000)
   - Reports Server (8888)

4. **Install dependencies:**
   ```bash
   uv sync
   ```

### Access UIs

| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | N/A |
| FastAPI | http://localhost:8000/docs | N/A |
| Reports | http://localhost:8888 | N/A |

## Pipeline Architecture

### Stages (7 core + 4 optional — stages 6, 6b, and 6c run in parallel with validation)

**Core tasks (always run):**
1. **ingest** — Move files from `data/<pipeline>/landing` to `data/<pipeline>/raw/<run_id>`, write `manifest.yaml`
2. **validate_raw** — Pandera schema check per source file; rules (required columns, bounds, min rows) from `pipeline.yaml`
3. **clean** — Type coercion, sentinel replacement, missing value imputation, deduplication → `data/<pipeline>/interim/<run_id>`
4. **feature_engineer** — Pivot-join assembly, encoding, NZV filter, Box-Cox, VIF pruning, scaling, train/test split
5. **validate_features** *(parallel with 6b when enabled)* — Row count ≥ 100, all-numeric guard, Pandera check on target column (config-driven from `pipeline.yaml`)
6. **train** — All models from `models.yaml`, log metrics to MLflow
7. **register** — Register models to MLflow Staging (no auto-promotion); refuses if `test_rmse` is invalid or `test_r2 < -1.0`

**Optional tasks (controlled by `tasks.enabled` in `orchestration.yaml`):**
- **profile** *(after validate_raw)* — ydata-profiling HTML reports per source; set `profile: false` to skip
- **unsupervised_explore** *(parallel with validate_features)* — PCA + k-means segmentation; algorithm and `max_k` config-driven via `pipeline.yaml`; writes YAML report; does not feed into training; set `unsupervised_explore: false` to skip
- **create_benchmark** *(after validate_features, before train)* — no-op on a normal scheduled run; trigger the DAG with `conf={"refresh_benchmark": true}` to snapshot the current run's training features as the new fixed benchmark set, used by the regression check below
- **drift_report** *(after register)* — Evidently AI drift monitoring (current vs previous training set); set `drift_report: false` to skip

### Data Flow

```
data/<pipeline>/landing          (e.g. data/biomedical_clinical/landing)
    ↓ [ingest]
data/<pipeline>/raw/<run_id>/manifest.yaml
    ↓ [validate_raw] [profile]
data/<pipeline>/interim/<run_id>
    ↓ [clean]
data/<pipeline>/features/<run_id>/
├── train.parquet
├── test.parquet
└── manifest.yaml
    ↓ [validate_features] [train] [evaluate_register]
mlflow-artifacts/
├── linear_baseline
├── lightgbm_gbm
└── drift_reports/
```

### Model Registry

- **Linear Baseline** — sklearn regularized linear regression
- **LightGBM** — Gradient boosting model
- Both registered to MLflow `Staging` environment
- **NO automatic promotion to Production** — manual UI click only

### Champion/Challenger & Regression Detection

Each run's best-performing model (lowest `test_rmse`) is tagged `run_champion` in MLflow and
recorded in `reports/<pipeline>/<run_id>_evaluation.yaml` as a top-level `run_champion` key.

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

### Configuration

All config in `config/` directory, validated via Pydantic models:

- `config/base/defaults.yaml` — Shared defaults (retries, MLflow URI) inherited by all pipelines
- `config/<pipeline>/orchestration.yaml` — Per-pipeline DAG settings (dag_id, schedule, directories, `reports_base_url`); `tasks.retries` / `retry_delay_minutes`; `tasks.enabled.profile` / `unsupervised_explore` / `drift_report` — set any to `false` to drop that optional task from the DAG
- `config/<pipeline>/pipeline.yaml` — Sources, target, problem type, split ratio; `validation.sentinel_values` (dataset-specific missing-value strings); `validation.per_file_schemas` (per-file required columns and bounds); `unsupervised` (enable/disable PCA and clustering, set `max_k`)
- `config/<pipeline>/cleaning.yaml` — Data cleaning recipes (impute strategy, protect columns, drop patterns)
- `config/<pipeline>/features.yaml` — Feature engineering (encoding, join strategy, NZV filter, VIF threshold, scaling)
- `config/<pipeline>/models.yaml` — Model hyperparameters (linear, LightGBM)

Active pipelines: `biomedical_clinical` (@weekly), `bioinfo_gene` (@monthly).

**Key Feature:** `src/dags/dag_factory.py` auto-discovers pipeline directories and registers one Airflow DAG per pipeline. Add a new pipeline by dropping a new `config/<name>/` directory — no Python changes needed.

### Testing

**For comprehensive end-to-end testing guide, see [TESTING.md](TESTING.md).**

Unit tests:
```bash
uv sync
uv run pytest tests/test_pipeline.py -v   # 31 tests: config, validate, clean, features, profile
```

The TESTING.md guide covers:
- Triggering the full DAG pipeline
- Validating outputs at each stage
- Checking MLflow model training and registration
- Promoting models to Production
- Testing FastAPI `/health` and `/predict` endpoints
- Batch prediction validation
- Troubleshooting common issues

### Debugging & Diagnostics

**For step-by-step pipeline debugging guide, see [DIAGNOSTICS.md](DIAGNOSTICS.md).**

Quick diagnostic check: read Parquet outputs and query MLflow directly (see DIAGNOSTICS.md
for copy-pasteable snippets), or use the MLflow UI's Runs tab to compare metrics across runs.

The DIAGNOSTICS.md guide covers:
- Understanding baseline vs model performance
- Debugging each of 9 pipeline stages
- Data quality checks at each step
- Model performance interpretation (RMSE, improvement %)
- Feature engineering validation
- Configuration debugging
- MLflow integration troubleshooting
- Red flags and common issues

### Development

#### Running the DAG

1. Open Airflow UI: http://localhost:8080
2. Find `biomedical_clinical_pipeline` or `bioinfo_gene_pipeline`
3. Click **Trigger DAG** (or wait for next scheduled run)
4. Monitor task execution in the Graph view

#### Monitoring

- **MLflow Tracking:** http://localhost:5001 — browse runs, metrics, artifacts
- **Profile & Drift Reports:** http://localhost:8888 — nginx directory listing; tasks `03_profile_data`, `06b_unsupervised_explore`, and `09_drift_report` each have a "Docs" tab with a direct link
- **Logs:** `docker logs airflow-scheduler` or Airflow UI task logs

## Stack

- **Orchestration:** Apache Airflow 3 (LocalExecutor)
- **Data Validation:** Pandera (schemas at boundaries)
- **Profiling:** ydata-profiling (per-source HTML reports)
- **Modeling:** scikit-learn (linear) + LightGBM (gradient boosting)
- **ML Tracking:** MLflow (Postgres backend, local artifacts)
- **Drift Monitoring:** Evidently AI
- **Serving:** FastAPI (model inference)
- **Config:** Pydantic + YAML
- **Storage:** Local Parquet, bind-mounted Docker volumes
- **Database:** PostgreSQL 16 (separate instances for Airflow & MLflow)

## Demo Pipelines

Each pipeline is just a `config/<name>/` directory — the code in `src/` is shared and unaware
of which dataset it's processing.

### `biomedical_clinical` — CMS Hospital Compare (primary demo)

Multi-source CSV data, chosen because it needs the full range of the template's features:
- **Rows:** ~4,800+ hospitals
- **Columns:** Quality measures, HCAHPS scores, safety grades
- **Target:** `ExcessReadmissionRatio` (pneumonia readmissions, continuous)
- **Exercises:** pivot-join assembly across files, Box-Cox target transform, VIF pruning, all 6 model types, all 3 optional tasks

Place CSV or Parquet files in `data/biomedical_clinical/landing/` before running the DAG.

### `bioinfo_gene` — gene expression (second demo)

A deliberately different domain, added to prove the config-driven claim rather than to be
fully tuned:
- **Target:** `expression_level` (continuous)
- **Exercises:** a leaner config (2 models, no join strategy), profiling on but clustering/drift
  off via `tasks.enabled` — showing the same DAG factory adapts without touching Python

Place CSV files in `data/bioinfo_gene/landing/` before running its DAG.

## Key Design Decisions

1. **One DAG, task groups:** Single orchestration DAG with 9 task groups for clarity
2. **Manifest versioning:** `src/utils/io.py` consolidates all I/O helpers (readers, writers, manifest read/write, path resolution); every stage imports from here — no duplicated file logic
3. **Pandera schemas:** Config-driven validation at raw boundaries; `per_file_schemas` in `pipeline.yaml` allows per-file required columns and bounds without touching Python
4. **Config-driven sentinel values:** Dataset-specific missing-value strings (e.g. "Not Available") declared in `pipeline.yaml` under `validation.sentinel_values`; both Stage 2 (validate) and Stage 4 (clean) read from the same source
5. **Pivot-join feature assembly:** `features.yaml` join strategy config filters and pivots multi-source files onto a spine; no code changes needed to add join sources
6. **VIF pruning is optional:** Set `vif_threshold: null` in `features.yaml` to skip VIF pruning for datasets with intentionally correlated predictors (e.g. HCAHPS survey questions)
7. **Config-driven unsupervised exploration:** Stage 06b algorithm (kmeans/skip), PCA toggle, and `max_k` are all in `pipeline.yaml`; different datasets can disable or tune without touching code
8. **Config-driven optional tasks:** `profile`, `unsupervised_explore`, and `drift_report` are toggled per pipeline via `tasks.enabled` in `orchestration.yaml`; `dag_factory.py` only creates and wires a task when its flag is `true` — no Python changes needed to trim the DAG for a new pipeline
9. **No auto-promotion:** Manual MLflow UI click to move models to Production
10. **LocalExecutor:** Single-machine orchestration (suitable for POC)
11. **Separate Postgres:** Airflow metadata and MLflow tracking use distinct databases
12. **Read-only MLflow artifacts:** FastAPI mounts artifacts as read-only
13. **On-demand drift:** Drift monitoring runs inside training DAG as final optional task
14. **Reports server:** nginx container at `:8888` serves `reports/` with directory listing; URL per pipeline set in `orchestration.yaml` (`reports_base_url`); Airflow task "Docs" tab links directly to it

## Troubleshooting

### Airflow webserver not accessible
```bash
docker logs airflow-webserver
docker compose restart airflow-webserver
```

### MLflow models not visible
```bash
curl http://localhost:5000/health
# If unhealthy, check:
docker logs mlflow-server
```

### Data validation failures
Check schema definitions in `src/schemas/` and ensure raw/feature data matches.

### Running tests fails
```bash
uv sync  # Ensure all dependencies installed
uv run pytest tests/ -v  # Full verbose output
```

## Next Steps

1. Place sample Hospital Compare CSVs in `data/biomedical_clinical/landing/` (or your own data in
   a new `config/<pipeline>/landing/` — see [Demo Pipelines](#demo-pipelines) for what a second
   pipeline needs)
2. Follow the [TESTING.md](TESTING.md) guide to:
   - Trigger `biomedical_clinical_pipeline` DAG in Airflow
   - Validate outputs at each stage (raw → interim → features → models)
   - Promote a trained model to Production in MLflow
   - Test FastAPI `/predict` endpoint with sample data
3. Monitor in UIs:
   - **Airflow UI**: http://localhost:8080 (DAG progress)
   - **MLflow UI**: http://localhost:5000 (model metrics & registry)
   - **FastAPI Docs**: http://localhost:8000/docs (interactive API testing)
4. Scale and deploy:
   - Move to Kubernetes for production deployment
   - Set up monitoring dashboards (Grafana)
   - Automate scheduled retraining runs
