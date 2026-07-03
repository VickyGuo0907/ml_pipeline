# ML Pipeline POC

End-to-end machine learning pipeline demonstrating orchestration, validation, training, and serving at scale.

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

### Stages (7 core + 3 optional — stages 6 and 6b run in parallel)

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

## Dataset

CMS Hospital Compare data (multi-source CSV):
- **Rows:** ~4,800+ hospitals
- **Columns:** Quality measures, HCAHPS scores, safety grades
- **Target:** `ExcessReadmissionRatio` (pneumonia readmissions, continuous)

Place CSV or Parquet files in `data/biomedical_clinical/landing/` before running the DAG.

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

1. Place sample Hospital Compare CSVs in `data/biomedical_clinical/landing/`
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
