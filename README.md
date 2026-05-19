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
   docker-compose up -d
   ```

3. **Verify services are healthy:**
   ```bash
   docker-compose ps
   ```

   All 7 services should show `Healthy` or `Running`:
   - Airflow Postgres (5432)
   - MLflow Postgres (5433)
   - Airflow Webserver (8080)
   - Airflow Scheduler
   - Airflow Triggerer
   - MLflow Server (5000)
   - FastAPI Server (8000)

4. **Install dependencies:**
   ```bash
   uv sync
   ```

### Access UIs

| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | N/A |
| FastAPI | http://localhost:8000/docs | N/A |

## Pipeline Architecture

### Stages (9 tasks in sequence)

1. **ingest** — Move files from `data/landing` to `data/raw/<run_id>`, write `manifest.yaml`
2. **validate_raw** — Pandera schema validation on raw data per source
3. **profile** — ydata-profiling HTML reports per source
4. **clean** — Type coercion, missing value handling, deduplication → `data/interim/<run_id>`
5. **feature_engineer** — Joins, encoding, NZV filtering, train/test split → `data/features/<run_id>`
6. **validate_features** — Pandera schema check on feature matrix
7. **train** — sklearn linear + LightGBM models, log metrics to MLflow
8. **register** — Register both models to MLflow Staging (no auto-promotion)
9. **drift_report** — Evidently AI drift monitoring (current vs previous training set)

### Data Flow

```
data/landing
    ↓ [ingest]
data/raw/<run_id>/manifest.yaml
    ↓ [validate_raw] [profile]
data/interim/<run_id>
    ↓ [clean]
data/features/<run_id>/
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

- `orchestration.yaml` — DAG parameters (schedule, owner, retries, directories, MLflow URI)
- `pipeline.yaml` — Sources, target, problem type, split ratio
- `cleaning.yaml` — Data cleaning recipes (type coercion, missing handling, dedup)
- `features.yaml` — Feature engineering (encoding, polynomial features, scaling)
- `models.yaml` — Model hyperparameters (linear, LightGBM)

**Key Feature:** All DAG orchestration settings are in `orchestration.yaml` - change MLflow URI, directories, or schedule without touching Python code.

### Testing

**For comprehensive end-to-end testing guide, see [TESTING.md](TESTING.md).**

Unit tests:
```bash
# Unit tests + config validation (requires venv/uv)
uv sync
uv run pytest tests/test_config.py -v

# Schema validation tests
uv run pytest tests/test_schemas.py -v

# Serving tests
uv run pytest tests/test_serve.py -v
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

Quick diagnostic check:
```bash
# Run comprehensive diagnostics
python3 scripts/diagnose_pipeline.py

# Analyze model performance
python3 scripts/analyze_models.py

# Or in Docker environment
docker-compose exec airflow-scheduler python3 /path/to/script.py
```

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
2. Find `ml_pipeline` DAG
3. Click **Trigger DAG** (or wait for next scheduled run)
4. Monitor task execution in the Graph view

#### Monitoring

- **MLflow Tracking:** http://localhost:5001 — browse runs, metrics, artifacts
- **Drift Reports:** Check `reports/` directory after `drift_report` task completes
- **Logs:** `docker logs airflow-scheduler` or Airflow UI task logs

## Stack

- **Orchestration:** Apache Airflow 2.10 (LocalExecutor)
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

Place CSV files in `data/landing/` before running the DAG.

## Key Design Decisions

1. **One DAG, task groups:** Single orchestration DAG with 9 task groups for clarity
2. **Manifest versioning:** `manifest.yaml` at each storage boundary (ingest, clean, features)
3. **Pandera schemas:** Strict validation at raw → clean → features → train
4. **No auto-promotion:** Manual MLflow UI click to move models to Production
5. **LocalExecutor:** Single-machine orchestration (suitable for POC)
6. **Separate Postgres:** Airflow metadata and MLflow tracking use distinct databases
7. **Read-only MLflow artifacts:** FastAPI mounts artifacts as read-only
8. **On-demand drift:** Drift monitoring runs inside training DAG as final task

## Troubleshooting

### Airflow webserver not accessible
```bash
docker logs airflow-webserver
docker-compose restart airflow-webserver
```

### MLflow models not visible
```bash
curl http://localhost:5001/health
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

1. Place sample Hospital Compare CSVs in `data/landing/`
2. Follow the [TESTING.md](TESTING.md) guide to:
   - Trigger the complete `ml_pipeline` DAG in Airflow
   - Validate outputs at each stage (raw → interim → features → models)
   - Promote a trained model to Production in MLflow
   - Test FastAPI `/predict` endpoint with sample data
3. Monitor in UIs:
   - **Airflow UI**: http://localhost:8080 (DAG progress)
   - **MLflow UI**: http://localhost:5001 (model metrics & registry)
   - **FastAPI Docs**: http://localhost:8000/docs (interactive API testing)
4. Scale and deploy:
   - Move to Kubernetes for production deployment
   - Set up monitoring dashboards (Grafana)
   - Automate scheduled retraining runs
