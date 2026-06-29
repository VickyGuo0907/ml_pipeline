# ML Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE POC ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────────────┐
                           │   Data Sources       │
                           │  (CSV/Parquet in     │
                           │   landing/)          │
                           └──────────┬───────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │                    ORCHESTRATION LAYER                     │
        │                    (Apache Airflow 3)                      │
        │  ┌──────────────────────────────────────────────────────┐  │
        │  │  dag_factory → one DAG per config/pipelines/ dir     │  │
        │  │  ┌─────────┬──────────┬─────────┬───────┬──────────┐ │  │
        │  │  │ Ingest  │ Validate │ Profile │ Clean │ Features │ │  │
        │  │  └────┬────┴────┬─────┴────┬────┴───┬───┴────┬─────┘ │  │
        │  │       │         │          │        │        │       │  │
        │  │       └─────────┴──────────┴────────┴────────┘       │  │
        │  │                            │                         │  │
        │  │  ┌──────────────────────────▼────────────────────┐   │  │
        │  │  │  Validate Features │ Train │ Register │ Drift │   │  │
        │  │  └────────────────────────────────────────────────┘  │  │
        │  └──────────────────────────────────────────────────────┘  │
        │         │                              │           │       │
        │         ▼                              ▼           ▼       │
        └─────────┼──────────────────────────────┼───────────┼───────┘
                  │                              │           │
        ┌─────────▼─────────────┐        ┌──────▼───────┐    │
        │  DATA LAKE            │        │  ML Tracking │    │
        │  (Local Parquet)      │        │  (MLflow)    │    │
        │                       │        │              │    │
        │ ├─ data/<pipeline>/raw/│        │ ├─ Postgres  │    │
        │ │  <run_id>/          │        │ ├─ Artifacts │    │
        │ │  └─ *.csv / *.parquet│        │ │  Registry  │    │
        │ │  └─ manifest.yaml   │        │ └─ Models    │    │
        │ │                     │        │   (linear)   │    │
        │ ├─ data/<pipeline>/   │        │   (gbm)      │    │
        │ │  interim/<run_id>/  │        │              │    │
        │ │  └─ *.csv / *.parquet│       └──────────────┘    │
        │ │  └─ manifest.yaml   │                            │
        │ │                     │        ┌────────────────┐  │
        │ ├─ data/<pipeline>/   │        │  MONITORING    │  │
        │ │  features/<run_id>/ │        │  (Evidently)   │  │
        │ │  ├─ train.parquet   │        │                │  │
        │ │  ├─ test.parquet    │        │ └─ Drift       │  │
        │ │  └─ manifest.yaml   │        │    Reports     │  │
        │ │                     │        │    (HTML)      │  │
        │ └─ reports/           │        └────────────────┘  │
        │    └─ *.html          │                            │
        │       (profiling,     │                            │
        │        drift)         │                            │
        └───────────────────────┘                            │
                                                             │
                    ┌────────────────────────────────────────┘
                    │
        ┌───────────▼──────────────┐
        │   MODEL SERVING LAYER    │
        │   (FastAPI + MLflow)     │
        │                          │
        │  ┌────────────────────┐  │
        │  │  FastAPI Server    │  │
        │  │  :8000/predict     │  │
        │  │  :8000/health      │  │
        │  └────────────────────┘  │
        │           │              │
        │           ▼              │
        │  ┌────────────────────┐  │
        │  │ Load MLflow Model  │  │
        │  │ (Staging stage)    │  │
        │  │                    │  │
        │  │ • Linear Baseline  │  │
        │  │ • LightGBM (Best)  │  │
        │  └────────────────────┘  │
        └──────────────────────────┘
```

---

## Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose (7 Services)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────┐                       │
│  │  AIRFLOW CLUSTER                     │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ airflow-webserver (:8080)      │  │                       │
│  │  │ • Flask UI                     │  │                       │
│  │  │ • DAG Management               │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ airflow-scheduler              │  │                       │
│  │  │ • Triggers DAG runs            │  │                       │
│  │  │ • Monitors tasks               │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ airflow-triggerer              │  │                       │
│  │  │ • Event-based triggers         │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
│  ┌──────────────────────────────────────┐                       │
│  │  DATABASE TIER                       │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ airflow-postgres (5432)        │  │                       │
│  │  │ • Airflow metadata DB          │  │                       │
│  │  │ • DAG runs, task logs          │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ mlflow-postgres (5433)         │  │                       │
│  │  │ • MLflow tracking DB           │  │                       │
│  │  │ • Runs, metrics, params        │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
│  ┌──────────────────────────────────────┐                       │
│  │  ML SERVICES                         │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ mlflow-server (:5000)          │  │                       │
│  │  │ • Model registry               │  │                       │
│  │  │ • Artifact storage             │  │                       │
│  │  │ • Metrics tracking             │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  │  ┌────────────────────────────────┐  │                       │
│  │  │ fastapi-server (:8000)         │  │                       │
│  │  │ • Model inference              │  │                       │
│  │  │ • Health checks                │  │                       │
│  │  └────────────────────────────────┘  │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Pipeline (Stage Detail)

```
STAGE 1: INGEST
───────────────────────────────────────────────────────
data/<pipeline>/landing/*.csv / *.parquet   (e.g. data/biomedical_clinical/landing/)
        │
        ├─→ Read files (dispatcher: CSV, Parquet)
        ├─→ Compute SHA256 hashes
        ├─→ Create versioned directory
        │   data/<pipeline>/raw/<run_id>/
        │
        ├─→ Write manifest.yaml
        │   ├─ file sizes
        │   ├─ checksums
        │   └─ timestamp
        │
        └─→ Output: Raw data + manifest


STAGE 2: VALIDATE_RAW
───────────────────────────────────────────────────────
data/<pipeline>/raw/<run_id>/*.csv / *.parquet
        │
        ├─→ Load manifest
        ├─→ Apply Pandera schema
        │   ├─ Type validation (int, float, str)
        │   ├─ Nullable checks
        │   └─ Column existence
        │
        └─→ Output: Validation report


STAGE 3: PROFILE
───────────────────────────────────────────────────────
data/<pipeline>/raw/<run_id>/*.csv / *.parquet
        │
        ├─→ Load each file (dispatcher: CSV, Parquet)
        ├─→ Generate ydata-profiling reports
        │   ├─ Data types
        │   ├─ Missing values
        │   ├─ Distributions
        │   └─ Correlations
        │
        └─→ Output: HTML reports in reports/


STAGE 4: CLEAN
───────────────────────────────────────────────────────
data/<pipeline>/raw/<run_id>/*.csv / *.parquet
        │
        ├─→ Type coercion
        │   └─ Convert numeric columns
        │
        ├─→ Missing value handling
        │   └─ Drop columns with >50% missing
        │
        ├─→ Deduplication
        │   └─ Remove exact duplicates
        │
        ├─→ Write to interim (format preserved: CSV→CSV, Parquet→Parquet)
        │   data/<pipeline>/interim/<run_id>/*.csv / *.parquet
        │
        └─→ Output: Cleaned data + manifest


STAGE 5: FEATURE_ENGINEER
───────────────────────────────────────────────────────
data/<pipeline>/interim/<run_id>/*.csv / *.parquet
        │
        ├─→ Load all cleaned files
        ├─→ Concatenate
        │
        ├─→ Encoding
        │   ├─ Label encode categoricals
        │   ├─ One-hot encode comparisons
        │   └─ Drop original encoded cols
        │
        ├─→ Feature selection
        │   └─ Drop NZV columns
        │   └─ Drop specified columns
        │
        ├─→ Scaling
        │   └─ StandardScaler on numeric features
        │
        ├─→ Train/Test Split
        │   ├─ 80/20 split
        │   ├─ Stratified on target
        │   └─ Save as Parquet
        │
        ├─→ data/<pipeline>/features/<run_id>/
        │   ├─ train.parquet
        │   ├─ test.parquet
        │   └─ manifest.yaml
        │
        └─→ Output: Feature matrices ready for training


STAGE 6: VALIDATE_FEATURES
───────────────────────────────────────────────────────
data/<pipeline>/features/<run_id>/train.parquet
        │
        ├─→ Load feature matrix
        ├─→ Apply Pandera features schema
        │   ├─ Type validation
        │   ├─ Column presence
        │   └─ Numeric bounds
        │
        └─→ Output: Validation passed


STAGE 7: TRAIN
───────────────────────────────────────────────────────
data/<pipeline>/features/<run_id>/{train,test}.parquet
        │
        ├─→ Load config (models.yaml)
        │
        ├─→ Train Linear Baseline (Ridge)
        │   ├─ Fit on train
        │   ├─ Predict on test
        │   ├─ Log metrics (MSE, RMSE, train_rmse, test_rmse)
        │   └─ Log to MLflow run
        │
        ├─→ Train LightGBM
        │   ├─ Fit on train
        │   ├─ Predict on test
        │   ├─ Log metrics
        │   └─ Log to MLflow run
        │
        └─→ Output: MLflow run IDs via XCom


STAGE 8: REGISTER & EVALUATE
───────────────────────────────────────────────────────
MLflow runs (from train stage)
        │
        ├─→ Extract metrics from runs
        │   ├─ test_rmse, train_rmse, test_mse
        │   └─ Calculate baseline RMSE for performance comparison
        │
        ├─→ Register to Model Registry
        │   ├─ Model: linear_baseline (v1)
        │   └─ Model: lightgbm_gbm (v1)
        │
        ├─→ Add comprehensive tags to each model
        │   ├─ Lifecycle: stage (staging), registered_by, registered_at (ISO timestamp)
        │   ├─ Performance: test_rmse, train_rmse, test_mse (formatted)
        │   ├─ Metadata: model_type, run_id, environment (development)
        │   └─ Status: active
        │
        ├─→ Update model version with description
        │   └─ Human-readable: model name, type, RMSE, registration timestamp
        │
        ├─→ Move to "Staging" stage
        │   └─ Ready for testing (no auto-promotion to Production)
        │
        └─→ Output: Tagged models in MLflow Staging


STAGE 9: DRIFT_REPORT
───────────────────────────────────────────────────────
data/<pipeline>/features/<run_id>/train.parquet
        │
        ├─→ Compare current to previous run
        │   (if previous_run_id available)
        │
        ├─→ Generate Evidently AI report
        │   ├─ DatasetDriftMetric
        │   ├─ Statistical drift detection
        │   └─ Feature-level analysis
        │
        ├─→ Save HTML report
        │   reports/2026-05-18_drift_report.html
        │
        └─→ Output: Drift analysis + HTML
```

---

## Technology Stack

```
┌──────────────────────────────────────────────────────┐
│ ORCHESTRATION & WORKFLOW                             │
├──────────────────────────────────────────────────────┤
│ • Apache Airflow 2.10.0                              │
│   └─ DAG-based orchestration, LocalExecutor          │
│   └─ Task dependency management                      │
│   └─ Monitoring & alerting via webserver             │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ DATA VALIDATION & PROFILING                          │
├──────────────────────────────────────────────────────┤
│ • Pandera 0.18.x                                     │
│   └─ Schema validation at storage boundaries         │
│   └─ Type, nullable, and constraint checks           │
│                                                      │
│ • ydata-profiling 4.x                                │
│   └─ Automated EDA reports                           │
│   └─ Distribution analysis, missing values           │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ DATA MANIPULATION & FEATURE ENGINEERING              │
├──────────────────────────────────────────────────────┤
│ • Pandas 2.1.x                                       │
│   └─ DataFrame operations, merging, grouping         │
│                                                      │
│ • Scikit-learn 1.3.x                                 │
│   └─ Preprocessing, model training                   │
│   └─ StandardScaler, LabelEncoder                    │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ MODELING & TRAINING                                  │
├──────────────────────────────────────────────────────┤
│ • Scikit-learn Ridge Regression                      │
│   └─ Regularized linear baseline                     │
│                                                      │
│ • LightGBM 4.x                                       │
│   └─ Gradient boosting, fast training                │
│   └─ Feature importance, native categorical support  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ ML TRACKING & MODEL REGISTRY                         │
├──────────────────────────────────────────────────────┤
│ • MLflow 2.10.x                                      │
│   └─ Experiment tracking (metrics, params, artifacts)│
│   └─ Model registry (Staging/Production)             │
│   └─ PostgreSQL backend + local artifact store       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ DRIFT MONITORING                                     │
├──────────────────────────────────────────────────────┤
│ • Evidently AI 0.4.x                                 │
│   └─ DatasetDriftMetric for feature distributions    │
│   └─ HTML report generation                          │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ SERVING & API                                        │
├──────────────────────────────────────────────────────┤
│ • FastAPI 0.104.x                                    │
│   └─ RESTful API, /predict endpoint                  │
│   └─ Health checks, OpenAPI docs                     │
│                                                      │
│ • Uvicorn 0.24.x                                     │
│   └─ ASGI server for FastAPI                         │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ CONFIGURATION & SECRETS                              │
├──────────────────────────────────────────────────────┤
│ • Pydantic 2.5.x                                     │
│   └─ Config validation, type safety                  │
│                                                      │
│ • python-dotenv                                      │
│   └─ Environment variable loading (.env)             │
│                                                      │
│ • YAML                                               │
│   └─ Pipeline, cleaning, features, models configs    │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ STORAGE                                              │
├──────────────────────────────────────────────────────┤
│ • Parquet (Apache)                                   │
│   └─ Columnar format, compression                    │
│   └─ Feature matrices (train/test)                   │
│                                                      │
│ • PostgreSQL 16 (x2)                                 │
│   └─ Airflow metadata DB                             │
│   └─ MLflow tracking DB (separate instance)          │
│                                                      │
│ • Local filesystem (bind-mounted volumes)            │
│   └─ Raw data, interim, features, reports            │
│   └─ MLflow artifacts                                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ DEPENDENCY MANAGEMENT                                │
├──────────────────────────────────────────────────────┤
│ • uv (Rust-based, 90% faster than pip)               │
│   └─ Deterministic lock file (uv.lock)               │
│   └─ 309 packages pinned to exact versions           │
└──────────────────────────────────────────────────────┘
```

---

## XCom Data Passing (Airflow)

```
run_id (Airflow logical_date: 2026-05-18)
        │
        ├─→ Pushed by: 01_ingest_files
        │   Key: "run_id"
        │   Value: "2026-05-18"
        │
        ├─→ Used by all downstream tasks:
        │   • 02_validate_raw_schema
        │   • 03_profile_data
        │   • 04_clean_data
        │   • 05_engineer_features
        │   • 06_validate_features_schema
        │   • 07_train_models
        │   • 09_drift_report
        │
        └─→ Ensures all tasks operate on same data version


mlflow_run_ids (MLflow tracking)
        │
        ├─→ Pushed by: 07_train_models
        │   Key: "mlflow_run_ids"
        │   Value: {
        │     "linear_baseline": "abc123...",
        │     "lightgbm_gbm": "def456..."
        │   }
        │
        └─→ Used by: 08_register_to_mlflow
            └─ Retrieves run artifacts for registration
```

---

## Configuration Management

```
All pipeline orchestration and processing parameters are externalized to YAML configs.
Configs are split into two layers: shared base defaults and per-pipeline config directories.
src/dags/dag_factory.py discovers all pipeline directories and registers one Airflow DAG each.

config/base/defaults.yaml (shared across all pipelines)
├─ tasks: retry count, retry delay, per-task overrides
└─ mlflow: tracking URI (http://mlflow-server:5000)

config/<pipeline>/ (one directory per pipeline, e.g. biomedical_clinical, bioinfo_gene)
├─ orchestration.yaml  (dag_id, schedule, tags, data directories — overrides base defaults)
├─ pipeline.yaml       (target, sources, problem type, split ratio)
├─ cleaning.yaml       (type coercion, missing value handling)
├─ features.yaml       (encoding strategies, polynomial features, scaling)
└─ models.yaml         (hyperparameters for Ridge, LightGBM)

Adding a new pipeline: drop a new config/<name>/ directory with orchestration.yaml.
dag_factory.py picks it up automatically on next Airflow parse — no Python changes needed.

All configs loaded via Pydantic models in src/utils/config.py
- discover_pipelines() → scans config/ for pipeline directories
- load_pipeline_orchestration_config() → merges base defaults + pipeline overrides
- load_pipeline_config(config_dir) → defines target, sources, problem type
- load_cleaning_config(config_dir) → data cleaning recipes
- load_features_config(config_dir) → feature engineering recipes
- load_models_config(config_dir) → model hyperparameters

Benefits:
✓ No hardcoded parameters in Python code
✓ Base defaults DRY — only overrides live in pipeline configs
✓ Type-safe validation at load time
✓ Single source of truth for all parameters
✓ New pipelines added as config dirs, zero code changes
```

## Manifest.yaml Versioning

```
Each stage writes manifest.yaml for data lineage:

data/biomedical_clinical/raw/<run_id>/manifest.yaml
├─ run_id: <run_id>
├─ source_directory: data/biomedical_clinical/landing
├─ files:
│  ├─ hospital_compare_sample.csv
│  │  ├─ size_bytes: 5847
│  │  ├─ hash_sha256: a1b2c3...
│  │  └─ timestamp: 2026-05-18T21:42:00

data/biomedical_clinical/interim/<run_id>/manifest.yaml
├─ run_id: <run_id>
├─ stage: clean
├─ files:
│  └─ hospital_compare_sample.csv
│     └─ output_path: data/biomedical_clinical/interim/<run_id>/...

data/biomedical_clinical/features/<run_id>/manifest.yaml
├─ run_id: <run_id>
├─ stage: feature_engineer
├─ train:
│  ├─ path: data/biomedical_clinical/features/<run_id>/train.parquet
│  ├─ rows: 16
│  └─ columns: 26
├─ test:
│  ├─ path: data/biomedical_clinical/features/<run_id>/test.parquet
│  ├─ rows: 4
│  └─ columns: 26
```

---

## Model Lifecycle

```
┌─────────────────────────────────────────────────────┐
│              MODEL LIFECYCLE (MLflow)               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  TRAINING PHASE                                     │
│  ──────────────                                     │
│  Model trained → MLflow Run Created                 │
│                 ├─ Metrics logged (RMSE, MSE)       │
│                 ├─ Parameters logged                │
│                 ├─ Artifact: model binary           │
│                 └─ Tags: model_name, run_id         │
│                                                     │
│  ↓                                                  │
│                                                     │
│  REGISTRATION PHASE                                 │
│  ───────────────────                                │
│  Model registered → Model Version (v1)              │
│  to Registry       ├─ Source Run ID linked          │
│                    ├─ Staging stage assigned        │
│                    └─ Ready for evaluation          │
│                                                     │
│  ↓                                                  │
│                                                     │
│  STAGING PHASE (On-Demand Testing)                  │
│  ──────────────────────────────────                 │
│  FastAPI loads model from Staging                   │
│  ├─ Performance validated                           │
│  ├─ Integration tested                              │
│  └─ Approval gates (manual)                         │
│                                                     │
│  ↓ (Manual promotion via MLflow UI)                 │
│                                                     │
│  PRODUCTION PHASE (Manual Click)                    │
│  ───────────────────────────────────                │
│  Model moved to Production stage                    │
│  ├─ Served by FastAPI                               │
│  ├─ Monitored for drift                             │
│  └─ Superseded by newer models                      │
│                                                     │
│  ↓ (On next training cycle)                         │
│                                                     │
│  ARCHIVAL PHASE                                     │
│  ────────────────                                   │
│  Model archived                                     │
│  └─ Retained for audit & rollback                   │
│                                                     │
└─────────────────────────────────────────────────────┘

NO AUTOMATIC PROMOTION TO PRODUCTION
────────────────────────────────────
Every model registers to Staging only.
Manual UI click required to move to Production.
This prevents surprises in production.
```

---

## Error Handling & Resilience

```
┌──────────────────────────────────────────────────────┐
│ FAILURE SCENARIOS & RECOVERY                         │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Scenario: No supported files in data/<pipeline>/landing│
│ ─────────────────────────────────────────────        │
│ └─→ ingest_files() raises FileNotFoundError          │
│     └─→ Task marked FAILED                           │
│     └─→ DAG halted (upstream dependency)             │
│     └─→ Retry (configured: 1 retry, 5min delay)      │
│                                                      │
│ Scenario: Schema validation fails                    │
│ ─────────────────────────────────────────────        │
│ └─→ validate_raw_files() raises SchemaError          │
│     └─→ Detailed error with column info              │
│     └─→ Task marked FAILED                           │
│     └─→ DAG halted (upstream dependency)             │
│                                                      │
│ Scenario: MLflow server unavailable                  │
│ ─────────────────────────────────────────────        │
│ └─→ MLflow client connection fails                   │
│     └─→ train_models() raises exception              │
│     └─→ Task marked FAILED                           │
│     └─→ Retry logic retries connection               │
│                                                      │
│ Scenario: Partition too large for memory             │
│ ─────────────────────────────────────────────        │
│ └─→ Dispatcher reader raises MemoryError             │
│     └─→ Task marked FAILED                           │
│     └─→ Review data size, increase container memory  │
│                                                      │
│ SUCCESS CRITERIA                                     │
│ ─────────────────                                    │
│ • All 9 tasks completed                              │
│ • manifests.yaml written at each boundary            │
│ • MLflow models registered to Staging                │
│ • FastAPI can load Staging model                     │
│ • Drift report generated (or baseline created)       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Deployment Topologies

```
LOCAL DEVELOPMENT (Current)
──────────────────────────
┌──────────────────────────────────────────┐
│ Docker Compose (7 containers)            │
│ ├─ Airflow (webserver, scheduler, etc)   │
│ ├─ PostgreSQL x2                         │
│ ├─ MLflow                                │
│ └─ FastAPI                               │
├─ Bind-mounted volumes                    │
│  ├─ data/<pipeline>/{landing,raw,interim,features} │
│  ├─ mlflow-artifacts                     │
│  └─ reports                              │
└─ All services on single machine          │
   └─ Suitable for POC / exploration

SCALE-OUT ARCHITECTURE (Future)
────────────────────────────────
┌─────────────────────────────────────────────────────────┐
│ Kubernetes Cluster                                      │
├─────────────────────────────────────────────────────────┤
│ ├─ Airflow (distributed, HA)                            │
│ │  └─ KubernetesExecutor / CeleryExecutor               │
│ │                                                       │
│ ├─ PostgreSQL (managed, replicated)                     │
│ │  └─ CloudSQL / RDS / managed service                  │
│ │                                                       │
│ ├─ MLflow (containerized, scalable)                     │
│ │  └─ S3 / GCS artifact storage                         │
│ │  └─ Managed database backend                          │
│ │                                                       │
│ ├─ FastAPI (autoscaling deployment)                     │
│ │  └─ Load balancer                                     │
│ │  └─ Horizontal pod autoscaling                        │
│ │                                                       │
│ └─ Data Lake (S3 / GCS / HDFS)                          │
│    └─ Partitioned by run_id / date                      │
│    └─ Lifecycle policies for cleanup                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow 2.10 | DAG scheduling, task management |
| **Validation** | Pandera 0.18 | Schema enforcement at boundaries |
| **Profiling** | ydata-profiling 4.x | Automated EDA reports |
| **Data Processing** | Pandas 2.1 | ETL operations |
| **Modeling** | scikit-learn + LightGBM | Linear baseline + gradient boosting |
| **Tracking** | MLflow 2.10 | Experiment tracking, model registry |
| **Monitoring** | Evidently AI 0.4 | Data drift detection |
| **Serving** | FastAPI 0.104 | Model inference API |
| **Storage** | PostgreSQL 16 + Parquet | Metadata + feature matrices |
| **Config** | Pydantic + YAML | Validated configuration |
| **Dependency Mgmt** | uv + pyproject.toml | 309 packages pinned |

**Key Design Principles:**
- ✅ Manifest-based versioning at every boundary
- ✅ Pandera validation gates prevent bad data
- ✅ XCom links pipeline stages via run_id
- ✅ MLflow tracks all experiments (no manual logging)
- ✅ Staging-only registration (no auto-promotion)
- ✅ Local storage for POC (scales to S3/GCS)
- ✅ All infrastructure as code (docker-compose)
