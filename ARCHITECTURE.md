# ML Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE POC ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────────────┐
                           │   Data Sources       │
                           │  (CSV in landing/)   │
                           └──────────┬───────────┘
                                      │
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │                    ORCHESTRATION LAYER                     │
        │                    (Apache Airflow 3)                      │
        │  ┌──────────────────────────────────────────────────────┐  │
        │  │  ml_pipeline DAG (9 stages)                          │  │
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
        │ ├─ data/raw/          │        │ ├─ Postgres  │    │
        │ │  2026-05-18/        │        │ ├─ Artifacts │    │
        │ │  └─ *.csv           │        │ │  Registry  │    │
        │ │  └─ manifest.yaml   │        │ └─ Models    │    │
        │ │                     │        │   (linear)   │    │
        │ ├─ data/interim/      │        │   (gbm)      │    │
        │ │  2026-05-18/        │        │              │    │
        │ │  └─ *.csv (clean)   │        └──────────────┘    │
        │ │  └─ manifest.yaml   │                            │
        │ │                     │        ┌────────────────┐  │
        │ ├─ data/features/     │        │  MONITORING    │  │
        │ │  2026-05-18/        │        │  (Evidently)   │  │
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
data/landing/*.csv
        │
        ├─→ Read CSV files
        ├─→ Compute SHA256 hashes
        ├─→ Create versioned directory
        │   data/raw/2026-05-18/
        │
        ├─→ Write manifest.yaml
        │   ├─ file sizes
        │   ├─ checksums
        │   └─ timestamp
        │
        └─→ Output: Raw data + manifest


STAGE 2: VALIDATE_RAW
───────────────────────────────────────────────────────
data/raw/2026-05-18/*.csv
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
data/raw/2026-05-18/*.csv
        │
        ├─→ Load each CSV
        ├─→ Generate ydata-profiling reports
        │   ├─ Data types
        │   ├─ Missing values
        │   ├─ Distributions
        │   └─ Correlations
        │
        └─→ Output: HTML reports in reports/


STAGE 4: CLEAN
───────────────────────────────────────────────────────
data/raw/2026-05-18/*.csv
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
        ├─→ Write to interim
        │   data/interim/2026-05-18/*.csv
        │
        └─→ Output: Cleaned data + manifest


STAGE 5: FEATURE_ENGINEER
───────────────────────────────────────────────────────
data/interim/2026-05-18/*.csv
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
        ├─→ data/features/2026-05-18/
        │   ├─ train.parquet
        │   ├─ test.parquet
        │   └─ manifest.yaml
        │
        └─→ Output: Feature matrices ready for training


STAGE 6: VALIDATE_FEATURES
───────────────────────────────────────────────────────
data/features/2026-05-18/train.parquet
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
data/features/2026-05-18/{train,test}.parquet
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
data/features/2026-05-18/train.parquet
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
All pipeline orchestration and processing parameters are externalized to YAML configs:

config/orchestration.yaml (NEW - controls Airflow DAG behavior)
├─ dag: DAG name, owner, description, schedule, catchup policy
├─ tasks: retry count, retry delay, specific task overrides
├─ directories: data landing, raw, interim, features, reports
└─ mlflow: tracking URI (http://mlflow-server:5000)

config/pipeline.yaml (target, sources, problem type, split ratio)
config/cleaning.yaml (type coercion, missing value handling)
config/features.yaml (encoding strategies, polynomial features, scaling)
config/models.yaml (hyperparameters for Ridge, LightGBM)

All configs loaded via Pydantic models in src/utils/config.py
- load_orchestration_config() → controls dags/pipeline.py behavior
- load_pipeline_config() → defines target, sources, problem type
- load_cleaning_config() → data cleaning recipes
- load_features_config() → feature engineering recipes
- load_models_config() → model hyperparameters

Benefits:
✓ No hardcoded parameters in Python code
✓ Environment-specific overrides via environment variables
✓ Type-safe validation at load time
✓ Single source of truth for all parameters
```

## Manifest.yaml Versioning

```
Each stage writes manifest.yaml for data lineage:

data/raw/2026-05-18/manifest.yaml
├─ run_id: 2026-05-18
├─ source_directory: data/landing
├─ files:
│  ├─ hospital_compare_sample.csv
│  │  ├─ size_bytes: 5847
│  │  ├─ hash_sha256: a1b2c3...
│  │  └─ timestamp: 2026-05-18T21:42:00

data/interim/2026-05-18/manifest.yaml
├─ run_id: 2026-05-18
├─ stage: clean
├─ files:
│  └─ hospital_compare_sample.csv
│     └─ output_path: data/interim/2026-05-18/...

data/features/2026-05-18/manifest.yaml
├─ run_id: 2026-05-18
├─ stage: feature_engineer
├─ train:
│  ├─ path: data/features/2026-05-18/train.parquet
│  ├─ rows: 16
│  └─ columns: 26
├─ test:
│  ├─ path: data/features/2026-05-18/test.parquet
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
│ Scenario: CSV file missing in data/landing           │
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
│ └─→ Pandas read_csv() raises MemoryError             │
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
│  ├─ data/landing, raw, interim, features │
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
