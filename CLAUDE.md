# ML Pipeline POC — Project Context

## Goal
Build an end-to-end ML pipeline POC on local docker-compose, designed to demonstrate viability for large-scale deployment. Generic tabular ML template, not tied to a specific use case. Currently exercising the pipeline with CMS Hospital Compare CSV data.

## Stack (locked)
- **Orchestrator:** Apache Airflow 3
- **Data validation:** pandera
- **EDA / profiling:** ydata-profiling
- **Data manipulation:** pandas
- **Storage:** Local Parquet, bind-mounted Docker volumes (no MinIO, no S3)
- **Versioning:** DVC for data, git for code
- **Experiment tracking + model registry:** MLflow with Postgres backend, local filesystem artifact store
- **Modeling:** scikit-learn + LightGBM (regularized linear baseline + gradient boosting)
- **Serving:** FastAPI
- **Drift monitoring:** Evidently AI (on-demand inside training DAG)
- **Config:** YAML files + pydantic models for validation
- **Dependencies:** uv (not pip, not poetry)
- **Secrets:** .env file (gitignored) + .env.example committed
- **Testing:** pandera schemas + unit tests + one integration test

## Design decisions (locked, do not revisit)
- **Config depth:** light — target column, problem type, sources in pipeline.yaml; cleaning/features/models YAMLs hold recipes called by code. Code is not fully generic.
- **DAG structure:** one Airflow DAG with task groups, not multiple DAGs
- **Model promotion:** every trained model registers to Staging. NO auto-promotion to Production — manual UI click only.
- **Drift monitoring:** on-demand, runs as final task in training DAG
- **Testing scope:** pandera at every storage boundary + unit tests for clean/feature functions + one end-to-end integration test
- **Host OS:** macOS

## Pipeline stages (8, one DAG)
1. ingest — move files from data/<pipeline>/landing to data/<pipeline>/raw/<run_id>, write manifest.yaml with checksums
2. validate_raw — pandera schema check per source file
3. profile — ydata-profiling HTML report per source
4. clean — type coercion, missing handling, dedup → data/<pipeline>/interim/<run_id>/*.parquet
5. feature_engineer — joins, encoding, NZV filter, train/test split → data/<pipeline>/features/<run_id>/{train,test}.parquet
6. validate_features — pandera schema check on feature matrix
7. train — sklearn + lightgbm, MLflow autolog, both models per run
8. evaluate_and_register — compute metrics, register to MLflow Staging
9. drift_report — Evidently HTML, compares current features to previous training set

Serving (FastAPI) runs as always-on container, NOT a DAG stage. Loads model tagged Production from MLflow registry.

## Repo layout
ml-pipeline/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── dags/dag_factory.py
│   ├── scripts/diagnose_pipeline.py, analyze_models.py
│   ├── ingest.py, validate.py, profile.py, clean.py
│   ├── features.py, train.py, evaluate.py, register.py
│   ├── serve.py, monitoring.py
│   ├── schemas/{raw.py, features.py}
│   └── utils/{config.py, io.py}
├── config/
│   ├── base/defaults.yaml
│   ├── biomedical_clinical/{orchestration.yaml, pipeline.yaml, cleaning.yaml, features.yaml, models.yaml}
│   └── bioinfo_gene/{orchestration.yaml, pipeline.yaml, cleaning.yaml, features.yaml, models.yaml}
├── data/
│   ├── biomedical_clinical/{landing,raw,interim,features}/
│   └── bioinfo_gene/{landing,raw,interim,features}/
├── mlflow-artifacts/
├── reports/
├── tests/{test_schemas.py, test_clean.py, test_features.py, test_integration.py}
└── notebooks/

## docker-compose services (7)
- airflow-webserver (:8080), airflow-scheduler, airflow-triggerer
- airflow-postgres (Airflow metadata DB)
- mlflow-server (:5000)
- mlflow-postgres (MLflow tracking DB, separate from Airflow's)
- fastapi (:8000)

Two Postgres instances to avoid schema mixing. FastAPI mounts mlflow-artifacts as read-only.

## Sample dataset
CMS Hospital Compare CSVs. Multi-file, real, messy. Files in data/biomedical_clinical/landing/.
Target column for the demo run: ExcessReadmissionRatio for pneumonia (continuous regression).
Predictors include quality measures, HCAHPS scores, and hospital info — see previous R-based exploration that produced a 4,802 × 30 feature matrix.

## Build order (do these in sequence; confirm each works before moving on)
1. docker-compose.yml + .env.example + skeleton dirs → all 7 services healthy
2. pyproject.toml + uv.lock → dependency set pinned
3. Pydantic config models + sample YAMLs → config loading works in isolation
4. pandera schemas (raw + features) → schemas importable and testable
5. DAG skeleton with stub tasks → Airflow UI shows full pipeline graph
6. Stage implementations in pipeline order: ingest → validate → profile → clean → features → validate features → train → evaluate/register → drift
7. FastAPI serving → /predict and /health working against registered model
8. Tests → schemas, units, one integration
9. README → clone-to-prediction path documented

## Hard rules
- Do NOT introduce Kubernetes, Feast, Optuna, Ray Tune, MinIO, or streaming components. They are explicitly out of scope.
- Do NOT auto-promote models to Production.
- Do NOT skip the manifest.yaml at storage boundaries. Versioning without manifests is a lie.
- Do NOT use pip or poetry. Use uv.
- Do NOT commit data/, mlflow-artifacts/, reports/, or .env. Gitignore them.

## Conventions
- Type hints on all function signatures.
- Pandera schemas validate every Parquet at boundaries.
- Manifests (YAML) written alongside every dated data output.
- Run IDs are ISO dates (e.g., 2026-05-15) or Airflow's logical_date.
## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
