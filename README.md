# ML Pipeline POC

![CI](https://github.com/VickyGuo0907/ml_pipeline/actions/workflows/ci.yml/badge.svg)

A config-driven ML pipeline template on local docker-compose: ingestion, validation, profiling,
cleaning, feature engineering, training, evaluation, registration and serving. Adding a dataset
means adding a `config/<pipeline>/` directory. `src/dags/dag_factory.py` discovers it and
registers a new Airflow DAG, with no Python changes.

## Quick Start

Prerequisites: Docker & Docker Compose, Python 3.11+, [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
docker compose up -d      # 8 services: Airflow (webserver, scheduler, triggerer, Postgres),
                          # MLflow (server, Postgres), FastAPI, reports server
docker compose ps         # all should be Healthy / Running
uv sync
```

| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | - |
| FastAPI | http://localhost:8000/docs | - |
| Reports | http://localhost:8888 | - |

Put data in `data/<pipeline>/landing/`, then trigger `<pipeline>_pipeline` in the Airflow UI.

## Pipeline

One DAG per pipeline:

1. **ingest:** landing → `raw/<run_id>`, with a `manifest.yaml` of checksums
2. **validate_raw:** pandera checks per source file (rules from `pipeline.yaml`)
3. **clean:** type coercion, sentinel values, imputation, dedup → `interim/<run_id>`
4. **feature_engineer:** joins, encoding, NZV filter, Box-Cox, VIF pruning, train/test split → `features/<run_id>`
5. **validate_features:** pandera checks on the feature matrix
6. **train:** every model in `models.yaml`, logged to MLflow
7. **register:** registers to MLflow Staging. **No auto-promotion**: Production is a manual UI click.

Optional, toggled in `orchestration.yaml` (`tasks.enabled`):
- `profile`: ydata-profiling reports
- `unsupervised_explore`: PCA and k-means
- `drift_report`: Evidently, comparing against the previous training set

`create_benchmark` stays a no-op unless the DAG is triggered with `conf={"refresh_benchmark": true}`.
Each run tags its best model `run_champion`. When a benchmark exists, each new model is also
checked against its Production version using bootstrapped RMSE confidence intervals. This check
is informational only and never changes what gets promoted.

Diagrams (open in a browser): `diagrams/01-architecture.html` through
`06-forecasting-architecture.html`.

## Configuration

All config is YAML, validated by Pydantic (`src/utils/config.py`):

| File | Holds |
|---|---|
| `config/base/defaults.yaml` | Shared defaults (retries, MLflow URI) |
| `orchestration.yaml` | DAG id, schedule, directories, optional-task toggles |
| `pipeline.yaml` | Sources, target, problem type, split, validation rules |
| `cleaning.yaml` | Imputation, protected columns, drop patterns |
| `features.yaml` | Encoding, joins, NZV/VIF, scaling |
| `models.yaml` | Models and hyperparameters, registration thresholds, champion metric, optional CV / diagnostics / feature importance |

## Demo Pipelines

| Pipeline | Problem | Notes |
|---|---|---|
| `biomedical_clinical` | CMS Hospital Compare, pneumonia excess readmission ratio | Primary demo: multi-source pivot-joins, Box-Cox, VIF, six model types |
| `bioinfo_gene` | Gene expression level | Same code, a different domain and a leaner config |
| `hospital_readmission_lagged` | Predict 2025 readmission ratio from 2024 measures | One-year-ahead forecast. A leakage-prone CMS field was removed via `drop_columns` (test R² ~0.22 → an honest ~0.05-0.07) |
| `pjm_load_forecast` | PJM East hourly load, 24h ahead | `problem_type: forecasting`, see below |
| `m6_returns_risk` | Monthly ETF/stock log-returns (M6) | `problem_type: finance` |

Each pipeline is just a `config/<name>/` directory. The code in `src/` is shared.

### `biomedical_clinical`

CMS Hospital Compare CSVs (~4,800 hospitals: quality measures, HCAHPS scores, safety grades),
predicting the pneumonia `ExcessReadmissionRatio`. It uses the full template: pivot-joins
across files, a Box-Cox target transform, VIF pruning, all six model types and all three
optional tasks. Put the files in `data/biomedical_clinical/landing/`.

### `bioinfo_gene`

Gene expression data predicting `expression_level`. It's here to show the config-driven design
works on a different domain: a leaner config (two models, no joins), with profiling on and
clustering/drift off, and no Python changes. Put the CSVs in `data/bioinfo_gene/landing/`.

### `hospital_readmission_lagged`

The same CMS source, but predicting each hospital's 2025 pneumonia readmission ratio from its
2024 measures (~3,063 hospitals matched across both years). It adds a direct-join source
(`Hospital_General_Information`) alongside the pivot-joins.

Leakage lesson: an early version reached test R² ~0.22 because `Count of READM Measures
Better/Worse` was a proxy for the hospital's own past readmission performance. Dropping it in
`config/hospital_readmission_lagged/features.yaml` gave an honest R² of ~0.05-0.07. Any field
derived from the same event as the target deserves that suspicion.

### `m6_returns_risk`

Monthly log-returns for a panel of liquid US ETFs and stocks (M6 competition), forecasting
next-month return and reporting forecast-error uncertainty. Models: random walk, historical
mean, ARIMA and a cross-sectional LightGBM. Stage the data with
`uv run python -m scripts.stage_m6_returns_risk_landing` (yfinance).

### `pjm_load_forecast`

For `problem_type: forecasting`, `dag_factory.py` swaps in the `src/forecasting/` stage functions
for clean, features, train and register. The DAG shape stays the same.

- **Data:** [`PJME_hourly.csv`](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
  in `data/pjm_load_forecast/landing/`, plus hourly weather for PHL/EWR/BWI/DCA from Open-Meteo
  (`uv run python -m scripts.stage_pjm_load_forecast_landing`).
- **Models:** daily and weekly naive baselines, ETS, recursive LightGBM with and without
  weather, and a direct LightGBM benchmark. All are scored on the same 50 midnight cutoffs
  with a 24-hour horizon. The champion is the model with the lowest validation MAPE.
- **Result:** 2.72% test MAPE with weather vs 4.33% without and 6.89% for repeat-yesterday.
  The weather runs use observed weather for the forecast hours, so this is a best case, not an
  operational forecast.
- **Serving:** `curl "http://localhost:8000/predict/forecast?horizon_hours=24"` after promoting a
  model and setting `SERVING_MODEL_NAME`. Weather models return 501 until a weather forecast
  feed exists, so serve `pjm_load_forecast_gbm_no_weather`.
- **Notebooks:**
  - `notebooks/pjm_load_forecast_pipeline.ipynb`: walkthrough using the pipeline code
  - `notebooks/pjm_load_forecast_colab.ipynb`: standalone Google Colab version with the same results

## Stack

Airflow 3 (LocalExecutor) · pandera · ydata-profiling · pandas · scikit-learn + LightGBM
(+ statsmodels for forecasting) · MLflow (Postgres backend, local artifacts) · Evidently · FastAPI ·
Pydantic + YAML · local Parquet with manifest checksums · uv.

## Testing & Debugging

```bash
uv run pytest tests/ -v
```

- [TESTING.md](TESTING.md): end-to-end run, MLflow checks, promotion, `/predict`
- [DIAGNOSTICS.md](DIAGNOSTICS.md): debugging each stage, reading metrics, common issues
- [scripts/README.md](scripts/README.md): landing-zone staging scripts

Quick fixes: `docker logs <service>` and `docker compose restart <service>`. If MLflow looks
down, check `curl http://localhost:5001/health`.
