# Pipeline Diagnostics & Debugging Guide

When models aren't performing as expected, use this guide to diagnose issues systematically at each pipeline stage.

## Quick Start

### Understanding Model Performance

**Your baseline** is predicting the mean value. Your models should beat this:

```
Baseline RMSE = σ (standard deviation of target)
Model RMSE = your model's root mean squared error
Improvement = (Baseline - Model) / Baseline × 100%
```

✅ **60% improvement = Good performance**  
⚠️ **<10% improvement = Model barely learning**  
❌ **Negative improvement = Model worse than baseline**

### Run Diagnostics

Use the snippets in each stage section below directly in a Python shell (or
`docker compose exec airflow-scheduler python3`), or check the MLflow UI's
Runs tab for metrics and model comparisons.

## Pipeline Stages & Debugging

### Stage 1: Data Ingestion (`01_ingest_files`)

**What to check:**
```python
# Check ingested files exist
ls -lh data/biomedical_clinical/raw/$(date +%Y-%m-%d)/

# Verify manifest file
cat data/biomedical_clinical/raw/$(date +%Y-%m-%d)/manifest.yaml
```

**Common issues:**
- Files not found in `data/biomedical_clinical/landing/` → Add CSV or Parquet files
- Manifest missing → Check ingest task logs
- Wrong file format → Ensure supported format (CSV or Parquet)

### Stage 2: Raw Data Validation (`02_validate_raw_schema`)

**What to check:**
```python
import pandas as pd
from src.utils.config import load_pipeline_config

cfg = load_pipeline_config("config/biomedical_clinical")
print("Sentinels:", cfg.validation.sentinel_values)
print("Per-file schemas:", cfg.validation.per_file_schemas)

df = pd.read_csv("data/biomedical_clinical/raw/2026-06-29/FY_2024_Hospital_Readmissions_Reduction_Program_Hospital.csv")
print(df.shape, df.columns.tolist())
```

**Common issues:**
- Sentinel strings not replaced → Check `validation.sentinel_values` in `pipeline.yaml`
- Required column missing → Check `per_file_schemas` in `pipeline.yaml` for that file's pattern
- Row count below minimum → Check `min_rows` in the matching `per_file_schemas` entry
- Data type errors → Sentinel replacement must happen before numeric coercion

### Stage 3: Data Profiling (`03_profile_data`)

**What to check:**
```bash
# Browse all reports in the reports server
open http://localhost:8888/biomedical_clinical/

# Or list files directly
ls -lh reports/biomedical_clinical/*.html
```

In the Airflow UI, click the `03_profile_data` task instance → **Docs** tab → "Open reports →" link.

**What to look for:**
- Missing values percentages
- Outliers and distribution anomalies
- Cardinality issues (too many unique values)
- Correlations between features

### Stage 4: Data Cleaning (`04_clean_data`)

**What to check:**
```python
import pandas as pd
from src.utils import load_cleaning_config

config = load_cleaning_config()
interim_df = pd.read_parquet("data/biomedical_clinical/interim/2026-05-19/hospital_data.parquet")

# Check shape after cleaning
print(f"Rows remaining: {len(interim_df)}")
print(f"Nulls: {interim_df.isnull().sum().sum()}")
```

**Common issues:**
- Too many rows dropped → Adjust missing value threshold
- All nulls in a column → Column should be dropped
- Unexpected data loss → Review cleaning.yaml recipe

### Stage 5: Feature Engineering (`05_engineer_features`)

**What to check:**
```python
import pandas as pd

train_df = pd.read_parquet("data/biomedical_clinical/features/2026-05-19/train.parquet")
test_df = pd.read_parquet("data/biomedical_clinical/features/2026-05-19/test.parquet")

# Check feature distributions
print(train_df.describe())

# Check for NaN after engineering
print(f"Train nulls: {train_df.isnull().sum().sum()}")
print(f"Test nulls: {test_df.isnull().sum().sum()}")

# Check train/test consistency
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
```

**Common issues:**
- Nulls in engineered features → Check `features.yaml` transforms and `protect_columns` in `cleaning.yaml`
- Pivot-join produces empty result → Verify `spine.file_pattern` matches actual interim filename
- Too few features after NZV filter → Lower `nzv_threshold` in `features.yaml`
- VIF prunes too aggressively → Set `vif_threshold: null` for datasets with correlated predictors (e.g. survey data)
- Different feature sets train vs test → Train/test split happens after all transforms, so this shouldn't occur
- Leakage → Target column must not appear in predictors; check `drop_columns` in `features.yaml`

### Stage 6: Feature Validation (`06_validate_features_schema`)

**What to check:**
```python
from src.schemas.features import features_schema
import pandas as pd

df = pd.read_parquet("data/biomedical_clinical/features/2026-05-19/train.parquet")
features_schema.validate(df)  # Raises if invalid
```

**Common issues:**
- Schema validation fails → Check feature types in features.yaml
- Feature count mismatch → Verify NZV filtering didn't remove too much

### Stage 7: Model Training (`07_train_models`)

**Check training logs:**
```bash
# View task logs in Airflow UI
http://localhost:8080 → biomedical_clinical_pipeline → 07_train_models → Logs

# Check metrics in MLflow
http://localhost:5000 → Experiments → Default → Run details
```

**What to check:**
```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
client = mlflow.tracking.MlflowClient()

# Get latest runs
runs = client.search_runs(experiment_names=["0"], max_results=5)

for run in runs[:1]:
    print(f"Metrics: {run.data.metrics}")
    print(f"Parameters: {run.data.params}")
```

**Model Performance Interpretation:**

```
Baseline RMSE = 0.980 (standard deviation)

Linear Model: 0.443 (test RMSE)
  Improvement = (0.980 - 0.443) / 0.980 = 54.8% ✅ GOOD

LightGBM: 0.389 (test RMSE)  
  Improvement = (0.980 - 0.389) / 0.980 = 60.3% ✅ EXCELLENT
```

**Common issues:**
- High train RMSE → Model not fitting
- High gap (train vs test) → Overfitting, reduce complexity
- Worse than baseline → Check features, check for data leakage

### Stage 8: Model Registration (`08_register_to_mlflow`)

**What to check:**
```bash
# Check registered models
curl http://localhost:5000/api/2.0/mlflow/registered-models/list

# Or in Python
import mlflow
client = mlflow.tracking.MlflowClient("http://mlflow-server:5000")
models = client.search_registered_models()
for m in models:
    print(f"{m.name}: {m.latest_versions}")
```

**Common issues:**
- Models not registering → Check MLflow connectivity
- 404 errors → Models not logged as artifacts in training

### Stage 6c: Benchmark Snapshot (`06c_create_benchmark`) — Champion/Challenger Regression Check

**What to check:**
```bash
# The evaluation report has full audit detail — this is the first place to look
cat reports/biomedical_clinical/{run_id}_evaluation.yaml
# Look for: run_champion (best model this run), regression_vs_production,
# production_rmse_ci / candidate_rmse_ci, drift_detected (per model)
```

```bash
# Confirm whether a benchmark set actually exists yet — the regression check
# silently no-ops until one is created
ls -lh data/biomedical_clinical/benchmark/
cat data/biomedical_clinical/benchmark/current.yaml
```

**Refreshing the benchmark set** (this task is always in the DAG but does nothing on a normal
scheduled run):
```bash
docker exec airflow-scheduler airflow dags trigger biomedical_clinical_pipeline \
  --conf '{"refresh_benchmark": true}'
```

**Common issues:**
- `regression_vs_production` missing from a model's report entry → either no benchmark exists yet
  (`current.yaml` absent — trigger a refresh above), `pipeline.yaml`'s `benchmark.enabled` is
  `false`, or there's no Production version yet for that model name to compare against. All three
  are expected no-ops, not errors.
- Regression flagged (`regression_vs_production: true`) but you're not sure if it's real → check
  the same model's `drift_detected` field. Flagged + drift detected usually means the data shifted;
  flagged + no drift usually means the model itself got worse.
- A flagged regression never blocks Staging registration — this whole feature is informational
  only. If you expected a model to stop registering, check `evaluation.min_test_r2`/`max_test_rmse`
  in `models.yaml` instead (the actual gate).

See `docs/superpowers/specs/2026-07-03-champion-challenger-regression-check-design.md` for the
full design rationale.

### Stage 9: Drift Detection (`09_drift_report`)

**What to check:**
```bash
# Browse drift reports in the reports server
open http://localhost:8888/biomedical_clinical/

# Or list directly
ls -lh reports/biomedical_clinical/*drift*.html
```

In the Airflow UI, click the `09_drift_report` task instance → **Docs** tab → "Open reports →" link.

**What to look for:**
- Dataset drift detected → Features distribution changed
- Statistical warnings → May indicate model performance degradation

## Data Quality Checklist

| Check | Command | Expected |
|-------|---------|----------|
| **Missing values** | `df.isnull().sum()` | 0 |
| **Duplicates** | `df.duplicated().sum()` | 0 or expected # |
| **Data types** | `df.dtypes` | All numeric for features |
| **Outliers** | `df.describe()` | Check min/max reasonableness |
| **Class balance** | `df[target].value_counts()` | Reasonable distribution |
| **Train/test split** | `len(train) / len(all)` | ~80/20 |

## Performance Debugging

### Models are underperforming?

1. **Check baseline first**
   ```python
   baseline = ((y_test - y_test.mean()) ** 2).mean() ** 0.5
   # Your models should beat this RMSE
   ```

2. **Analyze features**
   ```python
   # Low variance features
   train_df.var().sort_values()
   
   # Correlation with target
   train_df.corr()[target].sort_values(ascending=False)
   ```

3. **Check for overfitting**
   ```python
   train_rmse = 0.35
   test_rmse = 0.45
   overfitting = (test_rmse - train_rmse) / train_rmse * 100  # 28% gap = bad
   ```

4. **Try improvements**
   - Add feature interactions: `f1 * f2`
   - Add polynomial features: `f1²`
   - Tune hyperparameters in `config/models.yaml`
   - Try different algorithms

### Models are great - optimize them?

1. **Feature importance analysis**
   ```python
   # For LightGBM models
   feature_importance = model.feature_importances_
   ```

2. **Hyperparameter tuning**
   - Linear: Adjust `alpha` in `models.yaml`
   - LightGBM: Tune `learning_rate`, `num_leaves`, `max_depth`

3. **Cross-validation**
   - Add CV scores to catch overfitting early
   - Prevents surprises in production

## Configuration Debugging

### Check your current config:

```bash
# View base defaults
cat config/base/defaults.yaml

# View pipeline orchestration (e.g. biomedical_clinical)
cat config/biomedical_clinical/orchestration.yaml

# View pipeline config
cat config/biomedical_clinical/pipeline.yaml

# View model config
cat config/biomedical_clinical/models.yaml
```

### Validate configs load correctly:

```python
from src.utils.config import (
    discover_pipelines,
    load_pipeline_orchestration_config,
    load_pipeline_config,
    load_models_config,
)

# Discover all registered pipelines
pipelines = discover_pipelines("config")
print([p.name for p in pipelines])

# Load merged config for a specific pipeline
cfg = load_pipeline_orchestration_config("config/biomedical_clinical", "config/base")
print(cfg.dag.dag_id, cfg.tasks.retries)

# These should not raise exceptions
pipe_cfg = load_pipeline_config("config/biomedical_clinical")
model_cfg = load_models_config("config/biomedical_clinical")
```

## MLflow Integration Debugging

### Check MLflow connectivity:

```bash
# Is MLflow running?
curl http://localhost:5000/health

# Check experiments
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

### View runs and metrics:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
client = mlflow.tracking.MlflowClient()

# List experiments
exps = client.list_experiments()
for exp in exps:
    print(f"Exp {exp.experiment_id}: {exp.name}")

# Search runs
runs = client.search_runs(experiment_names=["0"])
for run in runs[:3]:
    print(f"Run {run.info.run_id}: {run.data.metrics}")
```

## When All Else Fails

### Check Airflow task logs:

1. Go to http://localhost:8080
2. Click `biomedical_clinical_pipeline` or `bioinfo_gene_pipeline` DAG
3. Click failing task → Logs tab
4. Look for error messages

### Check Docker logs:

```bash
# Airflow scheduler logs
docker compose logs -f airflow-scheduler | grep -i error

# MLflow server logs
docker compose logs -f mlflow-server

# All services
docker compose logs -f
```

### Restart everything:

```bash
# Stop all services
docker compose down

# Rebuild and restart
docker compose up -d

# Check health
docker compose ps
```

## Reference Metrics

### Good Model Performance:

- **RMSE improvement**: > 40% better than baseline
- **Train/test gap**: < 15% relative difference
- **R² score**: > 0.5
- **No data loss**: All rows preserved through pipeline
- **Feature count**: > 10 relevant features

### Red Flags:

- **RMSE**: Worse than or barely better than baseline
- **Train/test gap**: > 30% (significant overfitting)
- **Data loss**: > 20% rows dropped in cleaning
- **Missing values**: After engineering (indicates bugs)
- **All features same variance**: Likely one-hot encoding issue
