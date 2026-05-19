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

```bash
# Full diagnostics report
python3 scripts/diagnose_pipeline.py

# Model performance analysis
python3 scripts/analyze_models.py

# Or in Docker environment
docker-compose exec airflow-scheduler python3 /path/to/script.py
```

## Pipeline Stages & Debugging

### Stage 1: Data Ingestion (`01_ingest_files`)

**What to check:**
```python
# Check ingested files exist
ls -lh data/raw/$(date +%Y-%m-%d)/

# Verify manifest file
cat data/raw/$(date +%Y-%m-%d)/manifest.yaml
```

**Common issues:**
- Files not found in `data/landing/` → Add CSV files
- Manifest missing → Check ingest task logs
- Wrong file format → Ensure CSV files

### Stage 2: Raw Data Validation (`02_validate_raw_schema`)

**What to check:**
```python
import pandas as pd
from src.schemas.raw import raw_schema

# Load raw data
df = pd.read_parquet("data/raw/2026-05-19/hospital_data.parquet")

# Validate schema
raw_schema.validate(df)  # Raises if invalid
```

**Common issues:**
- Schema mismatch → Check column names/types in config
- Extra/missing columns → Verify source data
- Data type errors → Check type coercion in cleaning stage

### Stage 3: Data Profiling (`03_profile_data`)

**What to check:**
```bash
# Open generated reports
open reports/2026-05-19_hospital_data.html

# Check report directory
ls -lh reports/*.html
```

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
interim_df = pd.read_parquet("data/interim/2026-05-19/hospital_data.parquet")

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

train_df = pd.read_parquet("data/features/2026-05-19/train.parquet")
test_df = pd.read_parquet("data/features/2026-05-19/test.parquet")

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
- Nulls in engineered features → Check feature.yaml transforms
- Different feature sets → Ensure both train/test get same features
- Leakage → Target info shouldn't be in features

### Stage 6: Feature Validation (`06_validate_features_schema`)

**What to check:**
```python
from src.schemas.features import features_schema
import pandas as pd

df = pd.read_parquet("data/features/2026-05-19/train.parquet")
features_schema.validate(df)  # Raises if invalid
```

**Common issues:**
- Schema validation fails → Check feature types in features.yaml
- Feature count mismatch → Verify NZV filtering didn't remove too much

### Stage 7: Model Training (`07_train_models`)

**Check training logs:**
```bash
# View task logs in Airflow UI
http://localhost:8080 → ml_pipeline → 07_train_models → Logs

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

### Stage 9: Drift Detection (`09_drift_report`)

**What to check:**
```bash
# Check generated drift reports
ls -lh reports/*drift*.html

# Open report
open reports/2026-05-19_baseline_drift_report.html
```

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
# View orchestration config
cat config/orchestration.yaml

# View pipeline config
cat config/pipeline.yaml

# View model config
cat config/models.yaml
```

### Validate configs load correctly:

```python
from src.utils import (
    load_orchestration_config,
    load_pipeline_config,
    load_models_config,
    load_features_config,
    load_cleaning_config
)

# These should not raise exceptions
orch_cfg = load_orchestration_config()
pipe_cfg = load_pipeline_config()
model_cfg = load_models_config()
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
2. Click `ml_pipeline` DAG
3. Click failing task → Logs tab
4. Look for error messages

### Check Docker logs:

```bash
# Airflow scheduler logs
docker-compose logs -f airflow-scheduler | grep -i error

# MLflow server logs
docker-compose logs -f mlflow-server

# All services
docker-compose logs -f
```

### Restart everything:

```bash
# Stop all services
docker-compose down

# Rebuild and restart
docker-compose up -d

# Check health
docker-compose ps
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
