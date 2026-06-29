# Testing Guide: End-to-End ML Pipeline Validation

This guide walks through validating the complete ML pipeline from data ingestion through model serving via FastAPI.

## Prerequisites

All services must be healthy:

```bash
docker-compose ps
# Should show all 7 services as "Up"
```

Access points:
- **Airflow UI**: http://localhost:8080 (user: `airflow`, password: set via `.env` or default from Docker)
- **MLflow UI**: http://localhost:5000
- **FastAPI Docs**: http://localhost:8000/docs
- **FastAPI Health**: http://localhost:8000/health

---

## Phase 1: Prepare Sample Data

Sample data is located in `data/biomedical_clinical/landing/`:

```bash
ls -lh data/biomedical_clinical/landing/
# Should contain hospital_compare_sample.csv (or other CSVs)
```

If sample data is missing, create it:

```bash
# Create a minimal test CSV with required columns
cat > data/biomedical_clinical/landing/hospital_compare_sample.csv << 'EOF'
FacilityId,Facility Name,State,City,Address,ZIP Code,ExcessReadmissionRatio,Mortality Rate,Safety Grade,HCAHPS_Cleanliness,HCAHPS_Communication,HCAHPS_Responsiveness,HCAHPS_Pain_Management,HCAHPS_Medication,HCAHPS_Discharge,HCAHPS_Quiet,ComparedToNational_Mortality,ComparedToNational_Safety,ComparedToNational_Readmission,ComparedToNational_MRSA Bacteremia,ComparedToNational_Clostridium difficile,Number of Beds,Hospital Beds,Census Region
1,Hospital A,NY,New York,123 Main St,10001,0.95,0.05,A,75.0,80.0,70.0,78.0,81.0,79.0,68.0,Below,Below,Same,Below,Below,150,150.0,Northeast
2,Hospital B,CA,Los Angeles,456 Oak Ave,90001,1.05,0.06,B,78.0,82.0,75.0,80.0,83.0,81.0,70.0,Same,Same,Above,Below,Same,200,200.0,West
3,Hospital C,TX,Houston,789 Pine Rd,77001,0.88,0.04,A,80.0,85.0,78.0,82.0,84.0,83.0,72.0,Above,Above,Below,Below,Below,175,175.0,South
EOF
```

---

## Phase 2: Trigger the DAG

### Option A: Via Airflow UI (Recommended)

1. Open http://localhost:8080
2. Navigate to **DAGs** → Search for `biomedical_clinical_pipeline`
3. Click the DAG name
4. Click the **Trigger DAG** button (play icon)
5. Leave config empty, click **Trigger**
6. Watch the **Graph** view as tasks execute

### Option B: Via Airflow CLI

```bash
# Trigger DAG from inside Airflow container
docker exec airflow-scheduler airflow dags trigger biomedical_clinical_pipeline

# List DAG runs
docker exec airflow-scheduler airflow dags list-runs --dag-id biomedical_clinical_pipeline

# Monitor a specific run
docker exec airflow-scheduler airflow tasks list --dag-id biomedical_clinical_pipeline --state success
```

### What to Expect

The DAG executes 9 stages in sequence:
1. **01_ingest_files** — Reads files from `data/<pipeline>/landing/`, outputs to `data/<pipeline>/raw/{run_id}/`
2. **02_validate_raw_schema** — Validates raw files against Pandera schema
3. **03_profile_data** — Generates ydata-profiling HTML reports in `reports/`
4. **04_clean_data** — Type coercion, missing handling → `data/<pipeline>/interim/{run_id}/`
5. **05_engineer_features** — Encoding, scaling, train/test split → `data/<pipeline>/features/{run_id}/`
6. **06_validate_features_schema** — Validates feature matrix against schema
7. **07_train_models** — Trains Ridge + LightGBM, logs to MLflow
8. **08_register_to_mlflow** — Registers models to MLflow **Staging** stage
9. **09_drift_report** — Generates Evidently drift report (compares to previous run if available)

---

## Phase 3: Verify Pipeline Outputs

After DAG completion, verify outputs at each stage:

### Check Raw Data

```bash
ls -lh data/biomedical_clinical/raw/
# Should see: {run_id}/ directory with manifest.yaml and ingested CSVs

# Check manifest for integrity
cat data/biomedical_clinical/raw/{run_id}/manifest.yaml
# Look for: file count, checksums, timestamps
```

### Check Interim (Cleaned) Data

```bash
ls -lh data/biomedical_clinical/interim/{run_id}/
# Should contain: cleaned CSVs, manifest.yaml

# Quick stats
python -c "
import pandas as pd
df = pd.read_csv('data/biomedical_clinical/interim/{run_id}/hospital_compare_sample.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Missing %: {df.isnull().mean().mean()*100:.1f}%')
"
```

### Check Feature Matrix

```bash
ls -lh data/biomedical_clinical/features/{run_id}/
# Should contain: train.parquet, test.parquet, manifest.yaml

# Inspect feature shapes and dtypes
python -c "
import pandas as pd
train = pd.read_parquet('data/biomedical_clinical/features/{run_id}/train.parquet')
test = pd.read_parquet('data/biomedical_clinical/features/{run_id}/test.parquet')
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'Columns: {list(train.columns)}')
print(f'Target range: [{train.ExcessReadmissionRatio.min():.2f}, {train.ExcessReadmissionRatio.max():.2f}]')
"
```

### Check Reports

```bash
ls -lh reports/
# Should contain: profile_*.html for each data source
# Open in browser: file:///path/to/reports/profile_*.html

# If drift report exists (second run onward):
# Look for: drift_report_*.html
```

---

## Phase 4: Check Model Training in MLflow

### Via MLflow UI

1. Open http://localhost:5000
2. Navigate to **Experiments** → **biomedical_clinical_pipeline** (default experiment)
3. Should see **2 runs** for the latest DAG execution:
   - Run 1: Linear Baseline (Ridge)
   - Run 2: LightGBM
4. Click each run to inspect:
   - **Parameters**: hyperparameters from `config/models.yaml`
   - **Metrics**: MSE, RMSE on test set
   - **Artifacts**: model files (sklearn joblib or LightGBM booster)

### Via CLI

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# List experiments
mlflow experiments list

# List runs for ml_pipeline experiment
mlflow runs list --experiment-name biomedical_clinical_pipeline

# Get details on a specific run
mlflow runs describe --run-id <run_id>

# Download artifacts
mlflow artifacts download --run-id <run_id> --dst-path ./artifacts
```

---

## Phase 5: Register and Promote Models to Production

### Via MLflow UI (Recommended)

1. **Open MLflow UI**: http://localhost:5000
2. **Navigate to Models** (in left sidebar, may be under "Registered Models")
3. **Click a run** → **Register model** button
4. **Name**: e.g., `lightgbm_hospital` or `ridge_baseline`
5. **Stage**: Leave as "None" (will default to "Staging")

### Register via Python/CLI

```bash
# Inside container
docker exec airflow-webserver python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')

# Register a model from a run
run_id = '<run_id_from_above>'
model_uri = f'runs:/{run_id}/model'
registered_model = mlflow.register_model(model_uri, 'lightgbm_hospital')
print(f'Registered: {registered_model.name}')
"
```

### Promote to Production

Once a model is registered:

1. **Open MLflow UI** → **Models**
2. **Select registered model** (e.g., `lightgbm_hospital`)
3. **Click latest version**
4. **Stage** dropdown → Select **Production**
5. Confirm

⚠️ **Important**: Only ONE model version can be in Production at a time. The FastAPI server loads the Production version on startup.

---

## Phase 6: Test FastAPI Serving

### Health Check

```bash
# Should return healthy status
curl -s http://localhost:8000/health | jq .
# Response:
# {
#   "status": "healthy",
#   "model_loaded": true
# }
```

If `model_loaded: false`, no Production model is registered. Go back to Phase 5.

### Interactive API Docs

Open http://localhost:8000/docs in a browser to see Swagger UI with:
- Full schema for `/predict` endpoint
- Example payload (auto-populated from Pydantic model)
- Try it out button

### Test Prediction via cURL

```bash
# Create test input JSON
cat > /tmp/test_input.json << 'EOF'
{
  "state_encoded": 1,
  "facility_id_encoded": 42,
  "compared_to_national_mortality_below": 1,
  "compared_to_national_mortality_same": 0,
  "compared_to_national_mortality_above": 0,
  "compared_to_national_safety_below": 0,
  "compared_to_national_safety_same": 1,
  "compared_to_national_safety_above": 0,
  "compared_to_national_readmission_below": 0,
  "compared_to_national_readmission_same": 0,
  "compared_to_national_readmission_above": 1,
  "mortality_rate": 12.5,
  "hcahps_cleanliness": 75.0,
  "hcahps_communication": 80.0,
  "hcahps_responsiveness": 70.0,
  "hcahps_pain_management": 72.0,
  "hcahps_medication": 68.0,
  "hcahps_discharge": 74.0,
  "hcahps_quiet": 71.0,
  "number_of_beds": 150.0
}
EOF

# Send prediction request
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @/tmp/test_input.json | jq .
  
# Expected response:
# {
#   "prediction": 0.95,
#   "model_version": "Production"
# }
```

### Test Prediction via Python

```bash
# Inside container or local environment with requests
python -c "
import requests
import json

input_data = {
    'state_encoded': 1,
    'facility_id_encoded': 42,
    'compared_to_national_mortality_below': 1,
    'compared_to_national_mortality_same': 0,
    'compared_to_national_mortality_above': 0,
    'compared_to_national_safety_below': 0,
    'compared_to_national_safety_same': 1,
    'compared_to_national_safety_above': 0,
    'compared_to_national_readmission_below': 0,
    'compared_to_national_readmission_same': 0,
    'compared_to_national_readmission_above': 1,
    'mortality_rate': 12.5,
    'hcahps_cleanliness': 75.0,
    'hcahps_communication': 80.0,
    'hcahps_responsiveness': 70.0,
    'hcahps_pain_management': 72.0,
    'hcahps_medication': 68.0,
    'hcahps_discharge': 74.0,
    'hcahps_quiet': 71.0,
    'number_of_beds': 150.0,
}

response = requests.post('http://localhost:8000/predict', json=input_data)
print(json.dumps(response.json(), indent=2))
"
```

---

## Phase 7: Validate Predictions

After getting predictions, verify they make sense:

1. **Output Range**: `ExcessReadmissionRatio` typically ranges 0.7–1.3 (hospital-specific variation)
2. **Feature Importance**: Higher HCAHPS scores → lower readmission predictions (inverse correlation expected)
3. **Consistency**: Same input → same prediction (deterministic)

### Batch Prediction Test

```bash
# Create feature matrix from test split
python << 'EOF'
import pandas as pd
import requests

# Load test features
test_df = pd.read_parquet('data/biomedical_clinical/features/{run_id}/test.parquet')

# Take first 5 samples (drop target)
samples = test_df.drop('ExcessReadmissionRatio', axis=1).head(5)

# Convert column names to snake_case for API
def to_snake_case(name):
    return name.replace('_', '').lower()

# Make predictions
predictions = []
for idx, row in samples.iterrows():
    # Map columns to API input names
    payload = {
        'state_encoded': int(row['State_encoded']),
        'facility_id_encoded': int(row['FacilityId_encoded']),
        'compared_to_national_mortality_below': int(row['ComparedToNational_Mortality_Below']),
        'compared_to_national_mortality_same': int(row['ComparedToNational_Mortality_Same']),
        'compared_to_national_mortality_above': int(row['ComparedToNational_Mortality_Above']),
        'compared_to_national_safety_below': int(row['ComparedToNational_Safety_Below']),
        'compared_to_national_safety_same': int(row['ComparedToNational_Safety_Same']),
        'compared_to_national_safety_above': int(row['ComparedToNational_Safety_Above']),
        'compared_to_national_readmission_below': int(row['ComparedToNational_Readmission_Below']),
        'compared_to_national_readmission_same': int(row['ComparedToNational_Readmission_Same']),
        'compared_to_national_readmission_above': int(row['ComparedToNational_Readmission_Above']),
        'mortality_rate': float(row['Mortality_Rate']),
        'hcahps_cleanliness': float(row['HCAHPS_Cleanliness']),
        'hcahps_communication': float(row['HCAHPS_Communication']),
        'hcahps_responsiveness': float(row['HCAHPS_Responsiveness']),
        'hcahps_pain_management': float(row['HCAHPS_Pain_Management']),
        'hcahps_medication': float(row['HCAHPS_Medication']),
        'hcahps_discharge': float(row['HCAHPS_Discharge']),
        'hcahps_quiet': float(row['HCAHPS_Quiet']),
        'number_of_beds': float(row['Number_of_Beds']),
    }
    
    # Add optional polynomial features if present
    for col in ['HCAHPS_Cleanliness_poly2', 'HCAHPS_Communication_poly2', 'HCAHPS_Cleanliness_Communication']:
        if col in row and pd.notna(row[col]):
            payload[col.lower().replace('_', '_')] = float(row[col])
    
    response = requests.post('http://localhost:8000/predict', json=payload)
    pred = response.json()['prediction']
    actual = test_df.loc[idx, 'ExcessReadmissionRatio']
    predictions.append({
        'actual': actual,
        'predicted': pred,
        'error': abs(actual - pred),
    })

# Display results
df_results = pd.DataFrame(predictions)
print(df_results.to_string())
print(f'\nMean Absolute Error: {df_results["error"].mean():.4f}')
EOF
```

---

## Troubleshooting

### Issue: DAG Tasks Fail

**Check logs**:
```bash
# View task logs in Airflow UI
# Or via CLI:
docker exec airflow-scheduler airflow tasks logs biomedical_clinical_pipeline 01_ingest_files {execution_date}
```

**Common causes**:
- Missing input files in `data/biomedical_clinical/landing/`
- Schema validation failure (column names/types mismatch)
- MLflow connection error (check MLFLOW_TRACKING_URI env var)

### Issue: Model Not Loaded in FastAPI

**Verify Production model exists**:
```bash
docker exec mlflow-server curl -s http://localhost:5000/api/2.0/model-registry/models | jq .
# Look for registered models with stage="Production"
```

**Check FastAPI logs**:
```bash
docker logs fastapi
# Look for model loading errors on startup
```

**Restart FastAPI service**:
```bash
docker-compose restart fastapi
# Check http://localhost:8000/health again
```

### Issue: Prediction Returns 503 Error

```bash
curl -s http://localhost:8000/health | jq .model_loaded
# If false: no Production model in MLflow registry
# Promote a model via MLflow UI (Phase 5)
```

### Issue: Data Not Visible After DAG Run

**Check volume mounts**:
```bash
# Verify data directories are bind-mounted
docker exec airflow-scheduler ls -lh /home/airflow/data/biomedical_clinical/raw/
docker exec airflow-scheduler ls -lh /home/airflow/data/biomedical_clinical/interim/
docker exec airflow-scheduler ls -lh /home/airflow/data/biomedical_clinical/features/
```

**Run date format**:
DAG uses ISO date (YYYY-MM-DD) as run_id. Verify directory names match.

---

## Full Test Checklist

- [ ] Sample data exists in `data/biomedical_clinical/landing/`
- [ ] DAG triggered successfully
- [ ] All 9 tasks complete in Airflow UI
- [ ] `data/biomedical_clinical/raw/{run_id}/manifest.yaml` created
- [ ] `data/biomedical_clinical/interim/{run_id}/` contains cleaned files
- [ ] `data/biomedical_clinical/features/{run_id}/{train,test}.parquet` created
- [ ] `reports/profile_*.html` generated
- [ ] 2 model runs visible in MLflow
- [ ] At least 1 model registered to MLflow registry
- [ ] 1 model promoted to Production stage
- [ ] `/health` returns `model_loaded: true`
- [ ] `/predict` accepts valid input and returns prediction
- [ ] Predictions are in expected range (0.7–1.3)
- [ ] Batch predictions vs. actual values computed

---

## Next Steps

Once all tests pass:

1. **Production Deployment**: Scale from Docker Compose to Kubernetes (future phase)
2. **Monitoring Dashboard**: Set up Grafana dashboards for pipeline metrics
3. **Retraining Schedule**: Automate monthly/quarterly retraining DAG runs
4. **API Rate Limiting**: Add request throttling to FastAPI
5. **Model A/B Testing**: Implement canary deployment of new model versions
