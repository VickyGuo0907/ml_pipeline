# Quick Reference: Pipeline Debugging

## 🚀 Start Here: Understanding Model Performance

Your models RMSE compared to **baseline** (just predicting the mean):

```
Baseline RMSE = 0.980 (what a naive model gets)
Your Model RMSE = 0.389 
Improvement = (0.980 - 0.389) / 0.980 = 60.3% ✅ EXCELLENT!
```

**Improvement benchmarks:**
- ✅ >40% improvement = Good model, keep it
- ⚠️ 10-40% improvement = Okay, but room to improve  
- ❌ <10% improvement = Model barely learning
- 🔴 Negative = Model worse than baseline, big problem

## 🔍 Run Diagnostics

```bash
# Full pipeline analysis
python3 scripts/diagnose_pipeline.py

# Model performance analysis  
python3 scripts/analyze_models.py

# View all config
cat config/orchestration.yaml config/pipeline.yaml config/models.yaml
```

## 📋 Check Each Stage

| Stage | Command | What to Look For |
|-------|---------|------------------|
| **Ingest** | `ls data/raw/` | Files exist for run_id |
| **Raw Data** | `ls reports/*.html` | Open HTML reports, check distributions |
| **Clean** | `ls data/interim/` | Less data after cleaning (expected) |
| **Features** | `ls data/features/*/train.parquet` | No nulls, correct shape |
| **Train** | Airflow UI → Logs | RMSE values, no errors |
| **Models** | http://localhost:5000 | Browse runs & metrics |
| **Drift** | `ls reports/*drift*.html` | Open report, check for issues |

## 🐛 Common Issues & Solutions

### Models underperforming?
1. Check baseline: `(baseline_rmse - model_rmse) / baseline_rmse`
2. Add features: Interactions `(f1*f2)`, polynomials `(f1²)`
3. Tune hyperparameters in `config/models.yaml`
4. Try new algorithms in training code

### Data looks wrong?
1. Check raw reports: `open reports/*.html`
2. Look for outliers, missing values, skewed distributions
3. Review `config/cleaning.yaml` - adjust thresholds
4. Review `config/features.yaml` - check transforms

### MLflow not showing models?
1. Models logged? Check train.py exception handling
2. MLflow running? `curl http://localhost:5000/health`
3. Metrics in Airflow logs? Check task logs in UI

### Configuration changed but no effect?
```bash
# Validate configs load
python3 -c "from src.utils import load_orchestration_config; print(load_orchestration_config())"

# Should not raise exceptions
```

### DAG not running?
1. Check Airflow scheduler: `docker-compose logs airflow-scheduler | grep ERROR`
2. Is DAG enabled in UI? http://localhost:8080
3. File permissions? `ls -la dags/pipeline.py`

## 📊 Performance Metrics Reference

```python
# Interpret these in MLflow or logs:

train_rmse = 0.35  # Training performance
test_rmse = 0.45   # Generalization performance

# Check overfitting
gap = (test_rmse - train_rmse) / train_rmse * 100
# <15% gap = good generalization
# >30% gap = overfitting, reduce model complexity
```

## 🛠️ Config Files Quick Reference

```yaml
# orchestration.yaml - DAG settings
dag:
  dag_id: ml_pipeline
  schedule_interval: "@weekly"
  
directories:
  landing: data/landing      # Where to PUT input CSVs
  raw: data/raw              # After ingest
  interim: data/interim      # After clean
  features: data/features    # After feature eng
  
mlflow:
  tracking_uri: http://mlflow-server:5000  # Change for production

# pipeline.yaml - Target & sources
target:
  name: Excess Readmission Ratio
  
sources:
  - name: hospital_data
    path: Hospital_Readmissions_Reduction_Program.csv
    
# models.yaml - Model hyperparameters
models:
  - name: linear_baseline
    type: linear
    hyperparameters:
      alpha: 1.0  # Decrease = less regularization
      
  - name: lightgbm_gbm
    type: gbm
    hyperparameters:
      learning_rate: 0.05  # Decrease for slower, better learning
      num_leaves: 31       # Increase for more complexity
```

## 🔗 Full Guides

| Issue | Document |
|-------|----------|
| **Step-by-step debugging** | [DIAGNOSTICS.md](DIAGNOSTICS.md) |
| **End-to-end testing** | [TESTING.md](TESTING.md) |
| **Configuration details** | See `config/` YAML files |

## ✅ Healthy Pipeline Looks Like

```
✓ All 9 Airflow tasks show "success" status
✓ 0 nulls in feature matrices
✓ No errors in MLflow or Docker logs
✓ Model RMSE improvement >40% over baseline
✓ Train/test RMSE gap <15%
✓ MLflow shows registered models
✓ Reports generated in reports/
```

## 📞 When Stuck

1. **Read DIAGNOSTICS.md** - Most answers are there
2. **Check logs**: Airflow UI or `docker-compose logs`
3. **Validate configs**: Run diagnostic scripts
4. **Reset**: `docker-compose down && docker-compose up -d`

---

**Remember:** 60% improvement over baseline = your models are working well! 🎉
