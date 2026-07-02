# Biomedical Clinical Pipeline — Business Context

## Problem Statement

CMS (Centers for Medicare & Medicaid Services) publishes annual hospital performance
data under the Hospital Readmissions Reduction Program (HRRP). Hospitals with
higher-than-expected pneumonia readmission rates face Medicare reimbursement penalties
of up to 3% — millions of dollars for a mid-size hospital.

**Core question:** Can we predict which hospitals will have excessive pneumonia
readmission rates using patient experience survey scores, so hospital administrators
can identify and intervene before penalties are applied?

---

## Datasets

### FY_2024_Hospital_Readmissions_Reduction_Program (spine)
Source: CMS Hospital Compare — HRRP annual report

Contains one row per hospital per measure. This pipeline filters to:
- `Measure Name == "READM-30-PN-HRRP"` (30-day pneumonia readmissions)

Key column:
- `Excess Readmission Ratio` — actual readmissions / expected readmissions
  - `< 1.0` → hospital performs better than expected (no penalty)
  - `= 1.0` → exactly at expected rate
  - `> 1.0` → higher than expected (penalty risk)

This is the **target variable** for the regression model.

### HCAHPS Hospital Survey (pivot)
Source: CMS Hospital Compare — patient experience survey

Contains one row per hospital per survey question. This pipeline pivots to wide
format so each HCAHPS question becomes a feature column. Key questions include:
- Communication with nurses / doctors
- Responsiveness of hospital staff
- Pain management
- Communication about medicines
- Discharge information clarity
- Overall hospital rating
- Willingness to recommend

These become the **predictor features** after pivoting on `HCAHPS Question`.

---

## Hypothesis

Hospitals that score poorly on patient experience (HCAHPS) also tend to have higher
pneumonia readmission rates. Poor communication at discharge leads to patients not
understanding follow-up care instructions, causing preventable readmissions.

---

## Join Strategy

The two datasets are joined in Stage 05 (feature engineering) on `Facility ID`:

```
PN Readmissions (spine)          HCAHPS Survey (pivot → wide)
one row per hospital        LEFT JOIN     one row per hospital
containing target column    ─────────►   containing survey scores
```

Left join ensures every hospital from the PN file is retained even if it has
no matching HCAHPS data.

---

## Model Results (FY2024 data, run 2026-07-01)

| Metric | Linear Baseline (Ridge) | LightGBM |
|--------|------------------------|----------|
| Train shape | 2,534 × 11 | 2,534 × 11 |
| Test shape | 595 × 11 | 595 × 11 |

**R² ≈ 0.04** — HCAHPS scores alone are weak predictors of readmission ratio.
The relationship exists but is noisy. Many factors not captured here (patient
demographics, socioeconomic status, hospital size, case mix) also drive readmissions.

This is scientifically expected — patient satisfaction and clinical outcomes are
correlated but not tightly linked.

---

## Unsupervised Findings (Stage 06b)

PCA on the 11-feature matrix:
- **PC1 explains 59.8%** of total variance — one dominant quality dimension
- **Only 3 components needed for 80%** — features are highly correlated

K-means clustering (optimal k=2):
- **Cluster 0** — 823 hospitals (32%) — low_performance
- **Cluster 1** — 1,711 hospitals (68%) — high_performance
- Silhouette score 0.307 — clusters are real but soft (performance is a spectrum)

---

## Practical Use Cases

1. **Early warning** — flag hospitals likely to face HRRP penalties before the
   CMS reporting period closes, giving time to improve discharge processes
2. **Benchmarking** — the two-cluster segmentation gives health networks a tool
   for identifying which hospitals need targeted coaching
3. **Resource allocation** — systems managing multiple hospitals can prioritize
   interventions for the low-performance cluster

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pipeline.yaml` | Target column, train/test split, validation rules, unsupervised config |
| `cleaning.yaml` | Sentinel replacement, column drops, imputation strategy |
| `features.yaml` | Pivot-join strategy, encoding, Box-Cox, drop columns |
| `models.yaml` | Ridge and LightGBM hyperparameters |
| `orchestration.yaml` | Airflow DAG schedule, data directories |
