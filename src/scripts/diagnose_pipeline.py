#!/usr/bin/env python3
"""Diagnostic script to analyze pipeline performance at each stage."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_pipeline_config


def analyze_features(run_id: str = None) -> None:
    """Analyze feature matrices for the latest or specified run."""
    features_dir = Path("data/features")

    if run_id is None:
        runs = sorted([d for d in features_dir.iterdir() if d.is_dir()], reverse=True)
        if not runs:
            print("❌ No feature matrices found")
            return
        run_id = runs[0].name

    features_path = features_dir / run_id
    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"

    if not train_path.exists():
        print(f"❌ Feature files not found for run: {run_id}")
        return

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    config = load_pipeline_config()
    target = config.target.name

    print(f"\n{'='*60}")
    print(f"FEATURE ANALYSIS: {run_id}")
    print(f"{'='*60}")

    print("\n📊 Data Shapes:")
    print(f"  Train: {train_df.shape} | Test: {test_df.shape}")

    print(f"\n🎯 Target Variable: {target}")
    print(f"  Train - Mean: {train_df[target].mean():8.4f}, Std: {train_df[target].std():6.4f}")
    print(f"  Test  - Mean: {test_df[target].mean():8.4f}, Std: {test_df[target].std():6.4f}")
    print(f"  Range: [{train_df[target].min():.4f}, {train_df[target].max():.4f}]")

    print("\n🔢 Feature Statistics:")
    X_train = train_df.drop(columns=[target])
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Numeric: {X_train.select_dtypes('number').shape[1]}")
    print(f"  Categorical: {X_train.select_dtypes('object').shape[1]}")

    print("\n❌ Missing Values:")
    print(f"  Train: {train_df.isnull().sum().sum()}")
    print(f"  Test: {test_df.isnull().sum().sum()}")

    print("\n📈 Feature Variance:")
    for col in X_train.select_dtypes('number').columns[:5]:
        print(f"  {col:30s} - Mean: {X_train[col].mean():8.4f}, Std: {X_train[col].std():6.4f}")

    print("\n📊 Baseline Metrics (predicting mean):")
    baseline_rmse = ((test_df[target] - test_df[target].mean()) ** 2).mean() ** 0.5
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print("  (Model should beat this!)")


def analyze_training_results(run_id: str = None) -> None:
    """Analyze training results from latest run."""
    import mlflow

    mlflow.set_tracking_uri("http://mlflow-server:5000")

    print(f"\n{'='*60}")
    print("TRAINING RESULTS ANALYSIS")
    print(f"{'='*60}")

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_names=["0"], max_results=5)

    if not runs:
        print("❌ No training runs found in MLflow")
        return

    for run in runs[:3]:
        print(f"\n🏃 Run ID: {run.info.run_id}")
        print(f"   Status: {run.info.status}")

        metrics = run.data.metrics
        if metrics:
            print("   Metrics:")
            for key, value in sorted(metrics.items()):
                print(f"     {key:20s}: {value:.4f}")

        params = run.data.params
        if params:
            print("   Parameters:")
            for key, value in sorted(params.items())[:5]:
                print(f"     {key:20s}: {value}")


def analyze_data_quality(run_id: str = None) -> None:
    """Check data quality at different pipeline stages."""
    print(f"\n{'='*60}")
    print("DATA QUALITY CHECKS")
    print(f"{'='*60}")

    # Check raw data
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        raw_runs = sorted([d for d in raw_dir.iterdir() if d.is_dir()], reverse=True)
        if raw_runs:
            run = raw_runs[0].name
            raw_files = list((raw_dir / run).glob("*.parquet"))
            print(f"\n📁 Raw Data ({run}):")
            print(f"   Files: {len(raw_files)}")

            total_rows = 0
            for f in raw_files[:3]:
                df = pd.read_parquet(f)
                total_rows += len(df)
                print(f"   {f.name:30s}: {df.shape}")

    # Check interim data
    interim_dir = Path("data/interim")
    if interim_dir.exists():
        interim_runs = sorted([d for d in interim_dir.iterdir() if d.is_dir()], reverse=True)
        if interim_runs:
            run = interim_runs[0].name
            interim_files = list((interim_dir / run).glob("*.parquet"))
            print(f"\n🔄 Interim Data ({run}):")
            print(f"   Files: {len(interim_files)}")

            for f in interim_files[:3]:
                df = pd.read_parquet(f)
                print(f"   {f.name:30s}: {df.shape}")


def main() -> None:
    """Run all diagnostic analyses."""
    print("\n" + "="*60)
    print("🔍 ML PIPELINE DIAGNOSTIC REPORT")
    print("="*60)

    # Analyze features
    analyze_features()

    # Analyze training results
    try:
        analyze_training_results()
    except Exception as e:
        print(f"\n⚠️  Could not fetch MLflow results: {e}")

    # Analyze data quality
    analyze_data_quality()

    print(f"\n{'='*60}")
    print("✅ Diagnostic complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
