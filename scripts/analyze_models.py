#!/usr/bin/env python3
"""Analyze trained model performance and behavior."""
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_pipeline_config


def analyze_model_metrics(run_id: str = None) -> None:
    """Fetch and analyze model metrics from MLflow."""
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = mlflow.tracking.MlflowClient()

    print(f"\n{'='*70}")
    print(f"MODEL PERFORMANCE ANALYSIS")
    print(f"{'='*70}")

    # Get all runs sorted by timestamp
    runs = client.search_runs(experiment_names=["0"], max_results=20)

    models_data = {}
    for run in runs:
        if "train_rmse" in run.data.metrics:
            model_name = run.data.tags.get("model_name", "unknown")
            models_data[model_name] = {
                "run_id": run.info.run_id,
                "train_rmse": run.data.metrics.get("train_rmse", 0),
                "test_rmse": run.data.metrics.get("test_rmse", 0),
                "train_mse": run.data.metrics.get("train_mse", 0),
                "test_mse": run.data.metrics.get("test_mse", 0),
                "timestamp": run.info.start_time,
            }

    if not models_data:
        print("❌ No model training runs found")
        return

    print(f"\n📈 Latest Model Metrics:")
    print(f"{'Model':<25} {'Train RMSE':<12} {'Test RMSE':<12} {'Improvement':<12}")
    print("-" * 65)

    config = load_pipeline_config()
    target = config.target.name
    features_dir = Path("data/features")
    runs = sorted([d for d in features_dir.iterdir() if d.is_dir()], reverse=True)

    if runs:
        test_df = pd.read_parquet(runs[0] / "test.parquet")
        y_test = test_df[target]
        baseline_rmse = ((y_test - y_test.mean()) ** 2).mean() ** 0.5

        for model_name, metrics in models_data.items():
            improvement = (baseline_rmse - metrics["test_rmse"]) / baseline_rmse * 100
            print(f"{model_name:<25} {metrics['train_rmse']:<12.4f} {metrics['test_rmse']:<12.4f} {improvement:<12.1f}%")

        print(f"\nℹ️  Baseline (predict mean): {baseline_rmse:.4f}")
        print(f"    Models should beat this RMSE!")

    print(f"\n🎯 Key Insights:")
    for model_name, metrics in models_data.items():
        train_rmse = metrics["train_rmse"]
        test_rmse = metrics["test_rmse"]
        overfitting = (test_rmse - train_rmse) / train_rmse * 100

        print(f"\n  {model_name}:")
        if overfitting > 10:
            print(f"    ⚠️  Potential overfitting ({overfitting:.1f}% train→test increase)")
        elif overfitting < 0:
            print(f"    ✓ Test better than train (possible - different distributions)")
        else:
            print(f"    ✓ Reasonable generalization ({overfitting:.1f}% gap)")


def analyze_feature_importance(run_id: str = None) -> None:
    """Analyze feature importance if available."""
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE HINTS")
    print(f"{'='*70}")

    features_dir = Path("data/features")
    runs = sorted([d for d in features_dir.iterdir() if d.is_dir()], reverse=True)

    if not runs:
        print("❌ No features found")
        return

    config = load_pipeline_config()
    target = config.target.name

    train_df = pd.read_parquet(runs[0] / "train.parquet")
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    print(f"\n📊 Top 10 Features by Variance (may indicate importance):")
    variances = X_train.var().sort_values(ascending=False)
    for i, (feat, var) in enumerate(variances.head(10).items(), 1):
        print(f"  {i:2d}. {feat:35s} (variance: {var:.6f})")

    print(f"\n📈 Feature Correlations with Target:")
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
        direction = "↑" if X_train[feat].corr(y_train) > 0 else "↓"
        print(f"  {i:2d}. {feat:35s} ({direction} corr: {corr:.4f})")


def suggest_improvements() -> None:
    """Suggest areas for improvement."""
    print(f"\n{'='*70}")
    print(f"SUGGESTED IMPROVEMENTS")
    print(f"{'='*70}")

    improvements = [
        ("Feature Engineering", [
            "• Add interaction features (e.g., feature1 * feature2)",
            "• Add polynomial features (e.g., feature²)",
            "• Consider domain-specific features based on hospital data",
        ]),
        ("Hyperparameter Tuning", [
            "• Linear: Increase/decrease alpha regularization",
            "• LightGBM: Tune learning_rate, num_leaves, depth",
            "• Grid search or Bayesian optimization",
        ]),
        ("Data Quality", [
            "• Check for outliers and data entry errors",
            "• Verify train/test split is representative",
            "• Ensure no data leakage from target engineering",
        ]),
        ("Model Selection", [
            "• Try ensemble methods (Random Forest, XGBoost)",
            "• Experiment with SVM or Neural Networks",
            "• Use Stacking/Voting for ensemble models",
        ]),
        ("Monitoring", [
            "• Add cross-validation to catch overfitting early",
            "• Track model drift over time",
            "• Monitor prediction distributions",
        ]),
    ]

    for category, items in improvements:
        print(f"\n🔧 {category}:")
        for item in items:
            print(f"   {item}")


def main() -> None:
    """Run all model analyses."""
    print("\n" + "="*70)
    print("🤖 MODEL ANALYSIS REPORT")
    print("="*70)

    try:
        analyze_model_metrics()
    except Exception as e:
        print(f"\n❌ Error analyzing metrics: {e}")

    try:
        analyze_feature_importance()
    except Exception as e:
        print(f"\n⚠️  Could not analyze feature importance: {e}")

    suggest_improvements()

    print(f"\n{'='*70}")
    print("✅ Analysis complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
