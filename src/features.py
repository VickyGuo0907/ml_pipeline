"""Feature engineering stage: encoding, scaling, train/test split."""
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import load_features_config, load_pipeline_config


def engineer_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Engineer features from cleaned data.

    Applies encoding, scaling, train/test split per config.

    Args:
        interim_dir: Directory containing cleaned data
        features_dir: Output directory for feature matrices
        run_id: Run identifier
        config_dir: Configuration directory

    Returns:
        Dictionary with feature matrix paths and shapes

    Raises:
        FileNotFoundError: If manifest or config doesn't exist
    """
    interim_path = Path(interim_dir) / run_id
    features_path = Path(features_dir) / run_id
    manifest_path = interim_path / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load configs
    pipeline_config = load_pipeline_config(config_dir)
    features_config = load_features_config(config_dir)

    # Create features directory
    features_path.mkdir(parents=True, exist_ok=True)

    # Load and combine cleaned data
    dfs = []
    for csv_file in interim_path.glob("*.csv"):
        dfs.append(pd.read_csv(csv_file))

    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {interim_path}")

    # Concatenate all files
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Label encode categorical columns
    encoders = {}
    for col_pattern, encoding_type in features_config.encoding.items():
        if encoding_type == "label":
            matching_cols = [c for c in df.columns if col_pattern.rstrip("_*") in c]
            for col in matching_cols:
                if col in df.columns and df[col].dtype == "object":
                    le = LabelEncoder()
                    df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    encoders[col] = le
                    df = df.drop(columns=[col])

    # Drop specified columns
    df = df.drop(columns=features_config.drop_columns, errors="ignore")

    # Fill missing values with column median (numeric) or mode (categorical)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown", inplace=True)

    # Drop rows where target is missing
    target_col = pipeline_config.target.name
    df = df.dropna(subset=[target_col])

    # Scale numeric features
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if features_config.scale and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Train/test split
    target_col = pipeline_config.target.name
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1 - pipeline_config.train_test_split,
        random_state=pipeline_config.random_state,
    )

    # Add target back
    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test

    # Save feature matrices
    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Write manifest
    feature_manifest = {
        "run_id": run_id,
        "source": "engineered features",
        "stage": "feature_engineer",
        "train": {
            "path": str(train_path),
            "rows": len(train_df),
            "columns": len(train_df.columns),
        },
        "test": {
            "path": str(test_path),
            "rows": len(test_df),
            "columns": len(test_df.columns),
        },
    }

    manifest_path = features_path / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(feature_manifest, f, default_flow_style=False, sort_keys=False)

    return {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "feature_count": len(train_df.columns) - 1,  # Exclude target
    }
