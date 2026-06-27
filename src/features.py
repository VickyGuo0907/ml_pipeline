"""Feature engineering stage: encoding, Box-Cox, VIF pruning, scaling, train/test split."""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import load_features_config, load_pipeline_config
from src.utils.transforms import (
    boxcox_transform,
    drop_high_vif,
    frequency_encode,
)

logger = logging.getLogger(__name__)


def _encode_columns(df: pd.DataFrame, encoding_map: dict[str, str]) -> pd.DataFrame:
    """Apply per-column encoding strategy from config.

    Supports: frequency | label. Unknown strategies fall back to label.

    Args:
        df: Input DataFrame.
        encoding_map: Maps column name → strategy string.

    Returns:
        DataFrame with encoded columns (originals replaced).
    """
    for col, strategy in encoding_map.items():
        if col not in df.columns:
            continue
        if strategy == "frequency":
            df = frequency_encode(df, col)
            logger.info("Frequency-encoded '%s'", col)
        else:  # label or unknown → LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info("Label-encoded '%s'", col)
    return df


def _apply_nzv_filter(df: pd.DataFrame, threshold: float, exclude_cols: list[str]) -> pd.DataFrame:
    """Drop near-zero variance columns (excluding specified cols).

    Args:
        df: Feature DataFrame.
        threshold: Fraction of identical values to trigger drop (e.g. 0.95).
        exclude_cols: Columns to protect from removal (e.g. target).

    Returns:
        DataFrame with NZV columns removed.
    """
    to_drop = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        top_freq = df[col].value_counts(normalize=True).iloc[0] if df[col].notna().any() else 0
        if top_freq >= threshold:
            to_drop.append(col)
    if to_drop:
        logger.info("NZV filter dropped %d columns: %s", len(to_drop), to_drop)
    return df.drop(columns=to_drop)


def engineer_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Engineer features from cleaned data.

    Pipeline (SVG Stages 1–2):
      encode → drop cols → NZV filter → Box-Cox target → VIF prune → scale → split

    Args:
        interim_dir: Directory containing cleaned interim data.
        features_dir: Output directory for feature matrices.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/biomedical_clinical).

    Returns:
        Dictionary with feature matrix paths, shapes, and transform metadata.

    Raises:
        FileNotFoundError: If manifest or config is missing.
        ValueError: If target column is absent after processing.
    """
    interim_path = Path(interim_dir) / run_id
    features_path = Path(features_dir) / run_id
    manifest_path = interim_path / "manifest.yaml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    pipeline_config = load_pipeline_config(config_dir)
    features_config = load_features_config(config_dir)
    target_col = pipeline_config.target.name

    features_path.mkdir(parents=True, exist_ok=True)

    # Load and combine all cleaned CSV files
    dfs = [pd.read_csv(f) for f in interim_path.glob("*.csv")]
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {interim_path}")
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Encode categorical columns per config strategy
    df = _encode_columns(df, features_config.encoding)

    # Drop explicitly excluded columns
    df = df.drop(columns=features_config.drop_columns, errors="ignore")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # NZV filter (protects target column)
    df = _apply_nzv_filter(df, features_config.nzv_threshold, exclude_cols=[target_col])

    # Fill any remaining numeric NaN with median
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    transform_meta: dict[str, Any] = {}

    # SVG Stage 2: Box-Cox transform on target
    if features_config.boxcox_target:
        df[target_col], lambda_val = boxcox_transform(df[target_col])
        transform_meta["boxcox_lambda"] = lambda_val
        logger.info("Applied Box-Cox to target '%s': λ=%.4f", target_col, lambda_val)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # SVG Stage 2: VIF pruning on predictor matrix
    vif_dropped: list[str] = []
    if features_config.vif_threshold is not None:
        numeric_X = X.select_dtypes(include="number")
        if not numeric_X.empty:
            pruned, vif_dropped = drop_high_vif(numeric_X, features_config.vif_threshold)
            X = X.drop(columns=vif_dropped)
            transform_meta["vif_dropped"] = vif_dropped

    # SVG Stage 2: Center & scale numeric features
    if features_config.scale:
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - pipeline_config.train_test_split,
        random_state=pipeline_config.random_state,
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    feature_manifest = {
        "run_id": run_id,
        "source": "engineered features",
        "stage": "feature_engineer",
        "transform_meta": transform_meta,
        "train": {"path": str(train_path), "rows": len(train_df), "columns": len(train_df.columns)},
        "test": {"path": str(test_path), "rows": len(test_df), "columns": len(test_df.columns)},
    }
    with open(features_path / "manifest.yaml", "w") as f:
        yaml.dump(feature_manifest, f, default_flow_style=False, sort_keys=False)

    logger.info(
        "Features ready: train=%s test=%s vif_dropped=%d",
        train_df.shape, test_df.shape, len(vif_dropped),
    )

    return {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "feature_count": len(train_df.columns) - 1,
        **transform_meta,
    }
