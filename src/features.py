"""Feature engineering stage: encoding, Box-Cox, VIF pruning, scaling, train/test split."""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import JoinStrategyConfig, load_features_config, load_pipeline_config
from src.utils.io import READERS, load_manifest, resolve_run_path, write_manifest
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


def _pivot_join_sources(interim_path: Path, join_config: JoinStrategyConfig) -> pd.DataFrame:
    """Build a wide feature matrix from multiple long-format interim files.

    Identifies the spine file (filtered to a specific measure) and pivot files
    (filtered + pivoted wide on a measure column), then left-joins all onto the spine.

    Args:
        interim_path: Directory containing cleaned interim files for this run.
        join_config: Describes the spine file and any side files to pivot.

    Returns:
        Wide DataFrame with one row per id_column value.

    Raises:
        FileNotFoundError: If no file matching the spine pattern is found.
    """
    id_col = join_config.id_column
    spine_df: pd.DataFrame | None = None
    side_dfs: list[pd.DataFrame] = []

    for f in sorted(interim_path.iterdir()):
        reader = READERS.get(f.suffix.lower())
        if reader is None:
            continue

        df = reader(f)

        # Normalise id_column to nullable int so float ("10001.0") and string ("010001")
        # both resolve to the same integer key before the join.
        if id_col in df.columns:
            # Non-numeric values (e.g. alphanumeric CMS Facility IDs like "01014F") become <NA>;
            # they will never match in the merge and are silently dropped — this is intentional.
            df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")

        spine_cfg = join_config.spine
        if spine_cfg and spine_cfg.file_pattern in f.name:
            if spine_cfg.measure_column and spine_cfg.measure_value:
                df = df[df[spine_cfg.measure_column] == spine_cfg.measure_value].copy()
            dupes = df[id_col].duplicated().sum()
            if dupes:
                logger.warning("Spine '%s' has %d duplicate %s — deduplicating", f.name, dupes, id_col)
                df = df.drop_duplicates(subset=[id_col], keep="first")
            spine_df = df
            logger.info("Spine loaded from %s: %d rows × %d cols", f.name, len(df), len(df.columns))
            continue

        for pivot_cfg in join_config.pivots:
            if pivot_cfg.file_pattern in f.name:
                if pivot_cfg.measure_column in df.columns:
                    mask = df[pivot_cfg.measure_column].str.contains(
                        pivot_cfg.measure_filter, na=False, regex=False
                    )
                    df = df[mask].copy()
                    if pivot_cfg.strip_suffix:
                        df[pivot_cfg.measure_column] = df[pivot_cfg.measure_column].str.replace(
                            pivot_cfg.strip_suffix, "", regex=False
                        )
                    df[pivot_cfg.value_column] = pd.to_numeric(df[pivot_cfg.value_column], errors="coerce")
                    wide = df.pivot_table(
                        index=id_col,
                        columns=pivot_cfg.measure_column,
                        values=pivot_cfg.value_column,
                        aggfunc="first",
                    ).reset_index()
                    wide.columns.name = None
                    dupes = wide[id_col].duplicated().sum()
                    if dupes:
                        logger.warning("Pivot '%s' has %d duplicate %s after pivot", f.name, dupes, id_col)
                    logger.info("Pivot '%s' → %d rows × %d cols", f.name, len(wide), len(wide.columns))
                    side_dfs.append(wide)
                break

        for direct_cfg in join_config.direct_joins:
            if direct_cfg.file_pattern in f.name:
                dupes = df[id_col].duplicated().sum()
                if dupes:
                    logger.warning(
                        "Direct-join '%s' has %d duplicate %s — deduplicating", f.name, dupes, id_col
                    )
                    df = df.drop_duplicates(subset=[id_col], keep="first")
                logger.info("Direct-join '%s' loaded: %d rows × %d cols", f.name, len(df), len(df.columns))
                side_dfs.append(df)
                break

    if spine_df is None:
        pattern = join_config.spine.file_pattern if join_config.spine else "?"
        raise FileNotFoundError(f"No spine file matching '{pattern}' found in {interim_path}")

    result = spine_df
    for side_df in side_dfs:
        overlap = [c for c in side_df.columns if c != id_col and c in result.columns]
        if overlap:
            logger.warning(
                "Dropping %d column(s) already present before merge: %s", len(overlap), overlap
            )
            side_df = side_df.drop(columns=overlap)
        result = result.merge(side_df, on=id_col, how="left")

    logger.info("Pivot-join result: %d rows × %d cols", len(result), len(result.columns))
    return result


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
    interim_path = resolve_run_path(interim_dir, run_id)
    features_path = resolve_run_path(features_dir, run_id)
    load_manifest(interim_path)  # raises FileNotFoundError if absent

    pipeline_config = load_pipeline_config(config_dir)
    features_config = load_features_config(config_dir)
    target_col = pipeline_config.target.name

    features_path.mkdir(parents=True, exist_ok=True)

    # Load feature matrix — pivot-join for multi-source configs, naive concat otherwise
    if features_config.join_strategy.enabled:
        df = _pivot_join_sources(interim_path, features_config.join_strategy)
    else:
        dfs = []
        for f in sorted(interim_path.iterdir()):
            reader = READERS.get(f.suffix.lower())
            if reader is not None:
                dfs.append(reader(f))
        if not dfs:
            supported = sorted(READERS)
            raise FileNotFoundError(f"No supported files found in {interim_path}. Supported: {supported}")
        df = pd.concat(dfs, axis=0, ignore_index=True)

    # Encode categorical columns per config strategy
    df = _encode_columns(df, features_config.encoding)

    # Drop explicitly excluded columns
    df = df.drop(columns=features_config.drop_columns, errors="ignore")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # NZV filter (protects target column)
    df = _apply_nzv_filter(df, features_config.nzv_threshold, exclude_cols=[target_col])

    # Replace inf with NaN, then fill NaN with column median
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop columns that are still all-NaN (median was NaN, fill did nothing)
    all_nan_cols = [
        col for col in df.select_dtypes(include="number").columns if df[col].isna().all()
    ]
    if all_nan_cols:
        logger.warning("Dropping %d all-NaN columns after imputation: %s", len(all_nan_cols), all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    transform_meta: dict[str, Any] = {}

    # SVG Stage 2: Box-Cox transform on target
    if features_config.boxcox_target:
        df[target_col], lambda_val = boxcox_transform(df[target_col])
        transform_meta["boxcox_lambda"] = lambda_val
        logger.info("Applied Box-Cox to target '%s': λ=%.4f", target_col, lambda_val)

    # Final sweep: drop any predictor column still carrying NaN or inf before split.
    # Catches non-numeric columns not in encoding_map and any edge cases above missed.
    predictor_cols = [c for c in df.columns if c != target_col]
    final_bad = [
        c for c in predictor_cols
        if df[c].isna().any() or (df[c].dtype.kind in "fc" and np.isinf(df[c]).any())
    ]
    if final_bad:
        logger.warning(
            "Final cleanup: dropping %d columns with NaN/inf before split: %s",
            len(final_bad), final_bad,
        )
        df = df.drop(columns=final_bad)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # SVG Stage 2: VIF pruning on predictor matrix
    vif_dropped: list[str] = []
    if features_config.vif_threshold is not None:
        numeric_X = X.select_dtypes(include="number")
        # Guard: drop any column still carrying NaN or inf before passing to statsmodels
        bad_cols = numeric_X.columns[
            numeric_X.isin([np.inf, -np.inf]).any() | numeric_X.isna().any()
        ].tolist()
        if bad_cols:
            logger.warning("Dropping %d columns with NaN/inf before VIF: %s", len(bad_cols), bad_cols)
            numeric_X = numeric_X.drop(columns=bad_cols)
            X = X.drop(columns=bad_cols)
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

    write_manifest(features_path, {
        "run_id": run_id,
        "source": "engineered features",
        "stage": "feature_engineer",
        "transform_meta": transform_meta,
        "train": {"path": str(train_path), "rows": len(train_df), "columns": len(train_df.columns)},
        "test": {"path": str(test_path), "rows": len(test_df), "columns": len(test_df.columns)},
    })

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
