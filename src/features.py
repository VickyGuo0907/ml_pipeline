"""Feature engineering stage: encoding, Box-Cox, VIF pruning, scaling, train/test split."""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from src.utils.config import JoinStrategyConfig, load_features_config, load_pipeline_config
from src.utils.io import READERS, load_manifest, resolve_run_path, write_manifest
from src.utils.transforms import (
    apply_boxcox,
    drop_high_vif,
    fit_boxcox,
)

logger = logging.getLogger(__name__)

# Maps features.yaml's `scaler` string to a scikit-learn scaler class. Unknown strategies
# fall back to StandardScaler (the pre-existing default, so omitting `scaler` from a
# pipeline's features.yaml reproduces prior behavior exactly).
SCALER_REGISTRY: dict[str, type] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
}


def _fit_encoders(df: pd.DataFrame, encoding_map: dict[str, str]) -> dict[str, Any]:
    """Fit per-column encoding strategies on training data only.

    Fitting statistics (category counts for frequency, class tables for label)
    must come from the training set alone so no test information leaks into the
    encoded feature values.

    Supports: frequency | label. Unknown strategies fall back to label.

    Args:
        df: Training DataFrame.
        encoding_map: Maps column name → strategy string.

    Returns:
        Mapping of column name → (strategy, fitted artifact), where the artifact
        is a {category: count} dict (frequency) or a fitted LabelEncoder (label).
    """
    fitted: dict[str, Any] = {}
    for col, strategy in encoding_map.items():
        if col not in df.columns:
            continue
        if strategy == "frequency":
            fitted[col] = ("frequency", df[col].value_counts().to_dict())
        else:  # label or unknown → LabelEncoder
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            fitted[col] = ("label", le)
    return fitted


def _apply_encoders(df: pd.DataFrame, fitted: dict[str, Any]) -> pd.DataFrame:
    """Apply previously fitted encoders to a DataFrame (train or test).

    Categories unseen during fitting are mapped to 0 (frequency) or a reserved
    unknown class (label) instead of raising.

    Args:
        df: DataFrame to encode.
        fitted: Mapping from _fit_encoders.

    Returns:
        Copy of df with encoded columns (originals replaced).
    """
    df = df.copy()
    for col, (strategy, artifact) in fitted.items():
        if col not in df.columns:
            continue
        if strategy == "frequency":
            df[col] = df[col].map(artifact).fillna(0).astype(int)
        else:
            lookup = {category: i for i, category in enumerate(artifact.classes_)}
            df[col] = df[col].astype(str).map(lookup).fillna(len(artifact.classes_)).astype(int)
    return df


def _fit_medians(df: pd.DataFrame) -> dict[str, float]:
    """Compute column medians on training data for later imputation.

    Args:
        df: Training DataFrame.

    Returns:
        Mapping of column name → median for numeric columns containing NaN.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    return {c: df[c].median() for c in numeric_cols if df[c].isna().any()}


def _apply_medians(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Impute NaN values in a DataFrame using train-fitted column medians.

    Args:
        df: DataFrame to impute (train or test).
        medians: Mapping from _fit_medians.

    Returns:
        Copy of df with NaN values filled.
    """
    df = df.copy()
    for col, median in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    return df


def _dedupe_on_id(df: pd.DataFrame, id_col: str, label: str) -> pd.DataFrame:
    """Drop rows with a duplicate id_col value, warning first. Keeps the first occurrence.

    Args:
        df: DataFrame to check.
        id_col: Column that should be unique (the join key).
        label: Human-readable source name for the log message (e.g. filename).

    Returns:
        df with duplicate id_col rows removed (or unchanged if none found).
    """
    dupes = df[id_col].duplicated().sum()
    if dupes:
        logger.warning("%s has %d duplicate %s — deduplicating", label, dupes, id_col)
        df = df.drop_duplicates(subset=[id_col], keep="first")
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
            df = _dedupe_on_id(df, id_col, f"Spine '{f.name}'")
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
                df = _dedupe_on_id(df, id_col, f"Direct-join '{f.name}'")
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


def _apply_nzv_filter(
    df: pd.DataFrame, threshold: float, exclude_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Compute and drop near-zero variance columns (excluding specified cols).

    Returns the dropped column names so callers can apply the identical drop
    set to a paired frame (e.g. the test split) — dropping from train alone
    would leave train/test with different columns and break model scoring.

    Args:
        df: Feature DataFrame.
        threshold: Fraction of identical values to trigger drop (e.g. 0.95).
        exclude_cols: Columns to protect from removal (e.g. target).

    Returns:
        Tuple of (df with NZV columns removed, list of dropped column names).
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
    return df.drop(columns=to_drop), to_drop


def engineer_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Engineer features from cleaned data.

    Pipeline (SVG Stages 1–2):
      drop cols → split → encode → NZV filter → median impute →
      Box-Cox target → VIF prune → scale

    The train/test split happens first so every fitted statistic (encoding
    maps, medians, Box-Cox λ/offset, VIF, scaler) is computed on the training
    set only and merely applied to the test set — preventing data leakage.

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

    # Drop explicitly excluded columns (config-driven, no data statistics involved)
    df = df.drop(columns=features_config.drop_columns, errors="ignore")

    # Drop rows with missing target — must happen before the split
    df = df.dropna(subset=[target_col])

    # Split FIRST, before any statistic-fitting transform, so no test
    # information leaks into fitted parameters (encoders, medians, λ, scaler).
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1 - pipeline_config.train_test_split,
        random_state=pipeline_config.random_state,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Encode categorical columns — encoders fit on train only, applied to both
    fitted_encoders = _fit_encoders(X_train, features_config.encoding)
    if fitted_encoders:
        X_train = _apply_encoders(X_train, fitted_encoders)
        X_test = _apply_encoders(X_test, fitted_encoders)
        for col in features_config.encoding:
            if col in X_train.columns:
                logger.info("Encoded '%s' (%s)", col, features_config.encoding[col])

    # NZV filter — drop set computed on train, applied to both frames so train
    # and test keep identical columns
    X_train, nzv_dropped = _apply_nzv_filter(X_train, features_config.nzv_threshold, exclude_cols=[])
    if nzv_dropped:
        X_test = X_test.drop(columns=nzv_dropped)

    # Replace inf with NaN in both frames, then median-impute with train-fitted medians
    for frame in (X_train, X_test):
        numeric_cols = frame.select_dtypes(include="number").columns
        frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    medians = _fit_medians(X_train)
    X_train = _apply_medians(X_train, medians)
    X_test = _apply_medians(X_test, medians)

    # Drop columns that are all-NaN in train (median was NaN, fill did nothing)
    all_nan_cols = [
        col for col in X_train.select_dtypes(include="number").columns if X_train[col].isna().all()
    ]
    if all_nan_cols:
        logger.warning("Dropping %d all-NaN columns after imputation: %s", len(all_nan_cols), all_nan_cols)
        X_train = X_train.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    transform_meta: dict[str, Any] = {}

    # SVG Stage 2: Box-Cox transform on target — λ + offset fit on train,
    # applied to both. Offset is persisted so serve.py can invert it exactly.
    if features_config.boxcox_target:
        boxcox_params = fit_boxcox(y_train)
        y_train = apply_boxcox(y_train, boxcox_params)
        y_test = apply_boxcox(y_test, boxcox_params)
        transform_meta["boxcox_lambda"] = boxcox_params["lambda"]
        transform_meta["boxcox_offset"] = boxcox_params["offset"]
        logger.info(
            "Applied Box-Cox to target '%s': λ=%.4f offset=%.6f",
            target_col, boxcox_params["lambda"], boxcox_params["offset"],
        )

    # Final sweep: drop any predictor column still carrying NaN or inf. Catches
    # non-numeric columns not in encoding_map and any edge cases above missed.
    # Drop set is computed on train and applied to both frames.
    final_bad = [
        c for c in X_train.columns
        if X_train[c].isna().any() or (X_train[c].dtype.kind in "fc" and np.isinf(X_train[c]).any())
    ]
    if final_bad:
        logger.warning(
            "Final cleanup: dropping %d columns with NaN/inf before scaling: %s",
            len(final_bad), final_bad,
        )
        X_train = X_train.drop(columns=final_bad)
        X_test = X_test.drop(columns=final_bad)

    # SVG Stage 2: VIF pruning on training predictor matrix
    vif_dropped: list[str] = []
    if features_config.vif_threshold is not None:
        numeric_X = X_train.select_dtypes(include="number")
        # Guard: drop any column still carrying NaN or inf before passing to statsmodels
        bad_cols = numeric_X.columns[
            numeric_X.isin([np.inf, -np.inf]).any() | numeric_X.isna().any()
        ].tolist()
        if bad_cols:
            logger.warning("Dropping %d columns with NaN/inf before VIF: %s", len(bad_cols), bad_cols)
            numeric_X = numeric_X.drop(columns=bad_cols)
            X_train = X_train.drop(columns=bad_cols)
            X_test = X_test.drop(columns=bad_cols)
        if not numeric_X.empty:
            pruned, vif_dropped = drop_high_vif(numeric_X, features_config.vif_threshold)
            X_train = X_train.drop(columns=vif_dropped)
            X_test = X_test.drop(columns=vif_dropped)
            transform_meta["vif_dropped"] = vif_dropped

    # SVG Stage 2: Center & scale numeric features — scaler fit on train only
    if features_config.scale:
        numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            scaler_cls = SCALER_REGISTRY.get(features_config.scaler, StandardScaler)
            scaler = scaler_cls()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

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
