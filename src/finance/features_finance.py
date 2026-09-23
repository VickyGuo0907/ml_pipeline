"""Feature engineering stage for finance returns/risk pipelines: the
cross-sectional GBM's small lag/volatility feature set, ticker/date numeric
encoding, and the chronological per-asset train/test split.
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_finance_features_config, load_pipeline_config
from src.utils.io import load_manifest, resolve_run_path, write_manifest, write_parquet

logger = logging.getLogger(__name__)


def engineer_finance_features(
    interim_dir: str | Path,
    features_dir: str | Path,
    run_id: str,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Build the cross-sectional GBM's feature set and split each asset chronologically.

    Lag and trailing-volatility features are computed per asset (never across
    a Ticker boundary) on that asset's full cleaned series before splitting —
    legitimate historical data a forecaster would have at prediction time.
    Random Walk/Mean/ARIMA (a later plan) read the same train/test parquet
    but only ever use the target column; the lag/volatility columns exist for
    the cross-sectional GBM only. Ticker is label-encoded (sorted
    alphabetically, the mapping recorded in the manifest) and Date is
    replaced with its monthly period ordinal, since dag_factory.py's
    validate_features_wrapper rejects any non-numeric column before the
    pandera schema ever runs — a raw string Ticker or a datetime Date column
    would fail that guard.

    The split is chronological per asset (each asset's own first
    train_test_split fraction of months -> train, the rest -> test), not a
    single global date cutoff, since assets can have different date ranges.

    date_ordinal is a time AXIS for sorting/joining/reporting, not a real
    predictor: it is monotonic, so a tree-based model (LightGBM) would split
    entirely on it since every TEST-period value is numerically above every
    TRAINING value. It is recorded in the manifest's index_columns, not
    feature_columns, so a later training stage that naively predictor-selects
    via "every column except the target" does not hand it to the model. To
    convert a date_ordinal value back to a human-readable month, use
    `pd.Period(ordinal=n, freq="M")` -- NOT `pd.PeriodIndex(ordinals,
    freq="M")`, which raises on this repo's pandas version when given raw
    integer ordinals directly.

    Args:
        interim_dir: Directory containing cleaned interim data.
        features_dir: Output directory for feature matrices.
        run_id: Run identifier.
        config_dir: Pipeline config directory (e.g. config/m6_returns_risk).

    Returns:
        Dictionary with feature matrix paths, shapes, and transform metadata.

    Raises:
        FileNotFoundError: If the interim manifest is absent.
    """
    interim_path = resolve_run_path(interim_dir, run_id)
    features_path = resolve_run_path(features_dir, run_id)
    clean_manifest = load_manifest(interim_path)  # raises FileNotFoundError if absent

    features_config = load_finance_features_config(config_dir)
    pipeline_config = load_pipeline_config(config_dir)
    target_col = pipeline_config.target.name

    df = pd.read_parquet(clean_manifest["output_path"])
    df[target_col] = df[target_col].astype(float)
    df["Date"] = pd.to_datetime(df["Date"])

    tickers = sorted(df["Ticker"].unique())
    ticker_mapping = {ticker: code for code, ticker in enumerate(tickers)}

    window = features_config.volatility_window_months
    feature_parts = []
    for ticker, asset_df in df.groupby("Ticker"):
        asset_df = asset_df.sort_values("Date").copy()

        for lag in features_config.lag_months:
            asset_df[f"lag_{lag}_return"] = asset_df[target_col].shift(lag)

        shifted = asset_df[target_col].shift(1)
        asset_df[f"trailing_{window}m_vol"] = shifted.rolling(window).std()

        asset_df["ticker_encoded"] = ticker_mapping[ticker]
        asset_df["date_ordinal"] = pd.PeriodIndex(asset_df["Date"], freq="M").asi8

        feature_parts.append(asset_df)

    feature_df = pd.concat(feature_parts, ignore_index=True)
    feature_df = feature_df.drop(columns=["Date", "Ticker"])

    rows_before_dropna = len(feature_df)
    feature_df = feature_df.dropna()
    rows_dropped_warmup = rows_before_dropna - len(feature_df)

    if feature_df.empty:
        raise ValueError(
            f"No rows survive warm-up (lag_months={features_config.lag_months}, "
            f"volatility_window_months={features_config.volatility_window_months}) "
            "for any asset — check that assets have enough history."
        )

    surviving_codes = set(feature_df["ticker_encoded"].unique())
    dropped_tickers = [t for t, code in ticker_mapping.items() if code not in surviving_codes]
    if dropped_tickers:
        logger.warning(
            "Assets with no surviving rows after warm-up (dropped entirely): %s",
            sorted(dropped_tickers),
        )

    train_parts = []
    test_parts = []
    for _, asset_rows in feature_df.groupby("ticker_encoded"):
        asset_rows = asset_rows.sort_values("date_ordinal")
        split_idx = int(len(asset_rows) * pipeline_config.train_test_split)
        train_parts.append(asset_rows.iloc[:split_idx])
        test_parts.append(asset_rows.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = train_df.sort_values(["ticker_encoded", "date_ordinal"]).reset_index(drop=True)
    test_df = test_df.sort_values(["ticker_encoded", "date_ordinal"]).reset_index(drop=True)

    features_path.mkdir(parents=True, exist_ok=True)
    train_path = features_path / "train.parquet"
    test_path = features_path / "test.parquet"

    write_parquet(train_df, train_path)
    write_parquet(test_df, test_path)

    index_columns = ["date_ordinal"]
    feature_columns = [c for c in train_df.columns if c != target_col and c not in index_columns]

    write_manifest(features_path, {
        "run_id": run_id,
        "source": "engineered finance features",
        "stage": "feature_engineer_finance",
        "feature_columns": feature_columns,
        "index_columns": index_columns,
        "ticker_mapping": ticker_mapping,
        "rows_dropped_warmup": rows_dropped_warmup,
        "train": {"path": str(train_path), "rows": len(train_df), "columns": len(train_df.columns)},
        "test": {"path": str(test_path), "rows": len(test_df), "columns": len(test_df.columns)},
    })

    logger.info(
        "Finance features ready: train=%s test=%s, %d assets, %d warm-up rows dropped",
        train_df.shape, test_df.shape, len(tickers), rows_dropped_warmup,
    )

    return {
        "run_id": run_id,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "feature_count": len(feature_columns),
        "rows_dropped_warmup": rows_dropped_warmup,
    }
