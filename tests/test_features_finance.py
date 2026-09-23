"""Tests for the finance feature engineering stage: per-asset lag/volatility
features (no cross-asset leakage), ticker/date numeric encoding, and the
chronological per-asset train/test split."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.finance.features_finance import engineer_finance_features
from src.schemas.features import build_finance_features_schema


def _write_interim_fixture(interim_dir: Path, run_id: str, df: pd.DataFrame) -> Path:
    """Write a cleaned long-format parquet + manifest.yaml into interim_dir/run_id/,
    mirroring what clean_finance_data (Task 1) produces — independent of Task 1's
    implementation, so this test file exercises engineer_finance_features in isolation.
    """
    run_path = interim_dir / run_id
    run_path.mkdir(parents=True)
    output_path = run_path / "cleaned.parquet"
    df.to_parquet(output_path, index=False)
    manifest = {"run_id": run_id, "stage": "clean_finance", "output_path": str(output_path)}
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    return run_path


def _write_finance_config(
    config_dir: Path,
    lag_months: list[int],
    volatility_window_months: int,
    train_test_split: float = 0.8,
) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "sources:\n  - name: test\n    path: data/landing\n    format: csv\n"
        "target:\n  name: log_return\n  type: continuous\nproblem_type: finance\n"
        f"train_test_split: {train_test_split}\n"
    )
    (config_dir / "features.yaml").write_text(
        f"lag_months: {lag_months}\nvolatility_window_months: {volatility_window_months}\n"
    )


def _synthetic_panel(ticker_values: dict[str, list[float]], start: str = "2020-01") -> pd.DataFrame:
    """Build a long-format panel: {ticker: [monthly returns in chronological order]}."""
    rows = []
    for ticker, values in ticker_values.items():
        dates = pd.period_range(start, periods=len(values), freq="M").to_timestamp()
        for date, value in zip(dates, values):
            rows.append({"Date": date, "Ticker": ticker, "log_return": value})
    return pd.DataFrame(rows)


class TestLagFeatureNoCrossAssetLeakage:
    """The single highest-risk bug class in this task: a naive global .shift(1)
    (not grouped by Ticker) would leak one asset's last value into the next
    asset's first row after concatenation."""

    def test_lag_1_return_never_uses_another_tickers_value(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        # AAPL ends its series with 0.99 (a value MSFT never has). If lag_1_return
        # leaked across the Ticker boundary via a naive concatenated .shift(1),
        # MSFT's first surviving row would show lag_1_return == 0.99 instead of
        # MSFT's own prior value.
        df = _synthetic_panel({
            "AAPL": [0.01, 0.02, 0.03, 0.04, 0.99],
            "MSFT": [0.10, 0.11, 0.12, 0.13, 0.14],
        })
        _write_interim_fixture(interim_dir, run_id, df)
        _write_finance_config(config_dir, lag_months=[1], volatility_window_months=2)

        engineer_finance_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        with open(features_dir / run_id / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        msft_code = manifest["ticker_mapping"]["MSFT"]

        full = pd.concat([
            pd.read_parquet(features_dir / run_id / "train.parquet"),
            pd.read_parquet(features_dir / run_id / "test.parquet"),
        ])
        msft_rows = full[full["ticker_encoded"] == msft_code].sort_values("date_ordinal")
        # MSFT's earliest surviving row's lag is MSFT's own prior value (0.11), never AAPL's 0.99.
        assert msft_rows["lag_1_return"].iloc[0] == pytest.approx(0.11)
        assert not (full["lag_1_return"] == pytest.approx(0.99)).any()


class TestTrailingVolatility:
    def test_rolling_std_excludes_current_row_matches_hand_computed_value(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        df = _synthetic_panel({"AAPL": values})
        _write_interim_fixture(interim_dir, run_id, df)
        _write_finance_config(config_dir, lag_months=[1], volatility_window_months=3)

        engineer_finance_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        full = pd.concat([
            pd.read_parquet(features_dir / run_id / "train.parquet"),
            pd.read_parquet(features_dir / run_id / "test.parquet"),
        ])
        # Month index 4 (0-indexed, value 0.05): trailing_3m_vol excludes the
        # current row -> std of shift(1)'s trailing 3 values = values at month
        # indices 1,2,3 = (0.02, 0.03, 0.04).
        # Warm-up dropna removes month indices 0,1,2 (trailing_3m_vol needs 3
        # preceding shifted values, so it's NaN there), so the earliest
        # surviving row is month index 3, not 0 -- full["date_ordinal"].min()
        # already corresponds to month index 3, so month index 4 is min() + 1.
        target_date_ordinal = full["date_ordinal"].min() + 1
        row = full[full["date_ordinal"] == target_date_ordinal].iloc[0]
        expected = pd.Series([0.02, 0.03, 0.04]).std()
        assert row["trailing_3m_vol"] == pytest.approx(expected)


class TestTickerEncoding:
    def test_ticker_encoded_is_numeric_and_matches_manifest_mapping(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        df = _synthetic_panel({
            "AAPL": [0.01, 0.02, 0.03, 0.04],
            "MSFT": [0.10, 0.11, 0.12, 0.13],
        })
        _write_interim_fixture(interim_dir, run_id, df)
        _write_finance_config(config_dir, lag_months=[1], volatility_window_months=2)

        engineer_finance_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        with open(features_dir / run_id / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        # Sorted alphabetically: AAPL -> 0, MSFT -> 1
        assert manifest["ticker_mapping"] == {"AAPL": 0, "MSFT": 1}

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        assert pd.api.types.is_integer_dtype(train_df["ticker_encoded"])
        assert set(train_df["ticker_encoded"].unique()).issubset({0, 1})


class TestChronologicalSplitPerAsset:
    def test_train_test_no_overlap_per_asset(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        df = _synthetic_panel({
            "AAPL": [0.01 * i for i in range(20)],
            "MSFT": [0.02 * i for i in range(15)],  # different length -> different date range end
        })
        _write_interim_fixture(interim_dir, run_id, df)
        _write_finance_config(config_dir, lag_months=[1], volatility_window_months=2, train_test_split=0.8)

        engineer_finance_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        test_df = pd.read_parquet(features_dir / run_id / "test.parquet")

        for code in train_df["ticker_encoded"].unique():
            asset_train = train_df[train_df["ticker_encoded"] == code]
            asset_test = test_df[test_df["ticker_encoded"] == code]
            if len(asset_test) == 0:
                continue
            assert asset_train["date_ordinal"].max() < asset_test["date_ordinal"].min()


class TestSchemaCompliance:
    def test_train_output_validates_against_finance_schema(self, tmp_path):
        """Real output, not a hand-built frame, must satisfy Plan 1's build_finance_features_schema."""
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        df = _synthetic_panel({"AAPL": [0.01 * i for i in range(10)]})
        _write_interim_fixture(interim_dir, run_id, df)
        _write_finance_config(config_dir, lag_months=[1], volatility_window_months=2)

        engineer_finance_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        train_df = pd.read_parquet(features_dir / run_id / "train.parquet")
        schema = build_finance_features_schema("log_return")
        validated = schema.validate(train_df)
        assert len(validated) == len(train_df)


class TestManifestAndReturn:
    def test_manifest_and_return_shape(self, tmp_path):
        interim_dir = tmp_path / "interim"
        features_dir = tmp_path / "features"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        df = _synthetic_panel({"AAPL": [0.01 * i for i in range(10)]})
        _write_interim_fixture(interim_dir, run_id, df)
        _write_finance_config(config_dir, lag_months=[1], volatility_window_months=2)

        result = engineer_finance_features(interim_dir, features_dir, run_id, config_dir=config_dir)

        assert result["train_path"].endswith("train.parquet")
        assert result["test_path"].endswith("test.parquet")
        assert result["rows_dropped_warmup"] > 0  # lag_1_return/trailing_2m_vol need warm-up rows dropped

        with open(features_dir / run_id / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["stage"] == "feature_engineer_finance"
        assert "log_return" not in manifest["feature_columns"]
        assert manifest["ticker_mapping"] == {"AAPL": 0}
