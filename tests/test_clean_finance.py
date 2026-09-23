"""Tests for the finance clean stage: per-asset gap detection and the full
clean_finance_data stage function."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.finance.clean_finance import _detect_gaps, clean_finance_data


def _write_raw_fixture(raw_dir: Path, run_id: str, df: pd.DataFrame, filename: str = "m6_returns_panel_test.csv") -> Path:
    """Write a raw CSV + manifest.yaml into raw_dir/run_id/, mirroring what ingest_files produces."""
    run_path = raw_dir / run_id
    run_path.mkdir(parents=True)
    df.to_csv(run_path / filename, index=False)
    manifest = {"run_id": run_id, "files": {filename: {"format": "csv"}}}
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    return run_path


def _write_finance_config(config_dir: Path, max_gap_months: int = 2, fill_strategy: str = "interpolate") -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        'sources:\n  - name: test\n    path: data/landing\n    format: csv\n'
        'target:\n  name: log_return\n  type: continuous\nproblem_type: finance\n'
    )
    (config_dir / "cleaning.yaml").write_text(
        f"max_gap_months: {max_gap_months}\nfill_strategy: {fill_strategy}\n"
    )


class TestDetectGaps:
    """Unit tests for the pure gap-grouping helper, keyed on a monthly PeriodIndex."""

    def test_no_gaps_returns_empty_list(self):
        idx = pd.period_range("2020-01", periods=5, freq="M")
        is_missing = pd.Series([False] * 5, index=idx)
        assert _detect_gaps(is_missing) == []

    def test_single_contiguous_gap(self):
        idx = pd.period_range("2020-01", periods=6, freq="M")
        is_missing = pd.Series([False, True, True, True, False, False], index=idx)
        gaps = _detect_gaps(is_missing)
        assert len(gaps) == 1
        start, end, length = gaps[0]
        assert start == idx[1]
        assert end == idx[3]
        assert length == 3

    def test_two_disjoint_gaps(self):
        idx = pd.period_range("2020-01", periods=8, freq="M")
        is_missing = pd.Series([False, True, False, False, True, True, False, False], index=idx)
        gaps = _detect_gaps(is_missing)
        assert len(gaps) == 2
        assert gaps[0] == (idx[1], idx[1], 1)
        assert gaps[1] == (idx[4], idx[5], 2)


class TestCleanFinanceData:
    """Tests for the full clean_finance_data stage function."""

    def test_dedup_and_short_gap_interpolated(self, tmp_path):
        """A duplicate (Date, Ticker) row is averaged; a 1-month gap (<= max_gap_months=2) is interpolated."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        # AAPL: Jan, Feb(dup x2), Mar, then Apr missing, May present.
        df = pd.DataFrame({
            "Date": ["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01", "2020-05-01"],
            "Ticker": ["AAPL"] * 5,
            "log_return": [0.01, 0.02, 0.04, 0.03, 0.07],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir, max_gap_months=2, fill_strategy="interpolate")

        result = clean_finance_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        assert result["duplicates_dropped"] == 1
        assert result["gaps_found"] == 1
        assert result["gaps_filled_months"] == 1

        cleaned = pd.read_parquet(result["output_path"])
        assert cleaned["log_return"].isna().sum() == 0
        # Feb's duplicate averages to (0.02 + 0.04) / 2 = 0.03
        feb_row = cleaned[(cleaned["Ticker"] == "AAPL") & (cleaned["Date"] == "2020-02-01")]
        assert feb_row["log_return"].iloc[0] == pytest.approx(0.03)
        # Apr (missing) linearly interpolated between Mar (0.03) and May (0.07): 0.05
        apr_row = cleaned[(cleaned["Ticker"] == "AAPL") & (cleaned["Date"] == "2020-04-01")]
        assert apr_row["log_return"].iloc[0] == pytest.approx(0.05)

    def test_long_gap_raises(self, tmp_path):
        """A gap longer than max_gap_months fails loud instead of being silently filled."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        # AAPL: Jan present, then a 4-month gap (Feb-May), Jun present. max_gap_months=2.
        df = pd.DataFrame({
            "Date": ["2020-01-01", "2020-06-01"],
            "Ticker": ["AAPL", "AAPL"],
            "log_return": [0.01, 0.05],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir, max_gap_months=2, fill_strategy="interpolate")

        with pytest.raises(ValueError, match="exceeds max_gap_months"):
            clean_finance_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

    def test_ffill_strategy_used_when_configured(self, tmp_path):
        """fill_strategy=ffill carries the last known value forward instead of interpolating."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        # 1-month gap between Jan (0.01) and Mar (0.05). Interpolate would give 0.03; ffill gives 0.01.
        df = pd.DataFrame({
            "Date": ["2020-01-01", "2020-03-01"],
            "Ticker": ["AAPL", "AAPL"],
            "log_return": [0.01, 0.05],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir, max_gap_months=2, fill_strategy="ffill")

        result = clean_finance_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        cleaned = pd.read_parquet(result["output_path"])
        feb_row = cleaned[(cleaned["Ticker"] == "AAPL") & (cleaned["Date"] == "2020-02-01")]
        assert feb_row["log_return"].iloc[0] == pytest.approx(0.01)

    def test_gap_in_one_asset_does_not_affect_another(self, tmp_path):
        """A gap/dedup issue in one asset's series never touches another asset's rows —
        the single most important correctness property of per-asset processing."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        df = pd.DataFrame({
            "Date": [
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-01-01", "2020-02-01", "2020-03-01",
            ],
            "Ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "log_return": [0.01, 0.02, 0.03, 0.10, 0.11, 0.12],
        })
        df = df.drop(index=1).reset_index(drop=True)  # drop AAPL's Feb row -> a 1-month gap for AAPL only
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir, max_gap_months=2, fill_strategy="interpolate")

        result = clean_finance_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        assert result["gaps_found"] == 1  # only AAPL's gap
        cleaned = pd.read_parquet(result["output_path"])
        msft_rows = cleaned[cleaned["Ticker"] == "MSFT"].sort_values("Date")
        assert len(msft_rows) == 3
        assert msft_rows["log_return"].tolist() == pytest.approx([0.10, 0.11, 0.12])

    def test_manifest_written_with_gap_and_dedup_stats(self, tmp_path):
        """The interim manifest records the stats Task 2 and future readers rely on."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-23"

        df = pd.DataFrame({
            "Date": ["2020-01-01", "2020-02-01"],
            "Ticker": ["AAPL", "AAPL"],
            "log_return": [0.01, 0.02],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir, max_gap_months=2, fill_strategy="interpolate")

        clean_finance_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        with open(interim_dir / run_id / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["stage"] == "clean_finance"
        assert "output_path" in manifest
        assert manifest["duplicates_dropped"] == 0
        assert manifest["gaps_found"] == 0
        assert manifest["fill_strategy"] == "interpolate"
        assert "AAPL" in manifest["per_asset_stats"]
