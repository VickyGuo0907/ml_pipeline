"""Tests for the forecasting clean stage: gap detection and the full
clean_forecast_data stage function."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.forecasting.clean_forecast import _detect_gaps, clean_forecast_data


def _write_raw_fixture(raw_dir: Path, run_id: str, df: pd.DataFrame, filename: str = "PJME_hourly_test.csv") -> Path:
    """Write a raw CSV + manifest.yaml into raw_dir/run_id/, mirroring what ingest_files produces."""
    run_path = raw_dir / run_id
    run_path.mkdir(parents=True)
    df.to_csv(run_path / filename, index=False)
    manifest = {"run_id": run_id, "files": {filename: {"format": "csv"}}}
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    return run_path


def _write_forecast_config(config_dir: Path, max_gap_hours: int = 3, fill_strategy: str = "interpolate") -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        'sources:\n  - name: test\n    path: data/landing\n    format: csv\n'
        'target:\n  name: PJME_MW\n  type: continuous\nproblem_type: forecasting\n'
    )
    (config_dir / "cleaning.yaml").write_text(
        f"max_gap_hours: {max_gap_hours}\nfill_strategy: {fill_strategy}\n"
    )


class TestDetectGaps:
    """Unit tests for the pure gap-grouping helper."""

    def test_no_gaps_returns_empty_list(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="h")
        is_missing = pd.Series([False] * 5, index=idx)
        assert _detect_gaps(is_missing) == []

    def test_single_contiguous_gap(self):
        idx = pd.date_range("2020-01-01", periods=6, freq="h")
        is_missing = pd.Series([False, True, True, True, False, False], index=idx)
        gaps = _detect_gaps(is_missing)
        assert len(gaps) == 1
        start, end, length = gaps[0]
        assert start == idx[1]
        assert end == idx[3]
        assert length == 3

    def test_two_disjoint_gaps(self):
        idx = pd.date_range("2020-01-01", periods=8, freq="h")
        is_missing = pd.Series([False, True, False, False, True, True, False, False], index=idx)
        gaps = _detect_gaps(is_missing)
        assert len(gaps) == 2
        assert gaps[0] == (idx[1], idx[1], 1)
        assert gaps[1] == (idx[4], idx[5], 2)


class TestCleanForecastData:
    """Tests for the full clean_forecast_data stage function."""

    def test_dedup_and_short_gap_interpolated(self, tmp_path):
        """A duplicate timestamp is dropped; a 2h gap (<= max_gap_hours=3) is linearly interpolated."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        # Hours 0,1,2 present; hours 3,4 missing (2h gap); hour 5 present; hour 1 duplicated.
        df = pd.DataFrame({
            "Datetime": [
                "2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 01:00:00",
                "2020-01-01 02:00:00", "2020-01-01 05:00:00",
            ],
            "PJME_MW": [10.0, 20.0, 20.0, 30.0, 60.0],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_forecast_config(config_dir, max_gap_hours=3, fill_strategy="interpolate")

        result = clean_forecast_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        assert result["duplicates_dropped"] == 1
        assert result["gaps_found"] == 1
        assert result["gaps_filled_hours"] == 2
        assert result["row_count"] == 6  # hours 0-5 inclusive, complete hourly range

        cleaned = pd.read_parquet(result["output_path"])
        assert isinstance(cleaned.index, pd.DatetimeIndex)
        assert cleaned.index.name == "Datetime"
        assert cleaned["PJME_MW"].isna().sum() == 0
        # Linear interpolation between hour 2 (30.0) and hour 5 (60.0) over 3 steps: 40, 50 at hours 3, 4
        assert cleaned.loc["2020-01-01 03:00:00", "PJME_MW"] == pytest.approx(40.0)
        assert cleaned.loc["2020-01-01 04:00:00", "PJME_MW"] == pytest.approx(50.0)

    def test_long_gap_raises(self, tmp_path):
        """A gap longer than max_gap_hours fails loud instead of being silently filled."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        # Hour 0 present, then a 5h gap (hours 1-5), hour 6 present. max_gap_hours=3.
        df = pd.DataFrame({
            "Datetime": ["2020-01-01 00:00:00", "2020-01-01 06:00:00"],
            "PJME_MW": [10.0, 70.0],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_forecast_config(config_dir, max_gap_hours=3, fill_strategy="interpolate")

        with pytest.raises(ValueError, match="exceeds max_gap_hours"):
            clean_forecast_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

    def test_ffill_strategy_used_when_configured(self, tmp_path):
        """fill_strategy=ffill carries the last known value forward instead of interpolating."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        # 1h gap between hour 0 (10.0) and hour 2 (30.0). Interpolate would give 20.0; ffill gives 10.0.
        df = pd.DataFrame({
            "Datetime": ["2020-01-01 00:00:00", "2020-01-01 02:00:00"],
            "PJME_MW": [10.0, 30.0],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_forecast_config(config_dir, max_gap_hours=3, fill_strategy="ffill")

        result = clean_forecast_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        cleaned = pd.read_parquet(result["output_path"])
        assert cleaned.loc["2020-01-01 01:00:00", "PJME_MW"] == pytest.approx(10.0)

    def test_manifest_written_with_gap_and_dedup_stats(self, tmp_path):
        """The interim manifest records the stats Task 2 and future readers rely on."""
        raw_dir = tmp_path / "raw"
        interim_dir = tmp_path / "interim"
        config_dir = tmp_path / "config"
        run_id = "2026-09-16"

        df = pd.DataFrame({
            "Datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "PJME_MW": [10.0, 20.0],
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_forecast_config(config_dir, max_gap_hours=3, fill_strategy="interpolate")

        clean_forecast_data(raw_dir, interim_dir, run_id, config_dir=config_dir)

        with open(interim_dir / run_id / "manifest.yaml") as f:
            manifest = yaml.safe_load(f)
        assert manifest["stage"] == "clean_forecast"
        assert "output_path" in manifest
        assert manifest["duplicates_dropped"] == 0
        assert manifest["gaps_found"] == 0
        assert manifest["fill_strategy"] == "interpolate"
