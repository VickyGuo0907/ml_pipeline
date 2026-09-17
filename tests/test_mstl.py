"""Tests for MSTL seasonality diagnostics (forecasting pipelines only)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.profile import generate_mstl_report


def _write_raw_fixture(raw_dir: Path, run_id: str, n_hours: int = 24 * 21) -> None:
    """3 weeks of synthetic hourly data — long enough for daily+weekly MSTL
    periods, short enough to run fast in a test."""
    run_path = raw_dir / run_id
    run_path.mkdir(parents=True)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h")
    values = 50 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + rng.normal(0, 1, n_hours)
    df = pd.DataFrame({"Datetime": idx.astype(str), "PJME_MW": values})
    filename = "PJME_hourly_test.csv"
    df.to_csv(run_path / filename, index=False)
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump({"files": {filename: {"format": "csv"}}}, f)


def _write_forecast_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        'sources:\n  - name: test\n    path: data/landing\n    format: csv\n'
        'target:\n  name: PJME_MW\n  type: continuous\nproblem_type: forecasting\n'
    )


class TestGenerateMstlReport:
    def test_writes_html_report_with_expected_sections(self, tmp_path):
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-17"

        _write_raw_fixture(raw_dir, run_id)
        _write_forecast_config(config_dir)

        result = generate_mstl_report(raw_dir, run_id, reports_dir, config_dir)

        assert Path(result["report_path"]).exists()
        html = Path(result["report_path"]).read_text()
        assert "trend" in html.lower()
        assert "seasonal" in html.lower()
        assert "residual" in html.lower()

    def test_uses_daily_and_weekly_periods_only_for_a_short_series(self, tmp_path):
        """A 3-week series can't support a meaningful annual (8766h) period —
        confirm the function doesn't crash trying, and reports which periods
        it actually used."""
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-17"

        _write_raw_fixture(raw_dir, run_id)
        _write_forecast_config(config_dir)

        result = generate_mstl_report(raw_dir, run_id, reports_dir, config_dir)

        assert 24 in result["periods_used"]
        assert 8766 not in result["periods_used"]  # too long for 21 days of data

    def test_raises_filenotfounderror_when_manifest_missing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        _write_forecast_config(config_dir)

        with pytest.raises(FileNotFoundError):
            generate_mstl_report(raw_dir, "2026-09-17", tmp_path / "reports", config_dir)
