"""Tests for ADF stationarity diagnostics (finance pipelines only)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.finance.evaluate_finance import _load_adf_summary
from src.profile import generate_adf_report


def _write_raw_fixture(raw_dir: Path, run_id: str, df: pd.DataFrame, filename: str = "m6_returns_panel_test.csv") -> None:
    run_path = raw_dir / run_id
    run_path.mkdir(parents=True)
    df.to_csv(run_path / filename, index=False)
    with open(run_path / "manifest.yaml", "w") as f:
        yaml.dump({"files": {filename: {"format": "csv"}}}, f)


def _write_finance_config(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        'sources:\n  - name: test\n    path: data/landing\n    format: csv\n'
        'target:\n  name: log_return\n  type: continuous\nproblem_type: finance\n'
    )


def _stationary_series(n: int = 60, seed: int = 42) -> np.ndarray:
    """White noise — the textbook stationary series, should reject the unit-root null."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.05, n)


def _nonstationary_series(n: int = 60, seed: int = 42) -> np.ndarray:
    """A cumulative sum (random walk in levels) — the textbook non-stationary
    series, should fail to reject the unit-root null. Used here as a raw
    'log_return'-labeled column purely to exercise the ADF test's ability to
    tell the two shapes apart, not as a claim that real returns look like this."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, 0.05, n))


class TestGenerateAdfReport:
    def test_writes_html_and_yaml_reports(self, tmp_path):
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"

        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        df = pd.DataFrame({
            "Date": list(dates) * 2,
            "Ticker": ["AAPL"] * 60 + ["MSFT"] * 60,
            "log_return": list(_stationary_series()) + list(_stationary_series(seed=7)),
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir)

        result = generate_adf_report(raw_dir, run_id, reports_dir, config_dir)

        assert Path(result["report_path"]).exists()
        assert Path(result["yaml_path"]).exists()
        html = Path(result["report_path"]).read_text()
        assert "AAPL" in html and "MSFT" in html
        assert "stationar" in html.lower()

    def test_yaml_shape_matches_what_load_adf_summary_expects(self, tmp_path):
        """Round-trip: written by generate_adf_report, read back by
        Plan 4's _load_adf_summary — the interface contract between the
        profile stage and the evaluation stage."""
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"

        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        df = pd.DataFrame({"Date": dates, "Ticker": ["AAPL"] * 60, "log_return": list(_stationary_series())})
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir)

        generate_adf_report(raw_dir, run_id, reports_dir, config_dir)

        loaded = _load_adf_summary(reports_dir, run_id)
        assert loaded is not None
        assert "AAPL" in loaded
        assert "p_value" in loaded["AAPL"]
        assert "is_stationary" in loaded["AAPL"]

    def test_distinguishes_stationary_from_nonstationary_series(self, tmp_path):
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        reports_dir = tmp_path / "reports"
        run_id = "2026-09-23"

        dates = pd.date_range("2020-01-01", periods=80, freq="MS")
        df = pd.DataFrame({
            "Date": list(dates) * 2,
            "Ticker": ["STATIONARY"] * 80 + ["NONSTATIONARY"] * 80,
            "log_return": list(_stationary_series(n=80)) + list(_nonstationary_series(n=80)),
        })
        _write_raw_fixture(raw_dir, run_id, df)
        _write_finance_config(config_dir)

        result = generate_adf_report(raw_dir, run_id, reports_dir, config_dir)

        assert result["adf_results"]["STATIONARY"]["is_stationary"] is True
        assert result["adf_results"]["NONSTATIONARY"]["is_stationary"] is False

    def test_raises_filenotfounderror_when_manifest_missing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        config_dir = tmp_path / "config"
        _write_finance_config(config_dir)

        with pytest.raises(FileNotFoundError):
            generate_adf_report(raw_dir, "2026-09-23", tmp_path / "reports", config_dir)
