"""Tests for the m6_returns_risk landing-zone staging script. Uses a real,
small yfinance pull (not mocked) — matches this project's convention of
verifying real external-API behavior directly rather than assuming it."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.stage_m6_returns_risk_landing import stage_landing


class TestStageLanding:
    def test_pulls_real_data_and_writes_long_format_csv(self, tmp_path):
        result = stage_landing(tickers=["AAPL", "MSFT"], years=1, dest=tmp_path)

        assert Path(result["output_path"]).exists()
        assert result["ticker_count"] == 2
        assert result["row_count"] > 0

        df = pd.read_csv(result["output_path"])
        assert set(df.columns) == {"Date", "Ticker", "log_return"}
        assert set(df["Ticker"].unique()) == {"AAPL", "MSFT"}
        assert df["log_return"].notna().all()
        # Roughly one row per ticker per month over 1 year — loose bound,
        # not exact, since trading-calendar/API quirks can shift the count
        # by a month or two either way.
        assert 16 <= result["row_count"] <= 26

    def test_raises_when_no_tickers_produce_usable_data(self, tmp_path):
        """Verify the actual yfinance failure mode for a nonexistent ticker
        empirically before asserting on it — don't assume it raises cleanly
        on its own; if yfinance instead returns an empty/all-NaN frame for
        a bad ticker, stage_landing's own empty-rows check must be what
        raises ValueError, not yfinance itself."""
        with pytest.raises(ValueError):
            stage_landing(tickers=["NOT_A_REAL_TICKER_XYZ123"], years=1, dest=tmp_path)
