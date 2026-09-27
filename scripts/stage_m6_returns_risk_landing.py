"""Stage the m6_returns_risk landing zone by pulling monthly log-returns for
a representative subset of liquid US-listed ETFs/stocks via yfinance.

Real market data, not synthetic — the M6 Competition brief calls for a
diverse asset panel; a representative ~15-ticker subset (rather than the
full 100-asset M6 universe) avoids a large rate-limited fetch surface and
delisting/gap handling for this POC's scope, while staying honest with
real market data rather than synthetic noise (see
docs/superpowers/specs/2026-09-22-m6-returns-risk-design.md's "Decisions
made during brainstorming").
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# 15 liquid, well-known tickers spanning broad-market ETFs and large-cap
# equities across several sectors: broad market (SPY, QQQ), tech (AAPL,
# MSFT, GOOGL, AMZN), financials (JPM, V), energy (XOM), healthcare (JNJ,
# UNH), consumer (PG, KO, WMT, DIS).
DEFAULT_TICKERS: list[str] = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "V",
    "XOM", "JNJ", "UNH", "PG", "KO", "WMT", "DIS",
]
DEFAULT_YEARS = 8  # within the spec's 5-10 year window
DEFAULT_DEST = Path("data/m6_returns_risk/landing")
OUTPUT_FILENAME = "m6_returns_panel.csv"


def stage_landing(tickers: list[str], years: int, dest: Path) -> dict[str, Any]:
    """Pull monthly close prices for `tickers` over the trailing `years`
    years via yfinance, compute log-returns, and write a long-format CSV
    (Date, Ticker, log_return) to dest.

    Args:
        tickers: Ticker symbols to pull.
        years: How many trailing years of monthly data to pull.
        dest: Landing-zone directory to write into (created if absent).

    Returns:
        Dict with output_path, ticker_count, row_count.

    Raises:
        ValueError: If yfinance returns no usable data for any ticker.
    """
    # threads=False: yfinance's default threaded downloader has each worker
    # thread write to a shared local sqlite cookie/crumb cache with no lock
    # retry, which intermittently raises "database is locked" under CI's
    # tighter scheduling (see yfinance issue reports on concurrent
    # yf.download calls). Sequential fetching for our ~15-ticker panel costs
    # a couple seconds and removes the flake entirely.
    raw = yf.download(
        tickers, period=f"{years}y", interval="1mo", progress=False, auto_adjust=True, threads=False
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no data for tickers={tickers}, period={years}y")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    today = pd.Timestamp.today()
    current_month_start = pd.Timestamp(year=today.year, month=today.month, day=1)

    rows = []
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        prices = close[ticker].dropna()
        if len(prices) and prices.index[-1] >= current_month_start:
            # yfinance's monthly bar for the in-progress month is a partial
            # close (latest trade, not a real month-end close) — drop it so
            # this script's output for a given completed month is stable
            # regardless of which day of the current month the script runs.
            prices = prices.iloc[:-1]
        if len(prices) < 2:
            continue
        log_returns = np.log(prices / prices.shift(1)).dropna()
        for date, value in log_returns.items():
            rows.append({"Date": date.strftime("%Y-%m-%d"), "Ticker": ticker, "log_return": float(value)})

    if not rows:
        raise ValueError("No usable log-return rows produced from yfinance data for any ticker.")

    panel = pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    dest.mkdir(parents=True, exist_ok=True)
    output_path = dest / OUTPUT_FILENAME
    panel.to_csv(output_path, index=False)

    return {
        "output_path": str(output_path),
        "ticker_count": panel["Ticker"].nunique(),
        "row_count": len(panel),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    result = stage_landing(args.tickers, args.years, args.dest)
    print(f"Staged {result['row_count']} rows for {result['ticker_count']} tickers -> {result['output_path']}")


if __name__ == "__main__":
    main()
