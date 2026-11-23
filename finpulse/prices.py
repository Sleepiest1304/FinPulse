"""Daily price download from Yahoo Finance (via yfinance)."""
from __future__ import annotations

import pandas as pd

from finpulse.config import validate_ticker
from finpulse.retry import retry


@retry(attempts=3, base_delay=2.0)
def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Daily (split/dividend-adjusted) closes, one column per ticker."""
    import yfinance as yf

    tickers = [validate_ticker(t) for t in tickers]
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, group_by="column", threads=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        closes = df["Close"]
    else:  # single ticker, flat columns
        closes = df[["Close"]].rename(columns={"Close": tickers[0]})
    closes = closes.dropna(how="all")
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes
