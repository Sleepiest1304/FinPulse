"""Static configuration: tickers, paths and model names."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# SQLite database holding every scored headline. Override with FINPULSE_DB.
DB_PATH = Path(os.environ.get("FINPULSE_DB", ROOT / "data" / "finpulse.db"))

RESULTS_DIR = ROOT / "results"

FINBERT_MODEL = "ProsusAI/finbert"

# Nifty 50 constituents (Yahoo Finance symbols). Index membership changes
# twice a year; this list is a snapshot and should be checked against the
# NSE factsheet when it is updated. Last reviewed: 2026-11.
NIFTY50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
    "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS",
    "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "MAXHEALTH.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TMPV.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]

NIFTY50_SET = frozenset(NIFTY50)


def validate_ticker(ticker: str) -> str:
    """Normalise a ticker and check that it is in the configured universe.

    Raises ValueError for anything else, so user input never reaches the
    network layer unchecked.
    """
    if not isinstance(ticker, str):
        raise ValueError(f"Ticker must be a string, got {type(ticker).__name__}")
    t = ticker.strip().upper()
    if t and not t.endswith(".NS"):
        t += ".NS"
    if t not in NIFTY50_SET:
        raise ValueError(f"Unknown Nifty 50 ticker: {ticker!r}")
    return t
