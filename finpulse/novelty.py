"""Headline de-duplication (novelty), as defined in docs/v3-preregistration.md.

A headline is a *repeat* when the same ticker has an earlier headline within
the previous ``WINDOW`` whose normalised token set has Jaccard similarity of
at least ``THRESHOLD``. Otherwise it is *novel*. A normalised title stored
under ``MULTI_TICKER_MIN`` or more tickers on the same IST calendar day is
flagged ``multi_ticker`` (usually a market round-up rather than company news).
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import deque

import pandas as pd

WINDOW = pd.Timedelta(days=3)
THRESHOLD = 0.6
MULTI_TICKER_MIN = 3
IST = "Asia/Kolkata"

# Kept deliberately small: only words that carry no information about the event.
STOPWORDS = frozenset("""
a an the and or but of to in on at for from by with as is are was were be been
its it this that these those into over after before than vs via amid says said
""".split())

_SUFFIX = re.compile(r"\s+[-|]\s+(?P<tail>[^-|]+)$")
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "‘’“”–—…"})


def normalise_title(title: str) -> str:
    """NFKC, lowercase, publisher suffix dropped, punctuation removed."""
    t = unicodedata.normalize("NFKC", str(title or "")).strip()
    m = _SUFFIX.search(t)
    # Only a short trailing segment is treated as a publisher name
    # ("... - Reuters"); a long one is part of the headline.
    if m and len(m.group("tail").split()) <= 4:
        t = t[: m.start()]
    t = t.lower().translate(_PUNCT)
    return " ".join(t.split())


def tokens(title: str) -> frozenset[str]:
    return frozenset(w for w in normalise_title(title).split() if w not in STOPWORDS)


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity; 0 when either set is empty (nothing to compare)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _timestamps(df: pd.DataFrame) -> pd.Series:
    when = df["published_at"].where(df["published_at"].notna(), df["fetched_at"])
    return pd.to_datetime(when, utc=True, errors="coerce", format="ISO8601")


def mark_novelty(headlines: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``novel`` (bool) and ``multi_ticker`` (bool) columns.

    Needs ``ticker``, ``title``, ``published_at`` and ``fetched_at``. Rows
    without any usable timestamp are kept and treated as novel, since there
    is no earlier headline they can be compared against.
    """
    df = headlines.copy()
    df["_ts"] = _timestamps(df)
    df["_tok"] = df["title"].map(tokens)
    df["novel"] = True

    for _, g in df.sort_values("_ts").groupby("ticker", sort=False):
        seen: deque[tuple[pd.Timestamp, frozenset]] = deque()   # earlier headlines, oldest first
        for idx, ts, tok in zip(g.index, g["_ts"], g["_tok"]):
            if pd.isna(ts):
                continue
            while seen and ts - seen[0][0] > WINDOW:   # rows are in time order: drop what is out of range
                seen.popleft()
            if any(jaccard(tok, prev_tok) >= THRESHOLD for _, prev_tok in seen):
                df.at[idx, "novel"] = False
            seen.append((ts, tok))

    norm = df["title"].map(normalise_title)
    day = df["_ts"].dt.tz_convert(IST).dt.date
    n_tickers = df.assign(_n=norm, _d=day).groupby(["_n", "_d"])["ticker"].transform("nunique")
    df["multi_ticker"] = (n_tickers >= MULTI_TICKER_MIN) & day.notna()
    return df.drop(columns=["_ts", "_tok"])
