"""News and price ingestion from Yahoo Finance (via yfinance).

yfinance's news payload has changed shape several times. Older versions
return flat items (``title``, ``publisher``, ``providerPublishTime`` as a
Unix timestamp); newer ones nest everything under ``content`` (``title``,
``provider.displayName``, ``pubDate`` as an ISO string). ``parse_news_item``
handles both and returns ``None`` for items without a usable title, so
placeholder or error text is never scored as if it were a headline.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from engine.config import validate_ticker
from engine.retry import retry

log = logging.getLogger(__name__)

MAX_TITLE_LEN = 500


def _first(*values: Any) -> Any:
    for v in values:
        if v not in (None, "", {}, []):
            return v
    return None


def _parse_time(value: Any) -> str | None:
    """Return an ISO-8601 UTC timestamp, or None if the value is unusable."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError):
        return None


def parse_news_item(item: Any) -> dict | None:
    """Normalise one yfinance news item to a flat dict, or None if unusable."""
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else {}

    title = _first(item.get("title"), item.get("headline"),
                   content.get("title"), content.get("headline"))
    if not isinstance(title, str) or not title.strip():
        return None
    title = " ".join(title.split())[:MAX_TITLE_LEN]

    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
    publisher = _first(item.get("publisher"), provider.get("displayName"))

    link = _first(
        item.get("link"),
        (content.get("canonicalUrl") or {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else None,
        (content.get("clickThroughUrl") or {}).get("url") if isinstance(content.get("clickThroughUrl"), dict) else None,
    )

    published = _parse_time(_first(item.get("providerPublishTime"),
                                   content.get("pubDate"), content.get("displayTime")))

    return {
        "title": title,
        "publisher": publisher if isinstance(publisher, str) else None,
        "link": link if isinstance(link, str) else None,
        "published_at": published,
    }


@retry(attempts=3, base_delay=2.0)
def _raw_news(ticker: str) -> list:
    import yfinance as yf  # imported lazily so tests can patch it

    news = yf.Ticker(ticker).news
    return news if isinstance(news, list) else []


def fetch_news(ticker: str, limit: int = 20) -> list[dict]:
    """Fetch, normalise and de-duplicate recent headlines for one ticker.

    Returns an empty list (never placeholder text) when nothing usable comes
    back. Network errors are raised after retries so callers can report them.
    """
    ticker = validate_ticker(ticker)
    if limit < 1:
        raise ValueError("limit must be >= 1")

    items, seen = [], set()
    for raw in _raw_news(ticker):
        parsed = parse_news_item(raw)
        if parsed is None:
            continue
        key = parsed["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        parsed["ticker"] = ticker
        items.append(parsed)
        if len(items) >= limit:
            break
    return items


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
