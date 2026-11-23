"""One collection pass: fetch headlines, score the new ones, store them."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from finpulse import storage
from finpulse.config import FINBERT_MODEL
from finpulse.news import fetch_news

log = logging.getLogger(__name__)


def collect(tickers: list[str], db: Path, pause: float = 1.0, scorer=None) -> dict:
    """Fetch and score headlines for ``tickers`` and save them to ``db``.

    Headlines already stored (same ticker + title) are not re-scored. A ticker
    whose fetch fails after retries is recorded in ``failed`` and skipped.
    ``scorer`` defaults to FinBERT and is injectable for tests.
    """
    if scorer is None:
        from finpulse.sentiment import get_sentiment as scorer

    summary = {"fetched": 0, "new": 0, "failed": []}
    for i, ticker in enumerate(tickers):
        if i and pause:
            time.sleep(pause)  # be polite to Yahoo
        try:
            items = fetch_news(ticker)
        except Exception as exc:  # network / API failure after retries
            log.error("%s: fetch failed: %s", ticker, exc)
            summary["failed"].append(ticker)
            continue
        summary["fetched"] += len(items)

        known = storage.existing_titles(ticker, db)
        fresh = [it for it in items if it["title"] not in known]
        if not fresh:
            log.info("%s: %d headlines, none new", ticker, len(items))
            continue

        results = scorer([it["title"] for it in fresh])
        rows = [{**it, **res} for it, res in zip(fresh, results)]
        written = storage.save_scored(rows, FINBERT_MODEL, db)
        summary["new"] += written
        log.info("%s: %d headlines, %d new", ticker, len(items), written)
    return summary
