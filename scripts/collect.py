"""Fetch the latest headlines, score them with FinBERT and store them in SQLite.

Usage:
    python scripts/collect.py                    # all Nifty 50 tickers
    python scripts/collect.py RELIANCE TCS.NS    # a subset
    python scripts/collect.py --db other.db

Already-stored headlines (same ticker + title) are not re-scored. The exit
code is non-zero only if every requested ticker failed to fetch.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import storage  # noqa: E402
from engine.config import DB_PATH, FINBERT_MODEL, NIFTY50, validate_ticker  # noqa: E402
from engine.data_fetch import fetch_news  # noqa: E402

log = logging.getLogger("collect")


def collect(tickers: list[str], db: Path, pause: float = 1.0, scorer=None) -> dict:
    """Run one collection pass. ``scorer`` is injectable for tests."""
    if scorer is None:
        from engine.sentiment import get_sentiment as scorer

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", help="defaults to all Nifty 50")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds to wait between tickers")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        tickers = [validate_ticker(t) for t in args.tickers] or list(NIFTY50)
    except ValueError as exc:
        parser.error(str(exc))

    summary = collect(tickers, args.db, args.pause)
    log.info("done: %d fetched, %d new rows, %d failed tickers %s",
             summary["fetched"], summary["new"], len(summary["failed"]), summary["failed"])
    return 1 if len(summary["failed"]) == len(tickers) else 0


if __name__ == "__main__":
    sys.exit(main())
