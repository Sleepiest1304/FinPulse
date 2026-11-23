"""Run the sentiment-vs-returns backtest on the stored headlines.

Usage:
    python scripts/backtest.py [--db data/finpulse.db] [--out results]

Writes results/backtest_summary.csv and results/backtest_panel.csv and
prints the pooled rows. Numbers are whatever the data says; nothing is tuned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from engine import storage  # noqa: E402
from engine.backtest import build_panel, summarize  # noqa: E402
from engine.config import DB_PATH, RESULTS_DIR  # noqa: E402
from engine.data_fetch import fetch_prices  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    heads = storage.load_headlines(path=args.db)
    if heads.empty:
        print("No headlines stored yet. Run scripts/collect.py first.")
        return 1

    times = pd.to_datetime(heads["published_at"].fillna(heads["fetched_at"]),
                           utc=True, errors="coerce", format="ISO8601")
    start = (times.min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    prices = fetch_prices(sorted(heads["ticker"].unique()), start=start)

    panel = build_panel(heads, prices)
    summary = summarize(panel)

    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out / "backtest_panel.csv", index=False)
    summary.to_csv(args.out / "backtest_summary.csv", index=False)

    print(f"{len(heads)} headlines, {heads['ticker'].nunique()} tickers, "
          f"{times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}")
    with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
        print(summary[summary["ticker"] == "ALL"].to_string(index=False))
    print(f"Wrote {args.out / 'backtest_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
