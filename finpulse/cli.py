"""Command-line interface: ``python -m finpulse <command>``.

Commands:
    collect  [TICKER ...] [--db PATH] [--pause SECONDS]
        Fetch headlines, score new ones with FinBERT, store them in SQLite.
        Defaults to all Nifty 50 tickers. Exits 1 only if every ticker fails.
    backtest [--db PATH] [--out DIR]
        Compare stored sentiment with forward returns; writes
        backtest_summary.csv and backtest_panel.csv.
    evaluate [--subsets ...] [--models ...] [--out DIR] [--labelled CSV]
        Compare FinBERT, VADER and a majority baseline on Financial
        PhraseBank; writes model_eval.json and model_eval.md. With
        --labelled, instead compare majority, VADER, FinBERT and a zero-shot
        model on a labelled set (labels drafted with an LLM, reviewed by the author); writes model_eval_indian.{json,md}.
        --alternatives reruns the scoring with borderline labels swapped.
    tagger-eval --labelled CSV [--alternatives CSV] [--out DIR]
        Check the rule-based event-type tagger against labelled headlines;
        writes event_tagger_eval.{json,md}.
    eventstudy --prices CSV --earnings CSV [--db PATH] [--out DIR] [--perm N]
        Run the pre-registered v3 tests (docs/v3-preregistration.md) on
        locally saved price and earnings files; writes v3_*.csv summaries.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from finpulse.config import DB_PATH, FINBERT_MODEL, NIFTY50, RESULTS_DIR, validate_ticker


def cmd_collect(args) -> int:
    from finpulse.collect import collect

    summary = collect(args.tickers, args.db, args.pause)
    logging.info("done: %d fetched, %d new rows, %d failed tickers %s",
                 summary["fetched"], summary["new"], len(summary["failed"]), summary["failed"])
    return 1 if len(summary["failed"]) == len(args.tickers) else 0


def cmd_backtest(args) -> int:
    import pandas as pd

    from finpulse import storage
    from finpulse.backtest import build_panel, summarize
    from finpulse.prices import fetch_prices

    heads = storage.load_headlines(path=args.db)
    if heads.empty:
        print("No headlines stored yet. Run `python -m finpulse collect` first.")
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


def cmd_evaluate(args) -> int:
    if args.labelled:
        return _evaluate_labelled(args)
    from finpulse.evaluation import CAVEAT, DATASET, evaluate, to_markdown

    rows = evaluate(args.subsets, args.models)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dataset": DATASET, "finbert_model": FINBERT_MODEL, "caveat": CAVEAT,
           "results": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_eval.json").write_text(json.dumps(out, indent=2))
    table = to_markdown(rows)
    (args.out / "model_eval.md").write_text(table + f"\n\n> {CAVEAT}\n")
    print(table)
    return 0


def _evaluate_labelled(args) -> int:
    from collections import Counter

    import pandas as pd

    from finpulse import evaluation as ev

    df, dropped = ev.load_labelled_frame(args.labelled)
    if df.empty:
        print("No usable labelled headlines.")
        return 1
    texts, y = df["title"].tolist(), df["label"].tolist()
    models = [m for m in args.models if m != "majority"]
    models = ["majority"] + models if "majority" in args.models else models
    if args.models == ["majority", "vader", "finbert"]:   # default: include zero-shot too
        models.append("zeroshot")
    preds = ev.predict_labelled(texts, models)
    rows = ev.score_labelled(y, models, preds)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dataset": f"{args.labelled.name}: GDELT headlines from Indian outlets",
           "n": len(y), "dropped": dropped, "label_counts": dict(Counter(y)),
           "finbert_model": FINBERT_MODEL, "zeroshot_model": ev.ZEROSHOT_MODEL,
           "caveat": ev.LABELLED_CAVEAT, "results": rows}
    table = ev.to_markdown_ci(rows)
    md = table + f"\n\nn = {len(y)} ({dict(Counter(y))}), {dropped} rows dropped.\n"
    if args.alternatives:
        alt, swapped = ev.apply_alternatives(df, pd.read_csv(args.alternatives, dtype=str),
                                             "sentiment", ["record_id"], "label", ["id"])
        alt_rows = ev.score_labelled(alt["label"].tolist(), models, preds)
        cmp = ev.compare_rankings(rows, alt_rows)
        out["sensitivity"] = {"alternatives": args.alternatives.name, "swapped": swapped,
                              "results": alt_rows, "ranking": cmp}
        md += (f"\n### Sensitivity: {swapped} borderline labels swapped to their alternative\n\n"
               + ev.to_markdown_ci(alt_rows) + "\n\n"
               + ("Model ranking changes: " + ", ".join(m for m in ("accuracy", "macro_f1")
                                                         if cmp[m]["changed"])
                  if cmp["changed"] else "Model ranking is unchanged on accuracy and macro F1.")
               + "\n")
    md += f"\n> {ev.LABELLED_CAVEAT}\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_eval_indian.json").write_text(json.dumps(out, indent=2))
    (args.out / "model_eval_indian.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def cmd_tagger_eval(args) -> int:
    import pandas as pd

    from finpulse import evaluation as ev

    df = pd.read_csv(args.labelled, dtype=str)
    res = ev.evaluate_tagger(df, args.label_col)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dataset": args.labelled.name, "caveat": ev.LABELLED_CAVEAT, **res}
    md = "### Event-type tagger vs labels\n\n" + ev.tagger_markdown(res) + "\n"
    if args.alternatives:
        alt, swapped = ev.apply_alternatives(df, pd.read_csv(args.alternatives, dtype=str),
                                             "event_type", ["ticker", "title"], args.label_col,
                                             ["id", "title"])
        alt_res = ev.evaluate_tagger(alt, args.label_col)
        out["sensitivity"] = {"alternatives": args.alternatives.name, "swapped": swapped, **alt_res}
        md += (f"\n### Sensitivity: {swapped} borderline labels swapped to their alternative\n\n"
               + ev.tagger_markdown(alt_res) + "\n")
    md += f"\n> {ev.LABELLED_CAVEAT}\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "event_tagger_eval.json").write_text(json.dumps(out, indent=2))
    (args.out / "event_tagger_eval.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def cmd_eventstudy(args) -> int:
    import pandas as pd

    from finpulse import storage
    from finpulse.eventstudy import build
    from finpulse.v3tests import confirmatory, exploratory, robustness

    heads = storage.load_headlines(path=args.db)
    if heads.empty:
        print("No headlines stored yet.")
        return 1
    prices = pd.read_csv(args.prices, parse_dates=["date"]).set_index("date")
    earnings = pd.read_csv(args.earnings)
    data = build(heads, prices, earnings)
    ev, base = data["events"], data["base"]

    conf = confirmatory(ev, base, n_perm=args.perm)
    expl = exploratory(ev)
    rob = robustness(ev, base, data["ar"].index, n_perm=min(args.perm, 2000))
    tagged = data["tagged"]
    counts = {
        "headlines": int(len(heads)), "headlines_with_entry_day": int(len(tagged)),
        "repeats": int((~tagged["novel"]).sum()), "multi_ticker": int(tagged["multi_ticker"].sum()),
        "headline_types": tagged["event_type"].value_counts().to_dict(),
        "events": int(len(ev)), "events_with_novel_score": int(ev["score"].notna().sum()),
        "negative_events": int((ev["score"] < 0).sum()),
        "near_earnings": int(ev["near_earnings"].sum()),
        "events_with_surprise": int(ev["surprise"].notna().sum()),
        "terciles": ev["tercile"].value_counts().to_dict(),
        "no_news_stock_days": int(len(base)),
        "surprise_reference": data["ref"], "permutations": args.perm,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    conf.to_csv(args.out / "v3_confirmatory.csv", index=False)
    expl.to_csv(args.out / "v3_exploratory.csv", index=False)
    rob.to_csv(args.out / "v3_robustness.csv", index=False)
    (args.out / "v3_counts.json").write_text(json.dumps(counts, indent=2, default=float))

    cols = ["id", "n", "estimate", "ci_lo", "ci_hi", "p_cluster", "p_perm", "p_adj", "note"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70,
                           "display.float_format", "{:.4f}".format):
        print(json.dumps(counts, indent=2, default=float))
        print(conf[cols].to_string(index=False))
        print(expl[cols].to_string(index=False))
        print(rob[["variant"] + cols].to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m finpulse",
                                     description="FinPulse: Nifty 50 news sentiment.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="fetch, score and store headlines")
    p.add_argument("tickers", nargs="*", help="defaults to all Nifty 50")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--pause", type=float, default=1.0, help="seconds between tickers")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("backtest", help="sentiment vs forward returns")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("evaluate", help="FinBERT vs baselines on Financial PhraseBank")
    p.add_argument("--subsets", nargs="+", default=["allagree", "50agree"])
    p.add_argument("--models", nargs="+", default=["majority", "vader", "finbert"],
                   choices=["majority", "vader", "finbert", "zeroshot"])
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.add_argument("--labelled", type=Path, default=None,
                   help="CSV of record_id,label[,title] to evaluate instead of PhraseBank")
    p.add_argument("--alternatives", type=Path, default=None,
                   help="with --labelled: CSV of borderline items (sheet,id,title,label_used,"
                        "plausible_alternative) for a label-sensitivity rerun")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("tagger-eval", help="rule-based event-type tagger vs labels")
    p.add_argument("--labelled", type=Path, required=True,
                   help="CSV with ticker,title,multi_ticker and a label column")
    p.add_argument("--label-col", default="event_type_hand")
    p.add_argument("--alternatives", type=Path, default=None,
                   help="CSV of borderline items for a label-sensitivity rerun")
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.set_defaults(func=cmd_tagger_eval)

    p = sub.add_parser("eventstudy", help="pre-registered v3 event study")
    p.add_argument("--prices", type=Path, required=True, help="CSV: date + one column per ticker + ^NSEI")
    p.add_argument("--earnings", type=Path, required=True, help="CSV from yfinance get_earnings_dates")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    p.add_argument("--perm", type=int, default=10_000, help="permutations per confirmatory test")
    p.set_defaults(func=cmd_eventstudy)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "collect":
        try:
            args.tickers = [validate_ticker(t) for t in args.tickers] or list(NIFTY50)
        except ValueError as exc:
            parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
