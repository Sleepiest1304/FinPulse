"""H3 replication on GDELT Indian-outlet headlines.

Implements the 2026-11-25 amendment in docs/v3-preregistration.md. The
sample is built from cached GDELT records (``finpulse.gdelt.pull``) and kept
separate from the original 398 headlines. ``counts`` is reported first;
``run_h3`` is the single registered test, and ``exploratory`` holds E1-E3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eventstudy import build
from .gdelt import match_tickers
from .v3tests import H3_QUESTION, MIN_GROUP, N_PERM, NEG_THRESHOLD, SEED, tercile_difference

ORIGINAL_H3 = 0.02270399745025572          # results/v3_confirmatory.csv
MIN_EFFECT = ORIGINAL_H3 / 2
ALPHA_ONE_SIDED = 0.05
REFERENCE = {"winsor": (-42.4125, 43.545), "cuts": (-3.6166667, 4.1366667), "n_releases": 228}
LAST_CLOSE = "2026-11-23"                  # the 2026-11-25 row is intraday (see amendment)


# ---------------------------------------------------------------- sample
def to_headlines(records: pd.DataFrame, universe) -> pd.DataFrame:
    """One row per (ticker, headline): first appearance of each URL, named tickers only."""
    rec = records.copy()
    rec["_ts"] = pd.to_datetime(rec["date"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    rec = rec.dropna(subset=["_ts"]).sort_values(["_ts", "record_id"])
    rec = rec.drop_duplicates("url", keep="first")
    rec["tickers"] = rec["title"].map(match_tickers)
    rec["n_named"] = rec["tickers"].map(len)
    rec = rec[rec["n_named"] > 0].explode("tickers").rename(columns={"tickers": "ticker"})
    rec = rec[rec["ticker"].isin(set(universe))]
    iso = rec["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    out = pd.DataFrame({"ticker": rec["ticker"].values, "title": rec["title"].values,
                        "published_at": iso.values, "fetched_at": iso.values,
                        "url": rec["url"].values, "record_id": rec["record_id"].values,
                        "n_named": rec["n_named"].values})
    return out.reset_index(drop=True)


def attach_scores(headlines: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Join FinBERT scores (columns title, score); every title must have one."""
    out = headlines.merge(scores[["title", "score"]].drop_duplicates("title"), on="title", how="left")
    if out["score"].isna().any():
        raise ValueError(f"{int(out['score'].isna().sum())} headlines have no FinBERT score")
    return out


def score_titles(titles, scorer=None) -> pd.DataFrame:
    """FinBERT score = P(positive) - P(negative) for each distinct title."""
    if scorer is None:
        from .sentiment import get_sentiment as scorer
    uniq = sorted(set(titles))
    res = scorer(uniq)
    return pd.DataFrame({"title": uniq, "score": [r["score"] for r in res]})


def prepare_prices(prices: pd.DataFrame, last_close: str = LAST_CLOSE) -> pd.DataFrame:
    return prices.sort_index().loc[:last_close]


def check_reference(ref: dict, expected: dict = REFERENCE, tol: float = 1e-4) -> None:
    """The tercile reference must be the original one (amendment: identical cut-offs)."""
    got = list(ref["winsor"]) + list(ref["cuts"])
    want = list(expected["winsor"]) + list(expected["cuts"])
    if ref["n_releases"] != expected["n_releases"] or not np.allclose(got, want, atol=tol):
        raise ValueError(f"surprise reference {ref} differs from the registered {expected}")


def primary(events: pd.DataFrame, exclude_earnings: bool = True) -> pd.DataFrame:
    """Events entering H3: novel score, earnings exclusion, ``neg`` flag."""
    ev = events.dropna(subset=["score"]).copy()
    if exclude_earnings:
        ev = ev[~ev["near_earnings"]]
    ev["neg"] = ev["score"] < NEG_THRESHOLD
    return ev


def counts(data: dict, headlines: pd.DataFrame, pull: dict | None = None) -> dict:
    """The counts the amendment requires before the test."""
    ev, tagged = data["events"], data["tagged"]
    scored = ev.dropna(subset=["score"])
    neg = scored[scored["score"] < NEG_THRESHOLD]
    kept = primary(ev)
    groups = kept[kept["neg"] & kept["tercile"].isin(["top", "bottom"])].dropna(subset=["car_0_1"])
    top, bottom = int((groups["tercile"] == "top").sum()), int((groups["tercile"] == "bottom").sum())
    out = dict(pull or {})
    out.update({
        "ticker_headline_rows": int(len(headlines)),
        "distinct_headlines": int(headlines["url"].nunique()),
        "rows_with_entry_day": int(len(tagged)),
        "repeats": int((~tagged["novel"]).sum()),
        "multi_ticker_rows": int(tagged["multi_ticker"].sum()),
        "stock_day_events": int(len(ev)),
        "events_with_novel_score": int(len(scored)),
        "negative_events": int(len(neg)),
        "negative_after_earnings_exclusion": int(kept["neg"].sum()),
        "negative_with_car_0_1_and_tercile": int(len(groups)),
        "top_tercile": top, "bottom_tercile": bottom,
        "tickers": int(ev["ticker"].nunique()),
        "first_event": str(ev["date"].min().date()) if len(ev) else None,
        "last_event": str(ev["date"].max().date()) if len(ev) else None,
        "surprise_reference": data["ref"],
        "sufficient": top >= MIN_GROUP and bottom >= MIN_GROUP,
    })
    return out


# ---------------------------------------------------------------- test
def verdict(row: dict) -> dict:
    """Apply the registered replication criterion to an H3 result row."""
    est, p2 = row.get("estimate"), row.get("p_cluster")
    if est is None or not np.isfinite(est):
        return {"p_one_sided": np.nan, "verdict": "underpowered, not tested",
                "same_sign": None, "p_below": None, "at_least_half": None}
    p1 = p2 / 2 if est > 0 else 1 - p2 / 2
    checks = {"same_sign": bool(est > 0), "p_below": bool(p1 < ALPHA_ONE_SIDED),
              "at_least_half": bool(est >= MIN_EFFECT)}
    return {"p_one_sided": float(p1), **checks,
            "verdict": "replicated" if all(checks.values()) else "not replicated"}


def run_h3(events: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """The single registered test."""
    row = tercile_difference(primary(events), "car_0_1", "H3-rep", H3_QUESTION,
                             np.random.default_rng(seed), n_perm)
    return {**row, **verdict(row), "original_estimate": ORIGINAL_H3, "min_effect": MIN_EFFECT}


def exploratory(events, events_single, original_keys, n_perm: int = N_PERM, seed: int = SEED):
    """E1-E3 from the amendment. Not part of the verdict."""
    keys = set(map(tuple, original_keys[["ticker", "date"]].astype(str).values))
    in_orig = [(t, str(d)) in keys for t, d in zip(events["ticker"], events["date"].astype(str))]
    rows = []
    for eid, ev, excl, note in (
            ("E1", events[~np.array(in_orig, dtype=bool)], True,
             "stock-days that are also original events left out"),
            ("E2", events_single, True, "only headlines naming exactly one Nifty 50 company"),
            ("E3", events, False, "earnings-window events kept")):
        r = tercile_difference(primary(ev, exclude_earnings=excl), "car_0_1", eid, H3_QUESTION,
                               np.random.default_rng(seed), n_perm, family="exploratory")
        r["note"] = f"{note}; {r['note']}"
        rows.append(r)
    return pd.DataFrame(rows)


def build_sample(headlines, prices, earnings):
    data = build(headlines, prepare_prices(prices), earnings)
    check_reference(data["ref"])
    return data
