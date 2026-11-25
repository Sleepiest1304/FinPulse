"""Event-study data preparation for v3 (see docs/v3-preregistration.md).

Pipeline: headlines -> novelty + event type -> entry day D -> one event per
(ticker, D) -> market-adjusted CARs -> point-in-time earnings surprise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import assign_entry_dates
from .events import event_type, tag_headline
from .novelty import mark_novelty

INDEX = "^NSEI"
IST = "Asia/Kolkata"
WINDOWS = {"car_0_1": (0, 1), "car_1_1": (1, 1), "car_1_5": (1, 5), "car_2_5": (2, 5)}
MAX_SURPRISE_AGE_DAYS = 120
EARNINGS_WINDOW = (-1, 5)
BASE_GAP = 5
WINSOR = (5, 95)
TERCILE_START, TERCILE_END = "2025-09-01", "2026-11-25"


# ---------------------------------------------------------------- returns
def abnormal_returns(prices: pd.DataFrame, index: str = INDEX) -> pd.DataFrame:
    """AR_t = r_stock,t - r_index,t on simple daily returns (dates x tickers)."""
    if index not in prices.columns:
        raise ValueError(f"price table has no {index} column")
    r = prices.sort_index().pct_change(fill_method=None)
    return r.drop(columns=[index]).sub(r[index], axis=0)


def car(ar: pd.DataFrame, a: int, b: int) -> pd.DataFrame:
    """CAR[a,b] indexed by entry day D: sum of AR over D+a ... D+b.

    NaN when any day in the window is missing (e.g. past the end of data).
    """
    total = None
    for k in range(a, b + 1):
        s = ar.shift(-k)
        total = s if total is None else total + s
    return total


# ---------------------------------------------------------------- events
def tag_headlines(headlines: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-headline frame with novel, multi_ticker, event_type and entry date."""
    df = mark_novelty(headlines)
    df["event_type"] = [tag_headline(t, m) for t, m in zip(df["title"], df["multi_ticker"])]
    df["date"] = assign_entry_dates(df, trading_days)
    return df.dropna(subset=["date"])


def aggregate_events(tagged: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, D).

    ``score`` uses novel headlines only (NaN for repeat-only pairs);
    ``score_all`` uses every headline. ``event_type`` comes from the novel
    headlines when there are any.
    """
    rows = []
    for (ticker, date), g in tagged.groupby(["ticker", "date"]):
        novel = g[g["novel"]]
        src = novel if len(novel) else g
        rows.append({
            "ticker": ticker, "date": date,
            "score": novel["score"].mean() if len(novel) else np.nan,
            "score_all": g["score"].mean(),
            "n_novel": int(len(novel)), "n_repeat": int(len(g) - len(novel)),
            "event_type": event_type(src["event_type"]),
        })
    cols = ["ticker", "date", "score", "score_all", "n_novel", "n_repeat", "event_type"]
    return pd.DataFrame(rows, columns=cols)


def attach_cars(events: pd.DataFrame, ar: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    out = events.copy()
    for name, (a, b) in windows.items():
        long = car(ar, a, b).stack(future_stack=True).rename(name)
        long.index.names = ["date", "ticker"]
        out = out.merge(long.reset_index(), on=["date", "ticker"], how="left")
    return out


def base_days(ar: pd.DataFrame, entry_dates: pd.DataFrame, gap: int = BASE_GAP) -> pd.DataFrame:
    """Stock-days with no headline entry within +-gap trading days."""
    days = ar.index
    blocked = pd.DataFrame(False, index=days, columns=ar.columns)
    pos = days.get_indexer(pd.DatetimeIndex(entry_dates["date"]))
    for p, t in zip(pos, entry_dates["ticker"]):
        if p >= 0 and t in blocked.columns:
            blocked.iloc[max(p - gap, 0): p + gap + 1, blocked.columns.get_loc(t)] = True
    free = (~blocked).stack(future_stack=True)
    free = free[free]
    out = free.index.to_frame(index=False)
    out.columns = ["date", "ticker"]
    return out


# ---------------------------------------------------------------- earnings
def prepare_earnings(raw: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """ticker, edate (IST calendar date), available (first trading day after), surprise."""
    ts = pd.to_datetime(raw["earnings_date"], utc=True, errors="coerce")
    edate = ts.dt.tz_convert(IST).dt.tz_localize(None).dt.normalize()
    days = pd.DatetimeIndex(trading_days).sort_values()
    pos = days.searchsorted(edate.fillna(pd.Timestamp.max).values, side="right")
    available = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    ok = (pos < len(days)) & edate.notna().values
    available[ok] = days[pos[ok]]
    out = pd.DataFrame({
        "ticker": raw["ticker"].values, "edate": edate.values,
        "available": available.values,
        "surprise": pd.to_numeric(raw["Surprise(%)"], errors="coerce").values,
    })
    return out.dropna(subset=["edate"]).sort_values(["ticker", "edate"]).reset_index(drop=True)


def winsor_limits(values, pct=WINSOR) -> tuple[float, float]:
    v = pd.Series(values, dtype=float).dropna()
    return float(np.percentile(v, pct[0])), float(np.percentile(v, pct[1]))


def surprise_reference(earnings: pd.DataFrame, start=TERCILE_START, end=TERCILE_END):
    """Winsor limits and tercile cut points from the pre-registered sample."""
    e = earnings[(earnings["edate"] >= start) & (earnings["edate"] <= end)]
    s = e["surprise"].dropna()
    lo, hi = winsor_limits(s)
    w = s.clip(lo, hi)
    return {"winsor": (lo, hi), "cuts": tuple(np.percentile(w, [100 / 3, 200 / 3])),
            "n_releases": int(len(s))}


def attach_surprise(events: pd.DataFrame, earnings: pd.DataFrame, ref: dict,
                    max_age: int = MAX_SURPRISE_AGE_DAYS) -> pd.DataFrame:
    """Point-in-time surprise: latest release available strictly before D,
    with a reported surprise, at most ``max_age`` calendar days old."""
    out = events.copy()
    lo, hi = ref["winsor"]
    c1, c2 = ref["cuts"]
    known = earnings.dropna(subset=["surprise", "available"])
    by_ticker = {t: g for t, g in known.groupby("ticker")}
    surprise, age = [], []
    for t, d in zip(out["ticker"], out["date"]):
        g = by_ticker.get(t)
        prior = g[g["available"] < d] if g is not None else None
        if prior is None or prior.empty:
            surprise.append(np.nan); age.append(np.nan); continue
        last = prior.iloc[-1]
        days = (d - last["edate"]).days
        if days > max_age:
            surprise.append(np.nan); age.append(days); continue
        surprise.append(last["surprise"]); age.append(days)
    out["surprise"] = surprise
    out["surprise_age"] = age
    out["surprise_w"] = out["surprise"].clip(lo, hi)
    w = out["surprise_w"]
    out["tercile"] = np.select([w.isna(), w < c1, w >= c2], [None, "bottom", "top"], "middle")
    out["tercile"] = out["tercile"].where(w.notna(), None)
    return out


def flag_earnings_window(events: pd.DataFrame, earnings: pd.DataFrame,
                         trading_days: pd.DatetimeIndex, window=EARNINGS_WINDOW) -> pd.DataFrame:
    """``near_earnings``: an earnings date falls between trading day D-1 and D+5."""
    out = events.copy()
    days = pd.DatetimeIndex(trading_days).sort_values()
    by_ticker = {t: g["edate"].values for t, g in earnings.groupby("ticker")}
    flags = []
    for t, d in zip(out["ticker"], out["date"]):
        p = days.get_loc(d)
        lo = days[max(p + window[0], 0)]
        hi = days[min(p + window[1], len(days) - 1)]
        ed = by_ticker.get(t, np.array([], dtype="datetime64[ns]"))
        flags.append(bool(((ed >= np.datetime64(lo)) & (ed <= np.datetime64(hi))).any()))
    out["near_earnings"] = flags
    return out


def build(headlines: pd.DataFrame, prices: pd.DataFrame, earnings_raw: pd.DataFrame):
    """Everything the tests need: events, base stock-days, AR table, reference."""
    prices = prices.sort_index()
    ar = abnormal_returns(prices)
    days = ar.index
    tagged = tag_headlines(headlines, days)
    events = attach_cars(aggregate_events(tagged), ar)
    earnings = prepare_earnings(earnings_raw, days)
    ref = surprise_reference(earnings)
    events = flag_earnings_window(attach_surprise(events, earnings, ref), earnings, days)
    base = attach_cars(base_days(ar, tagged[["ticker", "date"]]), ar)
    return {"tagged": tagged, "events": events, "base": base, "ar": ar, "ref": ref}
