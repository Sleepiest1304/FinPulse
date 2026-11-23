"""Compare daily average headline sentiment with forward stock returns.

Alignment (no look-ahead):
  * Every headline is mapped to the first NSE close at or after the time it
    became known: published at or before 15:30 IST counts for that day's
    close, later headlines count for the next trading day's close.
    Weekends and holidays roll forward to the next date in the price index.
    If the source gave no publish time we use ``fetched_at`` (always later).
  * For that entry day D, the h-day forward return is close[D+h]/close[D] - 1.
    So the price move on the headline's own day is NOT credited to the signal.

Statistics per ticker and pooled:
  * n             number of (ticker, day) observations with a forward return
  * pearson / spearman correlation of daily mean score with forward return
  * t_stat        r * sqrt((n-2)/(1-r^2)) for the Pearson r (rough guide only;
                  pooled observations are not independent)
  * hit_rate      share of days with non-zero score and non-zero return where
                  sign(score) == sign(return)
  * up_rate       share of those same days with a positive return, i.e. the
                  hit rate of "always bullish" for comparison
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
MARKET_CLOSE = pd.Timedelta(hours=15, minutes=30)
HORIZONS = (1, 5)
MIN_N_FOR_CORR = 3


def assign_entry_dates(headlines: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map each headline to the trading day whose close first reflects it."""
    when = headlines["published_at"].where(headlines["published_at"].notna(),
                                           headlines["fetched_at"])
    ts = pd.to_datetime(when, utc=True, errors="coerce", format="ISO8601").dt.tz_convert(IST)
    day = ts.dt.normalize()
    after_close = (ts - day) > MARKET_CLOSE
    effective = (day + pd.to_timedelta(after_close.astype(int), unit="D")).dt.tz_localize(None)

    days = pd.DatetimeIndex(trading_days).sort_values()
    pos = days.searchsorted(effective.fillna(pd.Timestamp.max).values, side="left")
    out = pd.Series(pd.NaT, index=headlines.index, dtype="datetime64[ns]")
    ok = (pos < len(days)) & effective.notna().values
    out[ok] = days[pos[ok]]
    return out


def forward_returns(prices: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Long table: date, ticker, ret_<h>d for each horizon."""
    frames = []
    for h in horizons:
        r = prices.shift(-h) / prices - 1
        frames.append(r.stack(future_stack=True).rename(f"ret_{h}d"))
    out = pd.concat(frames, axis=1)
    out.index.names = ["date", "ticker"]
    return out.reset_index()


def daily_sentiment(headlines: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    df = headlines.copy()
    df["date"] = assign_entry_dates(df, trading_days)
    df = df.dropna(subset=["date"])
    return (df.groupby(["ticker", "date"])
              .agg(score=("score", "mean"), headlines=("score", "size"))
              .reset_index())


def build_panel(headlines: pd.DataFrame, prices: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Daily sentiment joined with forward returns, one row per (ticker, day)."""
    if headlines.empty or prices.empty:
        cols = ["ticker", "date", "score", "headlines"] + [f"ret_{h}d" for h in horizons]
        return pd.DataFrame(columns=cols)
    sent = daily_sentiment(headlines, prices.index)
    rets = forward_returns(prices, horizons)
    return sent.merge(rets, on=["ticker", "date"], how="left")


def _stats(score: pd.Series, ret: pd.Series) -> dict:
    mask = score.notna() & ret.notna()
    s, r = score[mask].astype(float), ret[mask].astype(float)
    n = int(mask.sum())
    res = {"n": n, "pearson": np.nan, "spearman": np.nan, "t_stat": np.nan,
           "hit_rate": np.nan, "up_rate": np.nan, "n_hit": 0}
    if n >= MIN_N_FOR_CORR and s.nunique() > 1 and r.nunique() > 1:
        rho = float(s.corr(r))
        res["pearson"] = rho
        res["spearman"] = float(s.rank().corr(r.rank()))
        if abs(rho) < 1:
            res["t_stat"] = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    directional = (np.sign(s) != 0) & (np.sign(r) != 0)
    res["n_hit"] = int(directional.sum())
    if res["n_hit"]:
        res["hit_rate"] = float((np.sign(s[directional]) == np.sign(r[directional])).mean())
        res["up_rate"] = float((r[directional] > 0).mean())
    return res


def summarize(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Per-ticker and pooled ("ALL") statistics for each horizon."""
    rows = []
    groups = [("ALL", panel)] + list(panel.groupby("ticker")) if not panel.empty else []
    for ticker, g in groups:
        for h in horizons:
            rows.append({"ticker": ticker, "horizon": f"{h}d",
                         **_stats(g["score"], g[f"ret_{h}d"])})
    cols = ["ticker", "horizon", "n", "pearson", "spearman", "t_stat",
            "hit_rate", "up_rate", "n_hit"]
    return pd.DataFrame(rows, columns=cols)
