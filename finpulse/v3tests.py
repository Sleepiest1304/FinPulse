"""The pre-registered v3 tests (docs/v3-preregistration.md, sections 3 to 5).

Every test returns one row: id, family, question, n, estimate, 95% CI,
two-way clustered p, permutation p (confirmatory only), adjusted p, note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import TYPE_ORDER
from .stats import benjamini_hochberg, holm, ols_twoway

MIN_GROUP = 15
NEG_THRESHOLD = 0.0
ROBUST_THRESHOLD = -0.3
N_PERM = 10_000
SEED = 2026


def _fit(y, X, df):
    return ols_twoway(np.asarray(y, float), np.asarray(X, float),
                      df["date"].astype(str).values, df["ticker"].values)


def _row(tid, family, question, df, res, k, perm=np.nan, note=""):
    lo, hi = res.ci(k)
    return {"id": tid, "family": family, "question": question, "n": int(res.n),
            "clusters_date": res.n_clusters[0], "clusters_ticker": res.n_clusters[1],
            "estimate": float(res.coef[k]), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_cluster": float(res.p[k]), "p_perm": perm,
            "note": (note + ("; one-way SE fallback" if res.fallback else "")).strip("; ")}


def _skip(tid, family, question, n, note="n too small"):
    return {"id": tid, "family": family, "question": question, "n": int(n),
            "clusters_date": np.nan, "clusters_ticker": np.nan, "estimate": np.nan,
            "ci_lo": np.nan, "ci_hi": np.nan, "p_cluster": np.nan, "p_perm": np.nan,
            "note": note}


def _months(df):
    return pd.to_datetime(df["date"]).dt.to_period("M").astype(str).values


def _shuffle_within(labels: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    out = labels.copy()
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        out[idx] = labels[rng.permutation(idx)]
    return out


def _perm_p(obs: float, null: np.ndarray, centre: float = 0.0) -> float:
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan
    return float((np.sum(np.abs(null - centre) >= abs(obs - centre)) + 1) / (len(null) + 1))


def _diff_means_fast(y, g):
    a, b = y[g], y[~g]
    return a.mean() - b.mean() if len(a) and len(b) else np.nan


H3_QUESTION = "CAR[0,1] after negative events: top minus bottom surprise tercile"


def tercile_difference(ev, col, tid, question, rng, n_perm, family="confirmatory"):
    """Top minus bottom surprise tercile in ``col`` among ``ev`` rows with ``neg`` set.

    The H3/H4 test: clustered OLS on a top dummy plus the within-month
    permutation. Groups below MIN_GROUP are skipped, not tested.
    """
    t = ev[ev["neg"] & ev["tercile"].isin(["top", "bottom"])].dropna(subset=[col])
    top = (t["tercile"] == "top").values
    if top.sum() < MIN_GROUP or (~top).sum() < MIN_GROUP:
        return _skip(tid, family, question, len(t),
                     f"n too small (top {int(top.sum())}, bottom {int((~top).sum())})")
    res = _fit(t[col], np.column_stack([np.ones(len(t)), top.astype(float)]), t)
    y, m = t[col].values, _months(t)
    null = np.array([_diff_means_fast(y, _shuffle_within(top, m, rng)) for _ in range(n_perm)])
    return _row(tid, family, question, t, res, 1, _perm_p(float(res.coef[1]), null),
                f"top {int(top.sum())}, bottom {int((~top).sum())}")


# ---------------------------------------------------------------- confirmatory
def confirmatory(events, base, threshold=NEG_THRESHOLD, exclude_earnings=True,
                 n_perm=N_PERM, seed=SEED, car_clip=None):
    rng = np.random.default_rng(seed)
    ev = events.dropna(subset=["score"]).copy()
    if exclude_earnings:
        ev = ev[~ev["near_earnings"]]
    if car_clip is not None:
        for c in ("car_0_1", "car_2_5"):
            lo, hi = np.nanpercentile(ev[c], car_clip)
            ev[c] = ev[c].clip(lo, hi)
    ev["neg"] = ev["score"] < threshold if threshold == 0 else ev["score"] <= threshold
    rows = []

    # H1: mean CAR[2,5] of negative events.
    d = ev.dropna(subset=["car_2_5"])
    neg = d[d["neg"]]
    q = "Mean CAR[2,5] after negative events"
    if len(neg) < MIN_GROUP:
        rows.append(_skip("H1", "confirmatory", q, len(neg)))
    else:
        res = _fit(neg["car_2_5"], np.ones((len(neg), 1)), neg)
        y, s, m = d["car_2_5"].values, d["score"].values, _months(d)
        isneg = (s < threshold) if threshold == 0 else (s <= threshold)
        obs = y[isneg].mean()
        null = np.array([y[_shuffle_within(isneg, m, rng)].mean() for _ in range(n_perm)])
        rows.append(_row("H1", "confirmatory", q, neg, res, 0,
                         _perm_p(obs, null, centre=float(null.mean())),
                         "permutation null: negative labels shuffled within month"))

    # H2: reversion slope after negative events minus slope on no-news days.
    b = base.dropna(subset=["car_0_1", "car_2_5"]).assign(neg=False)
    nn = neg.dropna(subset=["car_0_1"]).assign(neg=True)
    q = "Slope of CAR[2,5] on CAR[0,1]: negative events minus no-news days"
    if len(nn) < MIN_GROUP:
        rows.append(_skip("H2", "confirmatory", q, len(nn)))
    else:
        pool = pd.concat([nn[["date", "ticker", "car_0_1", "car_2_5", "neg"]],
                          b[["date", "ticker", "car_0_1", "car_2_5", "neg"]]], ignore_index=True)
        g = pool["neg"].values.astype(float)
        x = pool["car_0_1"].values
        X = np.column_stack([np.ones(len(pool)), g, x, g * x])
        res = _fit(pool["car_2_5"], X, pool)
        yv, m = pool["car_2_5"].values, _months(pool)
        null = []
        for _ in range(n_perm):
            gp = _shuffle_within(g, m, rng)
            Xp = np.column_stack([np.ones(len(pool)), gp, x, gp * x])
            null.append(np.linalg.lstsq(Xp, yv, rcond=None)[0][3])
        rows.append(_row("H2", "confirmatory", q, pool, res, 3,
                         _perm_p(float(res.coef[3]), np.array(null)),
                         f"{len(nn)} negative events vs {len(b)} no-news stock-days"))

    # H3 / H4: top minus bottom surprise tercile among negative events.
    for tid, col, q in (("H3", "car_0_1", H3_QUESTION),
                        ("H4", "car_2_5", "CAR[2,5] after negative events: top minus bottom surprise tercile")):
        rows.append(tercile_difference(ev, col, tid, q, rng, n_perm))

    out = pd.DataFrame(rows)
    out["p_adj"] = holm(out["p_cluster"].values)
    return out


# ---------------------------------------------------------------- exploratory
def exploratory(events):
    rows = []
    ev = events.dropna(subset=["score"])
    neg = ev[ev["score"] < 0]
    types = [t for t in TYPE_ORDER if t != "other"]
    for i, (col, label) in enumerate((("car_0_1", "CAR[0,1]"), ("car_2_5", "CAR[2,5]"))):
        for j, kind in enumerate(types):
            tid = f"S{1 + 5 * i + j}"
            q = f"Mean {label} after negative {kind} events"
            g = neg[neg["event_type"] == kind].dropna(subset=[col])
            if len(g) < MIN_GROUP:
                rows.append(_skip(tid, "exploratory", q, len(g)))
                continue
            rows.append(_row(tid, "exploratory", q, g,
                             _fit(g[col], np.ones((len(g), 1)), g), 0))

    for tid, col, label in (("S11", "car_1_5", "CAR[1,5]"), ("S12", "car_1_1", "CAR[1,1]")):
        q = f"Slope of {label} on score, novel events"
        g = ev.dropna(subset=[col])
        if len(g) < MIN_GROUP:
            rows.append(_skip(tid, "exploratory", q, len(g)))
            continue
        res = _fit(g[col], np.column_stack([np.ones(len(g)), g["score"]]), g)
        rows.append(_row(tid, "exploratory", q, g, res, 1,
                         note=f"Pearson r = {g['score'].corr(g[col]):+.3f}"))

    q = "CAR[0,1] after negative events: novel minus repeat-only"
    allneg = events[events["score_all"] < 0].dropna(subset=["car_0_1"])
    novel = (allneg["n_novel"] > 0).values
    if novel.sum() < MIN_GROUP or (~novel).sum() < MIN_GROUP:
        rows.append(_skip("S13", "exploratory", q, len(allneg),
                          f"n too small (novel {int(novel.sum())}, repeat-only {int((~novel).sum())})"))
    else:
        res = _fit(allneg["car_0_1"], np.column_stack([np.ones(len(allneg)), novel.astype(float)]), allneg)
        rows.append(_row("S13", "exploratory", q, allneg, res, 1))

    q = "Score x surprise interaction in CAR[1,5]"
    g = ev.dropna(subset=["car_1_5", "surprise_w"])
    if len(g) < MIN_GROUP:
        rows.append(_skip("S14", "exploratory", q, len(g)))
    else:
        s, w = g["score"].values, g["surprise_w"].values
        res = _fit(g["car_1_5"], np.column_stack([np.ones(len(g)), s, w, s * w]), g)
        rows.append(_row("S14", "exploratory", q, g, res, 3,
                         note="surprise winsorised, in % points"))

    out = pd.DataFrame(rows)
    # Untested groups count in the family at p = 1, as pre-registered.
    p = out["p_cluster"].fillna(1.0).values
    out["p_adj"] = benjamini_hochberg(p)
    out.loc[out["p_cluster"].isna(), "p_adj"] = np.nan
    return out


def first_events(events, trading_days, gap=5):
    """Keep a ticker's event only if its previous kept event is > gap trading days earlier."""
    days = pd.DatetimeIndex(trading_days).sort_values()
    keep, last = [], {}
    for idx, r in events.sort_values("date").iterrows():
        p = days.get_loc(r["date"])
        prev = last.get(r["ticker"])
        if prev is None or p - prev > gap:
            keep.append(idx)
            last[r["ticker"]] = p
    return events.loc[keep]


def robustness(events, base, trading_days, n_perm=2000, seed=SEED):
    """Section 5 robustness variants of H1-H4 (reported, not tested).

    The 1/99 winsorising applies to event CARs; no-news days are unchanged.
    """
    keep = first_events(events, trading_days).index
    variants = {
        "CARs winsorised 1/99": dict(car_clip=(1, 99)),
        "negative = score <= -0.3": dict(threshold=ROBUST_THRESHOLD),
        "earnings-window events kept": dict(exclude_earnings=False),
        "first event per ticker within 5 trading days": dict(_events=events.loc[keep]),
    }
    frames = []
    for name, kw in variants.items():
        ev = kw.pop("_events", events)
        r = confirmatory(ev, base, n_perm=n_perm, seed=seed, **kw)
        r.insert(0, "variant", name)
        frames.append(r)
    return pd.concat(frames, ignore_index=True)
