import numpy as np
import pandas as pd
import pytest

from finpulse.v3tests import confirmatory, exploratory, first_events, robustness

DAYS = pd.bdate_range("2026-01-05", periods=200)


def synthetic(n=240, reversion=0.0, seed=0):
    """Events across 24 tickers and many dates, with an optional planted effect."""
    rng = np.random.default_rng(seed)
    ev = pd.DataFrame({
        "ticker": [f"T{i % 24:02d}.NS" for i in range(n)],
        "date": DAYS[rng.integers(0, 190, n)],
        "score": rng.uniform(-1, 1, n),
        "n_novel": 1, "n_repeat": 0,
        "event_type": rng.choice(["earnings", "macro", "other"], n),
        "near_earnings": False,
        "surprise_w": rng.normal(0, 5, n),
    })
    ev["score_all"] = ev["score"]
    ev["tercile"] = np.where(ev["surprise_w"] < -2, "bottom",
                             np.where(ev["surprise_w"] >= 2, "top", "middle"))
    for c in ("car_0_1", "car_1_1", "car_1_5", "car_2_5"):
        ev[c] = rng.normal(0, 0.02, n)
    ev.loc[ev["score"] < 0, "car_2_5"] += reversion
    base = pd.DataFrame({
        "ticker": [f"T{i % 24:02d}.NS" for i in range(600)],
        "date": DAYS[rng.integers(0, 190, 600)],
        "car_0_1": rng.normal(0, 0.02, 600), "car_2_5": rng.normal(0, 0.02, 600),
    })
    return ev, base


def test_confirmatory_shape_and_holm():
    ev, base = synthetic()
    out = confirmatory(ev, base, n_perm=200)
    assert out["id"].tolist() == ["H1", "H2", "H3", "H4"]
    ok = out["p_cluster"].notna()
    assert (out.loc[ok, "p_adj"] >= out.loc[ok, "p_cluster"] - 1e-12).all()
    assert out["p_perm"].between(0, 1).all()


def test_planted_reversion_is_detected_and_null_is_not():
    ev, base = synthetic(reversion=0.03, seed=1)
    h1 = confirmatory(ev, base, n_perm=300).set_index("id").loc["H1"]
    assert h1["estimate"] == pytest.approx(0.03, abs=0.01)
    assert h1["p_cluster"] < 0.01 and h1["p_perm"] < 0.05
    ev0, base0 = synthetic(reversion=0.0, seed=1)
    assert confirmatory(ev0, base0, n_perm=300).set_index("id").loc["H1", "p_cluster"] > 0.01


def test_earnings_exclusion_and_small_groups():
    ev, base = synthetic(n=40)
    ev["near_earnings"] = True
    out = confirmatory(ev, base, n_perm=50)
    assert out["p_cluster"].isna().all()
    assert out["note"].str.contains("n too small").all()
    kept = confirmatory(ev, base, exclude_earnings=False, n_perm=50)
    assert kept.set_index("id").loc["H1", "n"] > 0


def test_permutation_is_reproducible():
    ev, base = synthetic()
    a = confirmatory(ev, base, n_perm=100, seed=7)
    b = confirmatory(ev, base, n_perm=100, seed=7)
    assert a["p_perm"].tolist() == b["p_perm"].tolist()


def test_exploratory_family_and_untested_rows():
    ev, base = synthetic()
    out = exploratory(ev)
    assert out["id"].tolist() == [f"S{i}" for i in range(1, 15)]
    # No rumour/regulatory/management events -> untested, no adjusted p.
    skipped = out[out["note"].str.startswith("n too small")]
    assert set(skipped["id"]) >= {"S2", "S3", "S4", "S7", "S8", "S9", "S13"}
    assert skipped["p_adj"].isna().all()
    assert "Pearson r" in out.set_index("id").loc["S11", "note"]


def test_first_events_drops_close_repeats():
    ev = pd.DataFrame({"ticker": ["A", "A", "A", "B"],
                       "date": [DAYS[0], DAYS[3], DAYS[6], DAYS[1]]})
    kept = first_events(ev, DAYS)
    assert sorted(kept["date"].tolist()) == [DAYS[0], DAYS[1], DAYS[6]]


def test_robustness_variants():
    ev, base = synthetic()
    out = robustness(ev, base, DAYS, n_perm=50)
    assert out["variant"].nunique() == 4
    assert len(out) == 16
