import json

import numpy as np
import pandas as pd
import pytest

from finpulse import replication as rp
from tests.test_v3tests import synthetic


def records(rows):
    return pd.DataFrame(rows, columns=["record_id", "date", "domain", "url", "title"])


def test_to_headlines_maps_dedups_and_explodes():
    rec = records([
        ["20260105040000-2", "20260105040000", "livemint.com", "u1", "Infosys, Wipro shares fall"],
        ["20260105043000-1", "20260105043000", "livemint.com", "u1", "Infosys, Wipro shares fall"],
        ["20260105050000-1", "20260105050000", "ndtv.com", "u2", "Rupee ends flat"],
        ["20260105051500-1", "20260105051500", "ndtv.com", "u3", "ITC hotels demerger"],
        ["20260105053000-1", "20260105053000", "ndtv.com", "u4", "TCS wins deal"],
    ])
    h = rp.to_headlines(rec, ["INFY.NS", "WIPRO.NS", "TCS.NS"])   # ITC is outside the universe
    assert sorted(h["ticker"]) == ["INFY.NS", "TCS.NS", "WIPRO.NS"]
    first = h[h["ticker"] == "INFY.NS"].iloc[0]
    assert first["record_id"] == "20260105040000-2"                # first appearance of the URL
    assert first["published_at"] == "2026-01-05T04:00:00+00:00" == first["fetched_at"]
    assert h.set_index("ticker").loc["INFY.NS", "n_named"] == 2
    assert h.set_index("ticker").loc["TCS.NS", "n_named"] == 1


def test_attach_scores_requires_every_title():
    h = pd.DataFrame({"title": ["a", "b"]})
    out = rp.attach_scores(h, pd.DataFrame({"title": ["a", "b", "a"], "score": [0.1, -0.2, 0.9]}))
    assert out["score"].tolist() == [0.1, -0.2]
    with pytest.raises(ValueError):
        rp.attach_scores(h, pd.DataFrame({"title": ["a"], "score": [0.1]}))


def test_score_titles_scores_each_distinct_title_once():
    seen = []

    def scorer(texts):
        seen.extend(texts)
        return [{"score": len(t) / 10} for t in texts]

    out = rp.score_titles(["bb", "a", "bb"], scorer)
    assert seen == ["a", "bb"] and out["score"].tolist() == [0.1, 0.2]


def test_prepare_prices_drops_rows_after_last_close():
    p = pd.DataFrame({"A": [1, 2, 3]}, index=pd.to_datetime(["2026-11-20", "2026-11-23", "2026-11-25"]))
    assert rp.prepare_prices(p).index[-1] == pd.Timestamp("2026-11-23")


def test_check_reference():
    rp.check_reference({"winsor": (-42.4125, 43.54499999999987),
                        "cuts": (-3.6166666666666654, 4.136666666666674), "n_releases": 228})
    with pytest.raises(ValueError):
        rp.check_reference({"winsor": (-42.4125, 43.545), "cuts": (-3.0, 4.1366667), "n_releases": 228})
    with pytest.raises(ValueError):
        rp.check_reference({**rp.REFERENCE, "n_releases": 227})


@pytest.mark.parametrize("est, p2, expected", [
    (0.02, 0.06, "replicated"),           # one-sided 0.03
    (0.02, 0.12, "not replicated"),       # one-sided 0.06
    (0.01, 0.001, "not replicated"),      # smaller than half the original
    (-0.03, 0.001, "not replicated"),     # wrong sign
    (np.nan, np.nan, "underpowered, not tested"),
])
def test_verdict(est, p2, expected):
    v = rp.verdict({"estimate": est, "p_cluster": p2})
    assert v["verdict"] == expected
    if np.isfinite(est):
        assert v["p_one_sided"] == pytest.approx(p2 / 2 if est > 0 else 1 - p2 / 2)


def planted(effect, seed=3):
    ev, _ = synthetic(n=400, seed=seed)
    ev.loc[(ev["score"] < 0) & (ev["tercile"] == "top"), "car_0_1"] += effect
    return ev


def test_run_h3_detects_planted_effect_and_matches_v3_code():
    from finpulse.v3tests import confirmatory
    ev = planted(0.03)
    r = rp.run_h3(ev, n_perm=300)
    assert r["verdict"] == "replicated" and r["estimate"] == pytest.approx(0.03, abs=0.01)
    _, base = synthetic(n=400, seed=3)
    h3 = confirmatory(ev, base, n_perm=10).set_index("id").loc["H3"]
    assert r["estimate"] == pytest.approx(h3["estimate"]) and r["p_cluster"] == pytest.approx(h3["p_cluster"])
    assert rp.run_h3(planted(0.0), n_perm=100)["verdict"] == "not replicated"


def test_run_h3_underpowered():
    ev, _ = synthetic(n=40)
    r = rp.run_h3(ev, n_perm=10)
    assert r["verdict"] == "underpowered, not tested" and "n too small" in r["note"]


def test_counts_and_sufficiency():
    ev = planted(0.0)
    tagged = pd.DataFrame({"novel": [True, False], "multi_ticker": [False, True]})
    heads = pd.DataFrame({"url": ["a", "a", "b"]})
    c = rp.counts({"events": ev, "tagged": tagged, "ref": rp.REFERENCE}, heads, {"files_read": 9})
    kept = ev[ev["score"] < 0]
    assert c["files_read"] == 9 and c["distinct_headlines"] == 2 and c["repeats"] == 1
    assert c["negative_events"] == len(kept)
    assert c["top_tercile"] == int((kept["tercile"] == "top").sum())
    assert c["sufficient"] is (c["top_tercile"] >= 15 and c["bottom_tercile"] >= 15)
    json.dumps(c, default=float)


def test_exploratory_rows():
    ev = planted(0.03)
    orig = ev.iloc[:50][["ticker", "date"]]
    out = rp.exploratory(ev, ev.iloc[:200], orig, n_perm=50)
    assert out["id"].tolist() == ["E1", "E2", "E3"]
    assert (out["family"] == "exploratory").all()
    assert out.loc[0, "n"] < rp.run_h3(ev, n_perm=10)["n"]       # E1 leaves events out


def test_cli_refuses_cache_inside_repo(capsys):
    from finpulse import cli
    assert cli.main(["replicate-h3", "--cache", "tmpcache", "--prices", "p.csv",
                     "--earnings", "e.csv", "--scores", "s.csv"]) == 2
    assert cli.main(["gdelt-pull", "--cache", "tmpcache"]) == 2
