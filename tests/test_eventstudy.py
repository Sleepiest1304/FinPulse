import numpy as np
import pandas as pd
import pytest

from finpulse.eventstudy import (abnormal_returns, aggregate_events, attach_cars,
                                 attach_surprise, base_days, build, car,
                                 flag_earnings_window, prepare_earnings,
                                 surprise_reference)

DAYS = pd.bdate_range("2026-01-05", periods=12)


def prices():
    idx = 100 * np.cumprod([1] + [1.01] * 11)        # index rises 1% a day
    stock = 50 * np.cumprod([1] + [1.03] * 11)       # stock rises 3% a day
    return pd.DataFrame({"AAA.NS": stock, "^NSEI": idx}, index=DAYS)


def test_abnormal_returns_subtract_index():
    ar = abnormal_returns(prices())
    assert list(ar.columns) == ["AAA.NS"]
    assert ar["AAA.NS"].iloc[1:].tolist() == pytest.approx([0.02] * 11)
    assert np.isnan(ar["AAA.NS"].iloc[0])


def test_abnormal_returns_needs_index():
    with pytest.raises(ValueError):
        abnormal_returns(prices().drop(columns="^NSEI"))


def test_car_window_and_end_of_data():
    ar = pd.DataFrame({"A": np.arange(1.0, 7.0)}, index=DAYS[:6])
    c = car(ar, 2, 3)
    assert c["A"].iloc[0] == 3 + 4             # AR on D+2 and D+3
    assert np.isnan(c["A"].iloc[-2])           # window runs past the data


def headlines(rows):
    return pd.DataFrame(rows, columns=["ticker", "title", "published_at", "fetched_at", "score"])


def test_aggregate_uses_novel_scores_only():
    from finpulse.eventstudy import tag_headlines
    h = headlines([
        ["AAA.NS", "AAA Q3 profit falls 10% on weak demand", "2026-01-06T04:00:00Z", None, -0.8],
        ["AAA.NS", "AAA Q3 profit falls 10% on weak demand - Reuters", "2026-01-06T05:00:00Z", None, 0.6],
        ["AAA.NS", "AAA names new CEO", "2026-01-06T06:00:00Z", None, 0.2],
    ])
    ev = aggregate_events(tag_headlines(h, DAYS))
    assert len(ev) == 1
    r = ev.iloc[0]
    assert r["n_novel"] == 2 and r["n_repeat"] == 1
    assert r["score"] == pytest.approx((-0.8 + 0.2) / 2)
    assert r["score_all"] == pytest.approx(0.0)
    assert r["event_type"] == "earnings"


def test_attach_cars_aligns_on_ticker_and_date():
    ar = abnormal_returns(prices())
    ev = pd.DataFrame({"ticker": ["AAA.NS"], "date": [DAYS[2]]})
    out = attach_cars(ev, ar)
    assert out["car_0_1"].iloc[0] == pytest.approx(0.04)
    assert out["car_2_5"].iloc[0] == pytest.approx(0.08)


def test_base_days_exclude_gap_around_headlines():
    ar = pd.DataFrame({"A": 0.0, "B": 0.0}, index=DAYS)
    free = base_days(ar, pd.DataFrame({"ticker": ["A"], "date": [DAYS[6]]}), gap=2)
    a_days = set(free[free["ticker"] == "A"]["date"])
    assert DAYS[4] not in a_days and DAYS[8] not in a_days
    assert DAYS[3] in a_days and DAYS[9] in a_days
    assert len(free[free["ticker"] == "B"]) == len(DAYS)


def earnings_raw(rows):
    return pd.DataFrame(rows, columns=["earnings_date", "EPS Estimate", "Reported EPS",
                                       "Surprise(%)", "ticker"])


def test_prepare_earnings_available_strictly_after_ist_date():
    # 05:00 US Eastern on Wed 7 Jan = 15:30 IST on 7 Jan -> available Thu 8 Jan.
    e = prepare_earnings(earnings_raw([["2026-01-07 05:00:00-05:00", 1, 1.1, 10.0, "AAA.NS"]]), DAYS)
    assert e["edate"].iloc[0] == pd.Timestamp("2026-01-07")
    assert e["available"].iloc[0] == pd.Timestamp("2026-01-08")


def test_prepare_earnings_late_utc_becomes_next_ist_day():
    # 20:00 UTC on 9 Jan (Fri) = 01:30 IST on Sat 10 Jan -> available Mon 12 Jan.
    e = prepare_earnings(earnings_raw([["2026-01-09 20:00:00+00:00", 1, 1, 0.0, "AAA.NS"]]), DAYS)
    assert e["edate"].iloc[0] == pd.Timestamp("2026-01-10")
    assert e["available"].iloc[0] == pd.Timestamp("2026-01-12")


def ref():
    return {"winsor": (-20.0, 20.0), "cuts": (-2.0, 2.0), "n_releases": 3}


def test_attach_surprise_is_point_in_time():
    e = prepare_earnings(earnings_raw([
        ["2026-01-05 05:00:00-05:00", 1, 1, -50.0, "AAA.NS"],   # available 6 Jan
        ["2026-01-07 05:00:00-05:00", 1, 1, 8.0, "AAA.NS"],     # available 8 Jan
    ]), DAYS)
    ev = pd.DataFrame({"ticker": ["AAA.NS"] * 3, "date": [DAYS[1], DAYS[2], DAYS[4]]})
    out = attach_surprise(ev, e, ref())
    # 6 Jan: the 5 Jan release is not yet *strictly before* the entry day.
    assert np.isnan(out["surprise"].iloc[0])
    # 7 Jan: only the 5 Jan release is known (winsorised to -20, bottom tercile).
    assert out["surprise"].iloc[1] == -50 and out["surprise_w"].iloc[1] == -20
    assert out["tercile"].iloc[1] == "bottom"
    # 9 Jan: the 7 Jan release (available 8 Jan) is now strictly earlier.
    assert out["surprise"].iloc[2] == 8 and out["tercile"].iloc[2] == "top"


def test_attach_surprise_ignores_stale_and_missing():
    e = prepare_earnings(earnings_raw([
        ["2025-06-02 05:00:00-04:00", 1, 1, 5.0, "AAA.NS"],     # > 120 days old
        ["2026-01-05 05:00:00-05:00", 1, None, None, "AAA.NS"],  # no reported surprise
    ]), pd.bdate_range("2025-06-01", "2026-01-31"))
    ev = pd.DataFrame({"ticker": ["AAA.NS", "BBB.NS"], "date": [DAYS[5], DAYS[5]]})
    out = attach_surprise(ev, e, ref())
    assert out["surprise"].isna().all()
    assert out["surprise_age"].iloc[0] > 120


def test_surprise_reference_terciles():
    e = pd.DataFrame({"edate": pd.to_datetime(["2026-01-01"] * 6),
                      "surprise": [-100, -3, -1, 1, 3, 400]})
    r = surprise_reference(e)
    assert r["n_releases"] == 6
    assert r["winsor"][0] > -100 and r["winsor"][1] < 400
    assert r["cuts"][0] < r["cuts"][1]


def test_flag_earnings_window():
    e = prepare_earnings(earnings_raw([["2026-01-12 05:00:00-05:00", 1, 1, 1.0, "AAA.NS"]]), DAYS)
    ev = pd.DataFrame({"ticker": ["AAA.NS"] * 3, "date": [DAYS[0], DAYS[2], DAYS[6]]})
    out = flag_earnings_window(ev, e, DAYS)
    # 12 Jan is D+5 for 5 Jan, inside [D+0, D+5] for 7 Jan and D-1 for 13 Jan.
    assert out["near_earnings"].tolist() == [True, True, True]
    ev2 = pd.DataFrame({"ticker": ["AAA.NS"], "date": [DAYS[8]]})   # 15 Jan: D-1 = 14 Jan
    assert not flag_earnings_window(ev2, e, DAYS)["near_earnings"].iloc[0]


def test_build_end_to_end():
    h = headlines([["AAA.NS", "AAA wins a large order", "2026-01-06T04:00:00Z", None, -0.5]])
    e = earnings_raw([["2026-01-02 05:00:00-05:00", 1, 1, 4.0, "AAA.NS"]])
    out = build(h, prices(), e)
    ev = out["events"]
    assert len(ev) == 1 and ev["car_2_5"].iloc[0] == pytest.approx(0.08)
    assert ev["surprise"].iloc[0] == 4.0
    assert not out["base"].empty
