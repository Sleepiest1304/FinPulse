import numpy as np
import pandas as pd
import pytest

from engine.backtest import assign_entry_dates, build_panel, forward_returns, summarize

# Mon 2 Nov 2026 .. Mon 9 Nov 2026, trading days only (no weekend)
DAYS = pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05",
                       "2026-11-06", "2026-11-09"])


def heads(times, ticker="TCS.NS", scores=None, fetched=None):
    n = len(times)
    return pd.DataFrame({
        "ticker": ticker, "published_at": times,
        "fetched_at": fetched or ["2026-11-10T00:00:00+00:00"] * n,
        "score": scores if scores is not None else [0.5] * n,
    })


@pytest.mark.parametrize("utc_time,expected", [
    ("2026-11-02T04:30:00+00:00", "2026-11-02"),  # 10:00 IST Mon -> Mon close
    ("2026-11-02T10:00:00+00:00", "2026-11-02"),  # 15:30 IST exactly -> Mon close
    ("2026-11-02T10:01:00+00:00", "2026-11-03"),  # 15:31 IST Mon -> Tue close
    ("2026-11-02T20:00:00+00:00", "2026-11-03"),  # 01:30 IST Tue -> Tue close
    ("2026-11-06T12:00:00+00:00", "2026-11-09"),  # Fri evening -> Mon
    ("2026-11-07T06:00:00+00:00", "2026-11-09"),  # Saturday -> Mon
])
def test_entry_date_alignment_has_no_lookahead(utc_time, expected):
    out = assign_entry_dates(heads([utc_time]), DAYS)
    assert out.iloc[0] == pd.Timestamp(expected)


def test_entry_date_falls_back_to_fetched_at_and_drops_future():
    h = heads([None, "2026-11-20T04:00:00+00:00"],
              fetched=["2026-11-03T04:00:00+00:00", "2026-11-20T05:00:00+00:00"])
    out = assign_entry_dates(h, DAYS)
    assert out.iloc[0] == pd.Timestamp("2026-11-03")
    assert pd.isna(out.iloc[1])  # after last known price


def test_forward_returns():
    px = pd.DataFrame({"TCS.NS": [100, 110, 99, 99, 120, 130]}, index=DAYS)
    fr = forward_returns(px, horizons=(1, 5)).set_index("date")
    assert fr.loc["2026-11-02", "ret_1d"] == pytest.approx(0.10)
    assert fr.loc["2026-11-03", "ret_1d"] == pytest.approx(-0.10)
    assert fr.loc["2026-11-02", "ret_5d"] == pytest.approx(0.30)
    assert np.isnan(fr.loc["2026-11-09", "ret_1d"])


def test_panel_averages_headlines_per_day():
    px = pd.DataFrame({"TCS.NS": [100, 110, 99, 99, 120, 130]}, index=DAYS)
    h = heads(["2026-11-02T04:00:00+00:00", "2026-11-02T05:00:00+00:00",
               "2026-11-03T04:00:00+00:00"], scores=[1.0, 0.0, -0.4])
    p = build_panel(h, px).set_index("date")
    assert p.loc["2026-11-02", "score"] == pytest.approx(0.5)
    assert p.loc["2026-11-02", "headlines"] == 2
    assert p.loc["2026-11-03", "ret_1d"] == pytest.approx(-0.10)


def test_summary_stats_known_values():
    panel = pd.DataFrame({
        "ticker": ["A"] * 4 + ["B"] * 2,
        "date": pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"] * 1
                               + ["2026-11-02", "2026-11-03"]),
        "score": [0.5, -0.5, 0.2, -0.1, 0.3, 0.0],
        "ret_1d": [0.02, -0.01, 0.01, 0.01, -0.03, 0.05],
        "ret_5d": [np.nan] * 6,
    })
    s = summarize(panel, horizons=(1, 5)).set_index(["ticker", "horizon"])
    a = s.loc[("A", "1d")]
    assert a.n == 4
    assert a.hit_rate == pytest.approx(3 / 4)   # last day: negative score, up move
    assert a.up_rate == pytest.approx(3 / 4)
    assert a.pearson == pytest.approx(np.corrcoef([0.5, -0.5, 0.2, -0.1],
                                                  [0.02, -0.01, 0.01, 0.01])[0, 1])
    b = s.loc[("B", "1d")]
    assert b.n == 2 and np.isnan(b.pearson)      # too few points for correlation
    assert b.n_hit == 1 and b.hit_rate == 0.0     # zero score excluded
    allr = s.loc[("ALL", "1d")]
    assert allr.n == 6 and allr.n_hit == 5
    assert s.loc[("ALL", "5d")].n == 0


def test_empty_inputs():
    panel = build_panel(pd.DataFrame(columns=["ticker", "published_at", "fetched_at", "score"]),
                        pd.DataFrame())
    assert panel.empty
    assert summarize(panel).empty
