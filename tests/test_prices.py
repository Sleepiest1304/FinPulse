import pandas as pd

from finpulse.prices import fetch_prices


def test_fetch_prices_multi_ticker(fake_yfinance):
    idx = pd.to_datetime(["2026-11-02", "2026-11-03"])
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["TCS.NS", "INFY.NS"]])
    fake_yfinance["download"] = pd.DataFrame([[1, 2, 0, 0], [3, 4, 0, 0]], index=idx, columns=cols)
    out = fetch_prices(["TCS.NS", "INFY.NS"], start="2026-11-01")
    assert list(out.columns) == ["TCS.NS", "INFY.NS"]
    assert out.loc["2026-11-03", "INFY.NS"] == 4


def test_fetch_prices_single_flat_and_empty(fake_yfinance):
    idx = pd.to_datetime(["2026-11-02"])
    fake_yfinance["download"] = pd.DataFrame({"Close": [10.0], "Open": [9.0]}, index=idx)
    assert list(fetch_prices(["TCS"], start="2026-11-01").columns) == ["TCS.NS"]
    fake_yfinance["download"] = pd.DataFrame()
    assert fetch_prices(["TCS"], start="2026-11-01").empty
