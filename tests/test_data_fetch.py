import pandas as pd
import pytest

from engine.config import validate_ticker
from engine.data_fetch import fetch_news, fetch_prices, parse_news_item

NEW_SCHEMA = {
    "id": "abc",
    "content": {
        "title": "Reliance shares rise after strong quarterly results",
        "pubDate": "2026-11-19T01:14:26Z",
        "provider": {"displayName": "Reuters"},
        "canonicalUrl": {"url": "https://example.com/a"},
    },
}
OLD_SCHEMA = {
    "title": "TCS wins large deal",
    "publisher": "Mint",
    "link": "https://example.com/b",
    "providerPublishTime": 1763514866,
}


def test_parse_new_schema():
    item = parse_news_item(NEW_SCHEMA)
    assert item == {
        "title": "Reliance shares rise after strong quarterly results",
        "publisher": "Reuters",
        "link": "https://example.com/a",
        "published_at": "2026-11-19T01:14:26+00:00",
    }


def test_parse_old_schema():
    item = parse_news_item(OLD_SCHEMA)
    assert item["title"] == "TCS wins large deal"
    assert item["publisher"] == "Mint"
    assert item["published_at"] == "2025-11-19T01:14:26+00:00"


@pytest.mark.parametrize("raw", [
    None, "a string", {}, {"content": None}, {"title": ""}, {"title": "   "},
    {"title": 123}, {"content": {"summary": "no title here"}},
])
def test_parse_rejects_items_without_title(raw):
    assert parse_news_item(raw) is None


def test_parse_tolerates_missing_and_malformed_fields():
    item = parse_news_item({"content": {"title": "  Spaced \n  title ", "pubDate": "not a date",
                                        "provider": "oops", "canonicalUrl": None}})
    assert item == {"title": "Spaced title", "publisher": None, "link": None,
                    "published_at": None}


def test_parse_truncates_long_titles():
    assert len(parse_news_item({"title": "x" * 5000})["title"]) == 500


def test_fetch_news_filters_and_dedupes(fake_yfinance):
    fake_yfinance["news"] = [NEW_SCHEMA, {"content": {}}, OLD_SCHEMA,
                             {"title": "TCS WINS LARGE DEAL"}]
    items = fetch_news("tcs")
    assert [i["title"] for i in items] == [
        "Reliance shares rise after strong quarterly results", "TCS wins large deal"]
    assert all(i["ticker"] == "TCS.NS" for i in items)


def test_fetch_news_empty_returns_empty_list(fake_yfinance):
    fake_yfinance["news"] = None
    assert fetch_news("INFY.NS") == []


def test_fetch_news_retries_then_succeeds(fake_yfinance):
    fake_yfinance["errors"] = 2
    fake_yfinance["news"] = [OLD_SCHEMA]
    assert len(fetch_news("TCS.NS")) == 1
    assert fake_yfinance["calls"] == 3


def test_fetch_news_raises_after_exhausting_retries(fake_yfinance):
    fake_yfinance["errors"] = 5
    with pytest.raises(ConnectionError):
        fetch_news("TCS.NS")
    assert fake_yfinance["calls"] == 3


def test_fetch_news_rejects_unknown_ticker(fake_yfinance):
    with pytest.raises(ValueError):
        fetch_news("AAPL")
    assert fake_yfinance["calls"] == 0


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


@pytest.mark.parametrize("raw,expected", [
    ("reliance", "RELIANCE.NS"), (" TCS.NS ", "TCS.NS"), ("m&m", "M&M.NS")])
def test_validate_ticker_ok(raw, expected):
    assert validate_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "HDFC BANK.NS", "AAPL", None, 5, "TCS.NS; DROP TABLE"])
def test_validate_ticker_rejects(raw):
    with pytest.raises(ValueError):
        validate_ticker(raw)
