import io
import zipfile

import pytest

from finpulse import gdelt


def gkg_line(rid, domain, url, title, date="20261015073000"):
    cols = [""] * 27
    cols[0], cols[1], cols[2], cols[3], cols[4] = rid, date, "1", domain, url
    cols[-1] = f"<PAGE_LINKS></PAGE_LINKS><PAGE_TITLE>{title}</PAGE_TITLE>" if title else ""
    return "\t".join(cols)


SAMPLE = "\n".join([
    gkg_line("20261015073000-1", "livemint.com", "https://www.livemint.com/market/stock-market-news/x.html",
             "TCS, Infosys gain; Nifty IT jumps 9% in 3 days"),
    gkg_line("20261015073000-2", "businesstoday.in", "https://www.businesstoday.in/latest/corporate/wipro-q2",
             "Wipro Q2 profit slips slightly YoY to Rs 3,201 crore"),
    gkg_line("20261015073000-3", "thehindu.com", "https://www.thehindu.com/entertainment/movies/x",
             "Tom Cruise reacts to &#39;Attack on Titan&#39;"),
    gkg_line("20261015073000-4", "reuters.com", "https://www.reuters.com/business/x",
             "Infosys wins European deal"),
    gkg_line("20261015073000-5", "economictimes.indiatimes.com",
             "https://economictimes.indiatimes.com/markets/stocks/news/x", "Rupee ends flat"),
    gkg_line("20261015073000-6", "livemint.com", "https://www.livemint.com/x", None),
])


@pytest.mark.parametrize("title, expected", [
    ("Wipro Q2 profit slips", ["WIPRO.NS"]),
    ("Dr Reddy’s flags semaglutide delay", ["DRREDDY.NS"]),
    ("L&T bags mega order; M&M sales rise", ["LT.NS", "M&M.NS"]),
    ("SBI raises lending rates", ["SBIN.NS"]),
    ("SBI Life premium grows", ["SBILIFE.NS"]),
    ("Attack on Titan season finale", []),          # "Titan" alone is not Titan Company
    ("Mastcs and Hultra", []),                       # whole words only
    ("Reliance Industries and Tata Motors in focus", ["RELIANCE.NS", "TMPV.NS"]),
    (None, []),
])
def test_match_tickers(title, expected):
    assert gdelt.match_tickers(title) == expected


def test_every_nifty_ticker_has_an_alias():
    from finpulse.config import NIFTY50
    assert set(gdelt.ALIASES) == set(NIFTY50)


def test_parse_gkg_extracts_titles_and_unescapes():
    df = gdelt.parse_gkg(SAMPLE)
    assert len(df) == 5                       # the record without a title is dropped
    assert df.loc[df["record_id"] == "20261015073000-3", "title"].item() == \
        "Tom Cruise reacts to 'Attack on Titan'"
    assert str(df["date"].dt.tz) == "UTC"


def test_indian_financial_filter():
    out = gdelt.indian_financial(gdelt.parse_gkg(SAMPLE))
    # Reuters is not Indian; the film story names no company and is not a business URL.
    assert out["record_id"].tolist() == ["20261015073000-1", "20261015073000-2", "20261015073000-5"]
    assert out.set_index("record_id").loc["20261015073000-1", "tickers"] == "INFY.NS,TCS.NS"
    assert out.set_index("record_id").loc["20261015073000-5", "business"]


def test_download_validates_timestamp():
    with pytest.raises(ValueError):
        gdelt.download_gkg.__wrapped__("2026101507") if hasattr(gdelt.download_gkg, "__wrapped__") \
            else gdelt.download_gkg("2026101507")


def test_download_reads_zip(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("20261015073000.gkg.csv", SAMPLE)

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        return Resp(buf.getvalue())

    monkeypatch.setattr(gdelt, "urlopen", fake_urlopen)
    text = gdelt.download_gkg("20261015073000")
    assert "Wipro" in text
    assert seen["url"].endswith("/gdeltv2/20261015073000.gkg.csv.zip")


def test_rebuild_titles_groups_by_file():
    calls = []

    def fetch(ts):
        calls.append(ts)
        return SAMPLE

    got = gdelt.rebuild_titles(["20261015073000-2", "20261015073000-5", "20261015073000-99"], fetch)
    assert calls == ["20261015073000"]
    assert got == {"20261015073000-2": "Wipro Q2 profit slips slightly YoY to Rs 3,201 crore",
                   "20261015073000-5": "Rupee ends flat"}


def test_record_timestamp_rejects_bad_ids():
    assert gdelt.record_timestamp("20261015073000-12") == "20261015073000"
    with pytest.raises(ValueError):
        gdelt.record_timestamp("abc-1")
