import csv
import gzip
import io
import zipfile
from urllib.error import HTTPError

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


def zipped(text, name="x.gkg.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_timestamps_cover_range_inclusive():
    ts = gdelt.gkg_timestamps("20260101000000", "20260101010000")
    assert ts == ["20260101000000", "20260101001500", "20260101003000",
                  "20260101004500", "20260101010000"]
    assert len(gdelt.gkg_timestamps("20260101000000", "20261124234500")) == 328 * 96
    with pytest.raises(ValueError):
        gdelt.gkg_timestamps("20260101000700", "20260101010000")


def test_indian_records_keeps_titled_indian_outlet_records():
    total, rows = gdelt.indian_records(SAMPLE)
    assert total == 6                                   # every record, titled or not
    # Reuters is not an Indian outlet, and record 6 has no title.
    assert [r[0] for r in rows] == ["20261015073000-1", "20261015073000-2",
                                    "20261015073000-3", "20261015073000-5"]
    assert rows[2][4] == "Tom Cruise reacts to 'Attack on Titan'"   # HTML entities unescaped
    assert rows[3][2] == "economictimes.indiatimes.com"


def test_indian_records_survives_quote_at_field_start():
    line = gkg_line("20261015073000-9", "livemint.com", "https://www.livemint.com/x",
                    'TCS "gains" today')
    line = line.replace("\t1\t", '\t"1\t', 1)           # a stray quote that would trip csv
    total, rows = gdelt.indian_records(line + "\n" + SAMPLE)
    assert total == 7 and rows[0][4] == 'TCS "gains" today'


def read_records(cache, ts):
    with gzip.open(cache / "records" / f"{ts}.csv.gz", "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_process_file_writes_records_and_manifest_row(tmp_path):
    (tmp_path / "records").mkdir()
    data = zipped(SAMPLE)
    row = gdelt.process_file("20261015073000", tmp_path, fetch=lambda url, t: data)
    assert row["status"] == "ok" and row["records"] == 6 and row["indian_records"] == 4
    assert row["compressed_bytes"] == len(data) and len(row["sha256"]) == 64
    assert len(read_records(tmp_path, "20261015073000")) == 4


def test_process_file_retries_then_skips(tmp_path):
    (tmp_path / "records").mkdir()
    calls = []

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) < 2:
            raise TimeoutError("slow")
        return zipped(SAMPLE)

    row = gdelt.process_file("20261015073000", tmp_path, fetch=flaky, sleep=lambda s: None)
    assert row["status"] == "retried" and row["attempts"] == 2

    def broken(url, timeout):
        raise ConnectionError("down")
    row = gdelt.process_file("20261015074500", tmp_path, fetch=broken, sleep=lambda s: None)
    assert row["status"] == "skipped" and row["attempts"] == 3 and "down" in row["error"]

    def missing(url, timeout):
        raise HTTPError(url, 404, "Not Found", None, None)
    row = gdelt.process_file("20261015080000", tmp_path, fetch=missing, sleep=lambda s: None)
    assert row["status"] == "skipped" and row["attempts"] == 1


def test_pull_resumes_without_refetching_or_double_counting(tmp_path):
    fetched = []
    fail_once = {"20261015074500"}

    def fetch(url, timeout):
        ts = url.rsplit("/", 1)[-1][:14]
        fetched.append(ts)
        if ts in fail_once:
            raise HTTPError(url, 404, "Not Found", None, None)
        return zipped(SAMPLE.replace("20261015073000-", f"{ts}-"))

    run = dict(cache=tmp_path, workers=2, pause=0, fetch=fetch, log=lambda m: None)
    s1 = gdelt.pull("20261015073000", "20261015080000", **run)
    assert s1["done"] == 3 and s1["skipped"] == 1 and s1["skipped_total"] == 1
    assert sorted(fetched) == ["20261015073000", "20261015074500", "20261015080000"]

    fetched.clear()
    fail_once.clear()
    s2 = gdelt.pull("20261015073000", "20261015080000", **run)
    assert fetched == ["20261015074500"]               # only the skipped file is tried again
    assert s2["skipped_total"] == 0

    manifest = gdelt.read_manifest(tmp_path / "manifest.csv")
    assert {r["status"] for r in manifest.values()} == {"ok"}
    recs = gdelt.load_records(tmp_path)
    assert len(recs) == 3 * 4 and recs["record_id"].is_unique
