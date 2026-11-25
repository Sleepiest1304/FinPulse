"""Indian financial headlines from GDELT 2.0 Global Knowledge Graph (GKG) files.

GDELT publishes one GKG file every 15 minutes. Each record has a source
domain, URL and, in the Extras field, the page title. This module downloads
a file, keeps records from Indian outlets, and tags titles that name a
Nifty 50 company (strict aliases, whole-word). GDELT data is free to use
with a citation: https://www.gdeltproject.org/.

Downloaded headlines are third-party text and stay out of the repository.
A record is identified by its GKG record id (``<timestamp>-<n>``), so a
labelled set can be shared as ids + labels and rebuilt from GDELT.
"""
from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from urllib.request import Request, urlopen

import pandas as pd

from .retry import retry

GKG_URL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
USER_AGENT = "finpulse-research/1.0"

INDIAN_DOMAINS = re.compile(
    r"(?:^|\.)(?:economictimes\.indiatimes\.com|economictimes\.com|moneycontrol\.com|"
    r"livemint\.com|business-standard\.com|thehindubusinessline\.com|"
    r"financialexpress\.com|businesstoday\.in|cnbctv18\.com|ndtvprofit\.com|"
    r"thehindu\.com|hindustantimes\.com|timesofindia\.indiatimes\.com|"
    r"indianexpress\.com|ndtv\.com|news18\.com|indiatoday\.in|zeebiz\.com|"
    r"business-standard\.in|[a-z0-9-]+\.in)$", re.IGNORECASE)

BUSINESS_PATH = re.compile(
    r"/(?:markets?|business|companies|company|economy|stocks?|industry|money|"
    r"finance|banking|earnings|results|corporate)(?:/|-|$)", re.IGNORECASE)

ALIASES = {
    "ADANIENT.NS": ["adani enterprises"], "ADANIPORTS.NS": ["adani ports"],
    "APOLLOHOSP.NS": ["apollo hospitals"], "ASIANPAINT.NS": ["asian paints"],
    "AXISBANK.NS": ["axis bank"], "BAJAJ-AUTO.NS": ["bajaj auto"],
    "BAJAJFINSV.NS": ["bajaj finserv"], "BAJFINANCE.NS": ["bajaj finance"],
    "BEL.NS": ["bharat electronics"], "BHARTIARTL.NS": ["bharti airtel", "airtel"],
    "CIPLA.NS": ["cipla"], "COALINDIA.NS": ["coal india"],
    "DRREDDY.NS": ["dr reddy's", "dr. reddy's", "dr reddys", "dr reddy"],
    "EICHERMOT.NS": ["eicher motors", "royal enfield"],
    "ETERNAL.NS": ["zomato", "blinkit", "eternal ltd"], "GRASIM.NS": ["grasim"],
    "HCLTECH.NS": ["hcltech", "hcl tech", "hcl technologies"],
    "HDFCBANK.NS": ["hdfc bank"], "HDFCLIFE.NS": ["hdfc life"], "HINDALCO.NS": ["hindalco"],
    "HINDUNILVR.NS": ["hindustan unilever", "hul"], "ICICIBANK.NS": ["icici bank"],
    "INDIGO.NS": ["indigo", "interglobe aviation"], "INFY.NS": ["infosys"],
    "ITC.NS": ["itc"], "JIOFIN.NS": ["jio financial"], "JSWSTEEL.NS": ["jsw steel"],
    "KOTAKBANK.NS": ["kotak mahindra bank", "kotak bank"],
    "LT.NS": ["larsen & toubro", "larsen and toubro", "l&t"],
    "M&M.NS": ["mahindra & mahindra", "mahindra and mahindra", "m&m"],
    "MARUTI.NS": ["maruti suzuki", "maruti"], "MAXHEALTH.NS": ["max healthcare"],
    "NESTLEIND.NS": ["nestle india"], "NTPC.NS": ["ntpc"], "ONGC.NS": ["ongc"],
    "POWERGRID.NS": ["power grid corporation", "powergrid"],
    "RELIANCE.NS": ["reliance industries", "ril"], "SBILIFE.NS": ["sbi life"],
    "SBIN.NS": ["state bank of india", r"sbi(?! life| card| general| mutual| funds?)"],
    "SHRIRAMFIN.NS": ["shriram finance"], "SUNPHARMA.NS": ["sun pharma", "sun pharmaceutical"],
    "TATACONSUM.NS": ["tata consumer"], "TATASTEEL.NS": ["tata steel"],
    "TCS.NS": ["tcs", "tata consultancy"], "TECHM.NS": ["tech mahindra"],
    "TITAN.NS": ["titan company"], "TMPV.NS": ["tata motors"], "TRENT.NS": ["trent"],
    "ULTRACEMCO.NS": ["ultratech"], "WIPRO.NS": ["wipro"],
}

_RAW = {"SBIN.NS"}   # aliases that are already regular expressions


def _pattern(ticker: str, alias: str) -> str:
    body = alias if ticker in _RAW and "(" in alias else re.escape(alias)
    return rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"


ALIAS_PATTERNS = {
    t: re.compile("|".join(_pattern(t, a) for a in names), re.IGNORECASE)
    for t, names in ALIASES.items()
}
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)


def match_tickers(title: str) -> list[str]:
    """Nifty 50 tickers whose company is named in the title."""
    text = (title or "").replace("’", "'")
    return [t for t, p in ALIAS_PATTERNS.items() if p.search(text)]


def page_title(extras: str) -> str | None:
    m = _TITLE.search(extras or "")
    if not m:
        return None
    t = " ".join(html.unescape(m.group(1)).split())
    return t or None


def parse_gkg(text: str) -> pd.DataFrame:
    """Records with a title: record_id, date (UTC), domain, url, title."""
    csv.field_size_limit(10 ** 9)
    rows = []
    for r in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(r) < 5:
            continue
        title = page_title(r[-1])
        if not title:
            continue
        rows.append({"record_id": r[0], "date": r[1], "domain": r[3].lower(),
                     "url": r[4], "title": title})
    df = pd.DataFrame(rows, columns=["record_id", "date", "domain", "url", "title"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    return df


def indian_financial(df: pd.DataFrame) -> pd.DataFrame:
    """Indian-outlet records that name a Nifty 50 company or sit in a business section."""
    if df.empty:
        return df.assign(tickers=[], business=[])
    out = df[df["domain"].str.contains(INDIAN_DOMAINS)].copy()
    out["tickers"] = out["title"].map(lambda t: ",".join(match_tickers(t)))
    out["business"] = out["url"].str.contains(BUSINESS_PATH)
    return out[(out["tickers"] != "") | out["business"]]


def download_gkg(ts: str, timeout: float = 90.0) -> str:
    """Text of one GKG file; ``ts`` is YYYYMMDDHHMMSS on a 15-minute boundary."""
    if not re.fullmatch(r"\d{14}", ts) or ts[10:12] not in ("00", "15", "30", "45") or ts[12:] != "00":
        raise ValueError(f"not a GKG timestamp: {ts!r}")
    return _fetch_zip(GKG_URL.format(ts=ts), timeout)


@retry(attempts=3, base_delay=5.0)
def _fetch_zip(url: str, timeout: float) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")


def record_timestamp(record_id: str) -> str:
    """The GKG file a record came from (its id starts with the file timestamp)."""
    ts = str(record_id).split("-")[0]
    if not re.fullmatch(r"\d{14}", ts):
        raise ValueError(f"not a GKG record id: {record_id!r}")
    return ts


def rebuild_titles(record_ids, fetch=download_gkg) -> dict[str, str]:
    """Look titles up again from GDELT, so only ids need to be stored."""
    wanted: dict[str, set[str]] = {}
    for rid in record_ids:
        wanted.setdefault(record_timestamp(rid), set()).add(str(rid))
    found = {}
    for ts, ids in sorted(wanted.items()):
        df = parse_gkg(fetch(ts))
        hit = df[df["record_id"].isin(ids)]
        found.update(zip(hit["record_id"], hit["title"]))
    return found
