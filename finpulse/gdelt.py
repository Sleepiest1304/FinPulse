"""Indian-outlet headlines from GDELT 2.0 Global Knowledge Graph (GKG) files.

GDELT publishes one GKG file every 15 minutes. Each record has a source
domain, URL and, in the Extras field, the page title. ``pull`` downloads a
range of files and keeps the records from Indian outlets; ``match_tickers``
tags titles that name a Nifty 50 company (strict aliases, whole words).
GDELT data is free to use with a citation: https://www.gdeltproject.org/.

``INDIAN_DOMAINS`` and ``ALIASES`` are fixed by the H3 replication
amendment in docs/v3-preregistration.md; do not change them.

The pull is resumable. Everything goes to a cache directory outside the
repository (headlines are third-party text):

    <cache>/records/<timestamp>.csv.gz   Indian-outlet records of one file
    <cache>/manifest.csv                 one row per file attempt

Zips are held in memory only and never written to disk. A file counts as
done once its records file exists and the manifest has an ``ok`` or
``retried`` row for it, so an interrupted pull can simply be rerun: done
files are never fetched again, and each file's records live in exactly one
file, so nothing is counted twice. Skipped files are tried again on a rerun.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

GKG_URL = "https://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"
USER_AGENT = "finpulse-research/1.0"

INDIAN_DOMAINS = re.compile(
    r"(?:^|\.)(?:economictimes\.indiatimes\.com|economictimes\.com|moneycontrol\.com|"
    r"livemint\.com|business-standard\.com|thehindubusinessline\.com|"
    r"financialexpress\.com|businesstoday\.in|cnbctv18\.com|ndtvprofit\.com|"
    r"thehindu\.com|hindustantimes\.com|timesofindia\.indiatimes\.com|"
    r"indianexpress\.com|ndtv\.com|news18\.com|indiatoday\.in|zeebiz\.com|"
    r"business-standard\.in|[a-z0-9-]+\.in)$", re.IGNORECASE)

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


MANIFEST_COLS = ["file", "compressed_bytes", "records", "indian_records", "sha256",
                 "status", "attempts", "error", "finished_utc"]
RECORD_COLS = ["record_id", "date", "domain", "url", "title"]
DONE = {"ok", "retried"}


def gkg_timestamps(start: str, end: str) -> list[str]:
    """Every 15-minute GKG timestamp from ``start`` to ``end`` inclusive (YYYYMMDDHHMMSS)."""
    t = datetime.strptime(start, "%Y%m%d%H%M%S")
    stop = datetime.strptime(end, "%Y%m%d%H%M%S")
    if t.minute % 15 or t.second:
        raise ValueError(f"start is not on a 15-minute boundary: {start}")
    out = []
    while t <= stop:
        out.append(t.strftime("%Y%m%d%H%M%S"))
        t += timedelta(minutes=15)
    return out


def indian_records(text: str) -> tuple[int, list[list[str]]]:
    """Total record count and the Indian-outlet records (with a title) of one GKG file.

    GKG is tab-separated with one record per line and no quoting, so lines
    are split directly (``csv`` would treat a leading quote as quoting).
    """
    total, keep = 0, []
    for line in text.split("\n"):
        if not line.strip():
            continue
        total += 1
        parts = line.split("\t", 5)
        if len(parts) < 6:
            continue
        domain = parts[3].strip().lower()
        if not INDIAN_DOMAINS.search(domain):
            continue
        title = page_title(line.rsplit("\t", 1)[-1])
        if title:
            keep.append([parts[0], parts[1], domain, parts[4], title])
    return total, keep


def read_manifest(path: Path) -> dict[str, dict]:
    """Latest manifest row per file."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def done_files(cache: Path) -> set[str]:
    rows = read_manifest(cache / "manifest.csv")
    return {f for f, r in rows.items()
            if r["status"] in DONE and (cache / "records" / f"{f}.csv.gz").exists()}


def _download(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def process_file(ts: str, cache: Path, fetch=_download, attempts: int = 3,
                 base_delay: float = 5.0, timeout: float = 120.0, sleep=time.sleep) -> dict:
    """Fetch one file, write its Indian-outlet records, return its manifest row."""
    url = GKG_URL.format(ts=ts)
    error = ""
    for attempt in range(1, attempts + 1):
        try:
            data = fetch(url, timeout)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
            total, rows = indian_records(text)
            out = cache / "records" / f"{ts}.csv.gz"
            tmp = out.with_suffix(".tmp")
            with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(RECORD_COLS)
                w.writerows(rows)
            os.replace(tmp, out)
            return {"file": ts, "compressed_bytes": len(data), "records": total,
                    "indian_records": len(rows), "sha256": hashlib.sha256(data).hexdigest(),
                    "status": "ok" if attempt == 1 else "retried", "attempts": attempt,
                    "error": error, "finished_utc": _now()}
        except Exception as exc:  # noqa: BLE001 - every failure is logged in the manifest
            error = f"{type(exc).__name__}: {exc}"[:200]
            if isinstance(exc, HTTPError) and exc.code == 404:
                break                       # the file does not exist; retrying will not help
            if attempt < attempts:
                sleep(base_delay * 2 ** (attempt - 1))
    return {"file": ts, "compressed_bytes": "", "records": "", "indian_records": "",
            "sha256": "", "status": "skipped", "attempts": attempt, "error": error,
            "finished_utc": _now()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pull(start: str, end: str, cache: Path, workers: int = 3, pause: float = 0.5,
         progress_every: int = 300, fetch=_download, log=print) -> dict:
    """Pull every file in the range that is not already done. Returns a summary."""
    cache = Path(cache)
    (cache / "records").mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.csv"
    new_manifest = not manifest.exists()
    todo = [ts for ts in gkg_timestamps(start, end) if ts not in done_files(cache)]
    log(f"{len(todo)} files to fetch ({len(gkg_timestamps(start, end)) - len(todo)} already done)")

    lock = threading.Lock()
    stats = {"done": 0, "skipped": 0, "bytes": 0, "indian": 0}
    t0 = time.monotonic()

    def job(ts):
        row = process_file(ts, cache, fetch=fetch)
        time.sleep(pause)
        return row

    with open(manifest, "a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_COLS)
        if new_manifest:
            w.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed(pool.submit(job, ts) for ts in todo):
                row = fut.result()
                with lock:
                    w.writerow(row)
                    mf.flush()
                    stats["done"] += 1
                    if row["status"] == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["bytes"] += row["compressed_bytes"]
                        stats["indian"] += row["indian_records"]
                    n = stats["done"]
                    if n % progress_every == 0 or n == len(todo):
                        rate = n / max(time.monotonic() - t0, 1e-9)
                        eta = (len(todo) - n) / rate if rate else float("nan")
                        log(f"{n}/{len(todo)} files, {stats['bytes'] / 1e9:.1f} GB, "
                            f"{stats['indian']} Indian-outlet records, {stats['skipped']} skipped, "
                            f"ETA {eta / 3600:.1f} h")
    stats["skipped_total"] = sum(r["status"] == "skipped" for r in read_manifest(manifest).values())
    return stats


def load_records(cache: Path):
    """All cached Indian-outlet records, one row per record id."""
    import pandas as pd

    cache = Path(cache)
    files = sorted(done_files(cache))
    frames = [pd.read_csv(cache / "records" / f"{f}.csv.gz", dtype=str, keep_default_na=False)
              for f in files]
    if not frames:
        return pd.DataFrame(columns=RECORD_COLS)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("record_id").reset_index(drop=True)
