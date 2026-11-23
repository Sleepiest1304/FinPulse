"""SQLite persistence for scored headlines."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    title        TEXT NOT NULL,
    publisher    TEXT,
    link         TEXT,
    published_at TEXT,             -- ISO-8601 UTC, from the news source
    fetched_at   TEXT NOT NULL,    -- ISO-8601 UTC, when we scored it
    label        TEXT NOT NULL,    -- positive / negative / neutral
    confidence   REAL NOT NULL,
    p_positive   REAL NOT NULL,
    p_negative   REAL NOT NULL,
    p_neutral    REAL NOT NULL,
    score        REAL NOT NULL,    -- p_positive - p_negative
    model        TEXT NOT NULL,
    UNIQUE (ticker, title)
);
CREATE INDEX IF NOT EXISTS idx_headlines_ticker_time
    ON headlines (ticker, published_at);
"""

COLUMNS = ["ticker", "title", "publisher", "link", "published_at", "fetched_at",
           "label", "confidence", "p_positive", "p_negative", "p_neutral",
           "score", "model"]


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def save_scored(rows: list[dict], model: str, path: Path | str = DB_PATH) -> int:
    """Insert scored headlines; duplicates (same ticker + title) are skipped.

    Returns the number of new rows written.
    """
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for r in rows:
        rec = {**r, "model": model, "fetched_at": now}
        records.append(tuple(rec.get(c) for c in COLUMNS))
    sql = (f"INSERT OR IGNORE INTO headlines ({', '.join(COLUMNS)}) "
           f"VALUES ({', '.join('?' for _ in COLUMNS)})")
    with closing(connect(path)) as conn, conn:
        before = conn.total_changes
        conn.executemany(sql, records)
        return conn.total_changes - before


def existing_titles(ticker: str, path: Path | str = DB_PATH) -> set[str]:
    with closing(connect(path)) as conn:
        cur = conn.execute("SELECT title FROM headlines WHERE ticker = ?", (ticker,))
        return {row[0] for row in cur}


def load_headlines(ticker: str | None = None, path: Path | str = DB_PATH) -> pd.DataFrame:
    """All stored headlines (optionally for one ticker) as a DataFrame."""
    query = f"SELECT {', '.join(COLUMNS)} FROM headlines"
    params: tuple = ()
    if ticker:
        query += " WHERE ticker = ?"
        params = (ticker,)
    query += " ORDER BY COALESCE(published_at, fetched_at) DESC"
    with closing(connect(path)) as conn:
        return pd.read_sql_query(query, conn, params=params)
