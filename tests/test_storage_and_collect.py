import sys

from engine import storage
from scripts.collect import collect


def row(title, ticker="TCS.NS", score=0.5):
    return {"ticker": ticker, "title": title, "publisher": "P", "link": None,
            "published_at": "2026-11-02T04:00:00+00:00", "label": "positive",
            "confidence": 0.9, "p_positive": 0.7, "p_negative": 0.2, "p_neutral": 0.1,
            "score": score}


def test_save_and_load_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    assert storage.save_scored([row("a"), row("b")], "m", db) == 2
    df = storage.load_headlines(path=db)
    assert set(df["title"]) == {"a", "b"}
    assert (df["model"] == "m").all()
    assert df["fetched_at"].notna().all()


def test_duplicates_are_ignored(tmp_path):
    db = tmp_path / "t.db"
    storage.save_scored([row("a")], "m", db)
    assert storage.save_scored([row("a"), row("a", ticker="INFY.NS")], "m", db) == 1
    assert len(storage.load_headlines(path=db)) == 2
    assert storage.existing_titles("TCS.NS", db) == {"a"}
    assert len(storage.load_headlines("INFY.NS", db)) == 1


def test_save_empty_is_noop(tmp_path):
    assert storage.save_scored([], "m", tmp_path / "t.db") == 0


def fake_scorer(texts):
    return [{"label": "neutral", "confidence": 0.8, "p_positive": 0.1, "p_negative": 0.1,
             "p_neutral": 0.8, "score": 0.0} for _ in texts]


def test_collect_scores_only_new_headlines(tmp_path, fake_yfinance):
    db = tmp_path / "t.db"
    fake_yfinance["news"] = [{"title": "one"}, {"title": "two"}]
    scored = []

    def scorer(texts):
        scored.extend(texts)
        return fake_scorer(texts)

    s1 = collect(["TCS.NS"], db, pause=0, scorer=scorer)
    fake_yfinance["news"] = [{"title": "two"}, {"title": "three"}]
    s2 = collect(["TCS.NS"], db, pause=0, scorer=scorer)
    assert (s1["new"], s2["new"]) == (2, 1)
    assert scored == ["one", "two", "three"]


def test_collect_continues_after_ticker_failure(tmp_path, fake_yfinance):
    fake_yfinance["errors"] = 3  # first ticker exhausts its retries
    fake_yfinance["news"] = [{"title": "ok"}]
    s = collect(["TCS.NS", "INFY.NS"], tmp_path / "t.db", pause=0, scorer=fake_scorer)
    assert s["failed"] == ["TCS.NS"]
    assert s["new"] == 1


def test_collect_does_not_import_torch(tmp_path, fake_yfinance):
    fake_yfinance["news"] = []
    collect(["TCS.NS"], tmp_path / "t.db", pause=0, scorer=fake_scorer)
    assert "torch" not in sys.modules
