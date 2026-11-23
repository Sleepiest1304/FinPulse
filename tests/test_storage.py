from finpulse import storage


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
