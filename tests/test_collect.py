import sys

from finpulse.collect import collect


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
