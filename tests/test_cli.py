import json

import pandas as pd
import pytest

from finpulse import cli, storage


def test_collect_rejects_unknown_ticker(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["collect", "AAPL"])
    assert exc.value.code == 2
    assert "Unknown Nifty 50 ticker" in capsys.readouterr().err


def test_requires_a_command():
    with pytest.raises(SystemExit):
        cli.main([])


def test_collect_exit_code_reflects_total_failure(monkeypatch, tmp_path):
    calls = {}

    def fake_collect(tickers, db, pause):
        calls["tickers"] = tickers
        return {"fetched": 0, "new": 0, "failed": list(tickers)}

    monkeypatch.setattr("finpulse.collect.collect", fake_collect)
    assert cli.main(["collect", "tcs", "INFY", "--db", str(tmp_path / "t.db")]) == 1
    assert calls["tickers"] == ["TCS.NS", "INFY.NS"]


def test_backtest_with_empty_db_returns_1(tmp_path, capsys):
    assert cli.main(["backtest", "--db", str(tmp_path / "t.db"), "--out", str(tmp_path)]) == 1
    assert "No headlines stored yet" in capsys.readouterr().out


def test_backtest_writes_results(monkeypatch, tmp_path):
    db = tmp_path / "t.db"
    storage.save_scored([{"ticker": "TCS.NS", "title": "a", "publisher": None, "link": None,
                          "published_at": "2026-11-02T04:00:00+00:00", "label": "positive",
                          "confidence": 0.9, "p_positive": 0.9, "p_negative": 0.0,
                          "p_neutral": 0.1, "score": 0.9}], "m", db)
    days = pd.to_datetime(["2026-11-02", "2026-11-03"])
    monkeypatch.setattr("finpulse.prices.fetch_prices",
                        lambda tickers, start: pd.DataFrame({"TCS.NS": [100.0, 101.0]}, index=days))
    assert cli.main(["backtest", "--db", str(db), "--out", str(tmp_path)]) == 0
    panel = pd.read_csv(tmp_path / "backtest_panel.csv")
    assert panel.loc[0, "ret_1d"] == pytest.approx(0.01)
    assert (tmp_path / "backtest_summary.csv").exists()


def test_evaluate_writes_json_and_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr("finpulse.evaluation.load_phrasebank",
                        lambda subset: (["good", "bad"], ["positive", "negative"]))
    assert cli.main(["evaluate", "--subsets", "allagree", "--models", "majority",
                     "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "model_eval.json").read_text())
    assert data["results"][0]["n"] == 2
    assert "in-sample" in data["caveat"]
    assert (tmp_path / "model_eval.md").read_text().startswith("| Subset |")


def test_eventstudy_writes_summaries(tmp_path):
    db = tmp_path / "t.db"
    days = pd.bdate_range("2026-01-05", periods=30)
    words = ["orders", "hiring", "deal", "outage", "margin", "visa", "cloud", "layoffs", "bonus", "office"]
    rows = [{"ticker": "TCS.NS", "title": f"TCS {words[i // 2]} update {i}",
             "publisher": None, "link": None,
             "published_at": f"{days[i].date()}T04:00:00+00:00", "label": "negative",
             "confidence": 0.9, "p_positive": 0.05, "p_negative": 0.9, "p_neutral": 0.05,
             "score": -0.85} for i in range(0, 20, 2)]
    storage.save_scored(rows, model="test", path=db)
    prices = pd.DataFrame({"date": days, "TCS.NS": range(100, 130), "^NSEI": range(200, 230)})
    prices.to_csv(tmp_path / "p.csv", index=False)
    pd.DataFrame({"earnings_date": ["2026-01-02 05:00:00-05:00"], "EPS Estimate": [1.0],
                  "Reported EPS": [1.1], "Surprise(%)": [10.0], "ticker": ["TCS.NS"]}
                 ).to_csv(tmp_path / "e.csv", index=False)
    code = cli.main(["eventstudy", "--db", str(db), "--prices", str(tmp_path / "p.csv"),
                     "--earnings", str(tmp_path / "e.csv"), "--out", str(tmp_path), "--perm", "20"])
    assert code == 0
    conf = pd.read_csv(tmp_path / "v3_confirmatory.csv")
    assert conf["id"].tolist() == ["H1", "H2", "H3", "H4"]
    counts = json.loads((tmp_path / "v3_counts.json").read_text())
    assert counts["headlines"] == 10 and counts["negative_events"] == 10
    assert (tmp_path / "v3_exploratory.csv").exists() and (tmp_path / "v3_robustness.csv").exists()
