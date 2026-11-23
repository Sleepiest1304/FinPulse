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
