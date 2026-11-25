import json

import pandas as pd
import pytest

from finpulse import cli
from finpulse.evaluation import (bootstrap_ci, accuracy, evaluate_labelled, load_labelled,
                                 to_markdown_ci)


def test_bootstrap_ci_brackets_point_estimate():
    y = ["positive", "negative", "neutral"] * 20
    p = ["positive", "negative", "positive"] * 20
    lo, hi = bootstrap_ci(y, p, accuracy, n_boot=500)
    assert lo <= accuracy(y, p) <= hi and hi - lo > 0
    assert bootstrap_ci(y, y, accuracy, n_boot=100) == (1.0, 1.0)


def test_load_labelled_rebuilds_missing_titles_and_drops_bad_rows(tmp_path):
    f = tmp_path / "l.csv"
    pd.DataFrame({"record_id": ["20261015073000-1", "20261015073000-2", "20261015073000-3",
                                "20261015073000-4"],
                  "label": ["Positive", "negative", "unsure", "neutral"]}).to_csv(f, index=False)
    rebuilt = {"20261015073000-1": "A up", "20261015073000-2": "B down", "20261015073000-3": "C"}
    texts, y, dropped = load_labelled(f, rebuild=lambda ids: {i: rebuilt[i] for i in ids if i in rebuilt})
    assert texts == ["A up", "B down"] and y == ["positive", "negative"]
    assert dropped == 2      # bad label, and a record GDELT no longer returns


def test_load_labelled_requires_columns(tmp_path):
    f = tmp_path / "l.csv"
    pd.DataFrame({"title": ["x"]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        load_labelled(f)


def fake_predictors():
    return {"vader": lambda ts: ["neutral"] * len(ts),
            "finbert": lambda ts: ["positive" if "up" in t else "negative" for t in ts],
            "zeroshot": lambda ts: ["positive"] * len(ts)}


def test_evaluate_labelled_rows_and_markdown():
    texts = ["A up", "B down", "C up", "D down", "E flat"]
    y = ["positive", "negative", "positive", "negative", "neutral"]
    rows = evaluate_labelled(texts, y, ["majority", "vader", "finbert", "zeroshot"],
                             predictors=fake_predictors())
    by = {r["model"].split(" (")[0]: r for r in rows}
    assert by["FinBERT"]["accuracy"] == pytest.approx(0.8)
    assert by["Zero-shot"]["accuracy"] == pytest.approx(0.4)
    md = to_markdown_ci(rows)
    assert md.count("\n") == len(rows) + 1 and "[" in md


def test_cli_evaluate_labelled(monkeypatch, tmp_path):
    f = tmp_path / "l.csv"
    pd.DataFrame({"record_id": ["1-1", "1-2", "1-3"], "label": ["positive", "negative", "neutral"],
                  "title": ["A up", "B down", "C flat"]}).to_csv(f, index=False)
    import finpulse.evaluation as ev
    real = ev.evaluate_labelled
    monkeypatch.setattr(ev, "evaluate_labelled",
                        lambda texts, y, models: real(texts, y, models, predictors=fake_predictors()))
    assert cli.main(["evaluate", "--labelled", str(f), "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "model_eval_indian.json").read_text())
    assert out["n"] == 3 and [r["model"].split(" (")[0] for r in out["results"]] == \
        ["majority", "VADER", "FinBERT", "Zero-shot"]
    assert (tmp_path / "model_eval_indian.md").read_text().startswith("| Model |")
