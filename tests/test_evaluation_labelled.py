import json

import pandas as pd
import pytest

from finpulse import cli
from finpulse.evaluation import (accuracy, apply_alternatives, bootstrap_ci, compare_rankings,
                                 evaluate_tagger, load_labelled, precision_recall, predict,
                                 score_labelled, tagger_markdown, to_markdown_ci)


def test_bootstrap_ci_brackets_point_estimate():
    y = ["positive", "negative", "neutral"] * 20
    p = ["positive", "negative", "positive"] * 20
    lo, hi = bootstrap_ci(y, p, accuracy, n_boot=500)
    assert lo <= accuracy(y, p) <= hi and hi - lo > 0
    assert bootstrap_ci(y, y, accuracy, n_boot=100) == (1.0, 1.0)


def test_load_labelled_normalises_and_drops_bad_rows(tmp_path):
    f = tmp_path / "l.csv"
    pd.DataFrame({"record_id": ["1", "2", "3", "4"],
                  "title": ["A up", "B down", "C", None],
                  "label": ["Positive", " negative", "unsure", "neutral"]}).to_csv(f, index=False)
    df, dropped = load_labelled(f)
    assert df["title"].tolist() == ["A up", "B down"]
    assert df["label"].tolist() == ["positive", "negative"]
    assert dropped == 2      # an unknown label, and a row with no title


def test_load_labelled_requires_columns(tmp_path):
    f = tmp_path / "l.csv"
    pd.DataFrame({"record_id": ["1"], "label": ["positive"]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="title"):
        load_labelled(f)


def fake_predictors():
    return {"vader": lambda ts: ["neutral"] * len(ts),
            "finbert": lambda ts: ["positive" if "up" in t else "negative" for t in ts],
            "zeroshot": lambda ts: ["positive"] * len(ts)}


def test_evaluate_labelled_rows_and_markdown():
    texts = ["A up", "B down", "C up", "D down", "E flat"]
    y = ["positive", "negative", "positive", "negative", "neutral"]
    models = ["majority", "vader", "finbert", "zeroshot"]
    rows = score_labelled(y, models, predict(texts, models, fake_predictors()))
    by = {r["model"].split(" (")[0]: r for r in rows}
    assert by["FinBERT"]["accuracy"] == pytest.approx(0.8)
    assert by["Zero-shot"]["accuracy"] == pytest.approx(0.4)
    md = to_markdown_ci(rows)
    assert md.count("\n") == len(rows) + 1 and "[" in md


def patch_predictors(monkeypatch):
    import finpulse.evaluation as ev
    real = ev.predict
    monkeypatch.setattr(ev, "predict",
                        lambda texts, models: real(texts, models, predictors=fake_predictors()))


def test_cli_evaluate_labelled(monkeypatch, tmp_path):
    f = tmp_path / "l.csv"
    pd.DataFrame({"record_id": ["1-1", "1-2", "1-3"], "label": ["positive", "negative", "neutral"],
                  "title": ["A up", "B down", "C flat"]}).to_csv(f, index=False)
    patch_predictors(monkeypatch)
    assert cli.main(["evaluate", "--labelled", str(f), "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "model_eval_indian.json").read_text())
    assert out["n"] == 3 and [r["model"].split(" (")[0] for r in out["results"]] == \
        ["majority", "VADER", "FinBERT", "Zero-shot"]
    assert "sensitivity" not in out
    assert (tmp_path / "model_eval_indian.md").read_text().startswith("| Model |")


AUDIT_COLS = ["sheet", "id", "title", "label_used", "plausible_alternative"]


def test_apply_alternatives_swaps_and_counts():
    df = pd.DataFrame({"record_id": ["a", "b", "c"], "label": ["neutral", "negative", "positive"]})
    audit = pd.DataFrame([["sentiment", "a", "t", "neutral", "positive"],
                          ["event_type", "X.NS", "t", "other", "macro"]], columns=AUDIT_COLS)
    out, n = apply_alternatives(df, audit, "sentiment", ["record_id"], "label", ["id"])
    assert n == 1 and out["label"].tolist() == ["positive", "negative", "positive"]
    assert df["label"].tolist() == ["neutral", "negative", "positive"]    # input untouched


@pytest.mark.parametrize("row", [["sentiment", "zz", "t", "neutral", "positive"],    # no such id
                                 ["sentiment", "a", "t", "negative", "positive"]])  # label differs
def test_apply_alternatives_rejects_stale_audit(row):
    df = pd.DataFrame({"record_id": ["a"], "label": ["neutral"]})
    with pytest.raises(ValueError):
        apply_alternatives(df, pd.DataFrame([row], columns=AUDIT_COLS), "sentiment",
                           ["record_id"], "label", ["id"])


def test_apply_alternatives_two_key_match():
    df = pd.DataFrame({"ticker": ["A.NS", "A.NS"], "title": ["x", "y"], "lab": ["other", "other"]})
    audit = pd.DataFrame([["event_type", "A.NS", "y", "other", "rumour"]], columns=AUDIT_COLS)
    out, n = apply_alternatives(df, audit, "event_type", ["ticker", "title"], "lab", ["id", "title"])
    assert n == 1 and out["lab"].tolist() == ["other", "rumour"]


def test_compare_rankings_detects_change():
    def rows(acc, f1):
        return [{"key": k, "accuracy": a, "macro_f1": f} for k, a, f in zip("xyz", acc, f1)]
    same = compare_rankings(rows([.5, .7, .6], [.4, .6, .5]), rows([.55, .72, .6], [.45, .6, .5]))
    assert not same["changed"] and same["macro_f1"]["original"] == ["y", "z", "x"]
    diff = compare_rankings(rows([.5, .7, .6], [.4, .6, .5]), rows([.5, .7, .6], [.4, .5, .6]))
    assert diff["changed"] and diff["macro_f1"]["changed"] and not diff["accuracy"]["changed"]


def test_cli_evaluate_labelled_with_alternatives(monkeypatch, tmp_path):
    f, a = tmp_path / "l.csv", tmp_path / "a.csv"
    pd.DataFrame({"record_id": ["1-1", "1-2", "1-3", "1-4"],
                  "label": ["positive", "negative", "neutral", "neutral"],
                  "title": ["A up", "B down", "C flat", "D up"]}).to_csv(f, index=False)
    pd.DataFrame([["sentiment", "1-4", "D up", "neutral", "positive"]],
                 columns=AUDIT_COLS).to_csv(a, index=False)
    patch_predictors(monkeypatch)
    assert cli.main(["evaluate", "--labelled", str(f), "--alternatives", str(a),
                     "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "model_eval_indian.json").read_text())
    sens = out["sensitivity"]
    assert sens["swapped"] == 1
    fin = {r["key"]: r for r in out["results"]}["finbert"]
    fin_alt = {r["key"]: r for r in sens["results"]}["finbert"]
    assert fin["accuracy"] == pytest.approx(0.5) and fin_alt["accuracy"] == pytest.approx(0.75)
    assert isinstance(sens["ranking"]["changed"], bool)
    assert "Sensitivity" in (tmp_path / "model_eval_indian.md").read_text()


def tagger_frame(labels):
    return pd.DataFrame({
        "ticker": ["A.NS", "B.NS", "C.NS", "D.NS"],
        "title": ["A Q2 profit rises", "SEBI fines B", "Rupee slips as crude rises",
                  "C opens new plant"],
        "multi_ticker": ["False", "False", "True", "False"],
        "event_type_hand": labels})


def test_evaluate_tagger_metrics():
    res = evaluate_tagger(tagger_frame(["earnings", "regulatory", "macro", "management"]))
    assert res["n"] == 4 and res["accuracy"] == pytest.approx(0.75)
    assert res["meets_threshold"]
    pc = res["per_class"]
    assert pc["other"]["precision"] == 0.0 and pc["other"]["support"] == 0
    assert pc["management"]["recall"] == 0.0 and pc["management"]["precision"] is None
    assert res["confusion"]["management"]["other"] == 1
    assert "| management | 1 | 0 | n/a | 0.00 |" in tagger_markdown(res)


def test_evaluate_tagger_rejects_unknown_categories():
    with pytest.raises(ValueError, match="categories"):
        evaluate_tagger(tagger_frame(["earnings", "legal", "macro", "other"]))


def test_precision_recall_counts():
    pr = precision_recall(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"])
    assert pr["a"] == {"precision": 1.0, "recall": 0.5, "support": 2, "predicted": 1}
    assert pr["b"]["precision"] == 0.5 and pr["c"]["recall"] is None


def test_cli_tagger_eval(tmp_path):
    f, a = tmp_path / "e.csv", tmp_path / "a.csv"
    tagger_frame(["earnings", "regulatory", "macro", "other"]).to_csv(f, index=False)
    pd.DataFrame([["event_type", "D.NS", "C opens new plant", "other", "management"]],
                 columns=AUDIT_COLS).to_csv(a, index=False)
    assert cli.main(["tagger-eval", "--labelled", str(f), "--alternatives", str(a),
                     "--out", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "event_tagger_eval.json").read_text())
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["sensitivity"]["swapped"] == 1
    assert out["sensitivity"]["accuracy"] == pytest.approx(0.75)
