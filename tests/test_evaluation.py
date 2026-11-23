import pytest

from finpulse.evaluation import (accuracy, evaluate, macro_f1, parse_phrasebank,
                                 per_class_f1, to_markdown, vader_label, weighted_f1)


def test_metrics_hand_computed():
    y = ["positive", "positive", "negative", "neutral"]
    p = ["positive", "neutral", "negative", "neutral"]
    assert accuracy(y, p) == 0.75
    f1 = per_class_f1(y, p)
    assert f1["negative"] == 1.0
    assert f1["positive"] == pytest.approx(2 / 3)   # P=1, R=0.5
    assert f1["neutral"] == pytest.approx(2 / 3)    # P=0.5, R=1
    assert macro_f1(y, p) == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
    assert weighted_f1(y, p) == pytest.approx((2 * 2 / 3 + 1 + 2 / 3) / 4)


def test_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_vader_thresholds():
    assert vader_label(0.05) == "positive"
    assert vader_label(-0.05) == "negative"
    assert vader_label(0.0) == "neutral"


def test_parse_phrasebank_handles_at_signs_and_junk():
    texts, labels = parse_phrasebank([
        "Profit rose to EUR 5 mn @ 3 % margin .@positive",
        "Sales were flat .@neutral\n",
        "garbage line",
        "Something@unknown",
    ])
    assert texts == ["Profit rose to EUR 5 mn @ 3 % margin .", "Sales were flat ."]
    assert labels == ["positive", "neutral"]


def fake_loader(subset):
    return (["Profit rose sharply", "Sales were flat", "Losses widened badly"],
            ["positive", "neutral", "negative"])


def test_evaluate_majority_and_vader_without_finbert():
    rows = evaluate(["allagree"], ["majority", "vader"], loader=fake_loader)
    assert [r["model"] for r in rows] == ["majority (positive)", "VADER"]
    assert rows[0]["n"] == 3 and rows[0]["accuracy"] == pytest.approx(1 / 3)
    assert all(r["subset"] == "allagree" for r in rows)


def test_to_markdown_formats_rows():
    md = to_markdown([{"subset": "s", "model": "m", "n": 2, "accuracy": 0.5,
                       "macro_f1": 0.25, "weighted_f1": 1 / 3}])
    assert md.splitlines()[-1] == "| s | m | 2 | 0.500 | 0.250 | 0.333 |"
