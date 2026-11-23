import numpy as np
import pytest

from engine.evaluation import (accuracy, macro_f1, parse_phrasebank, per_class_f1,
                               vader_label, weighted_f1)
from engine.sentiment import get_sentiment, probs_to_results

FINBERT_LABELS = {0: "positive", 1: "negative", 2: "neutral"}


def test_probs_to_results_uses_model_label_order():
    res = probs_to_results(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]), FINBERT_LABELS)
    assert res[0]["label"] == "positive" and res[0]["confidence"] == pytest.approx(0.7)
    assert res[0]["score"] == pytest.approx(0.5)
    assert res[1]["label"] == "neutral" and res[1]["score"] == pytest.approx(0.0)

    shuffled = {0: "Neutral", 1: "Positive", 2: "Negative"}
    r = probs_to_results(np.array([[0.1, 0.2, 0.7]]), shuffled)[0]
    assert r["label"] == "negative"
    assert r["p_positive"] == pytest.approx(0.2)
    assert r["score"] == pytest.approx(-0.5)


def test_probs_to_results_validates_input():
    with pytest.raises(ValueError):
        probs_to_results(np.array([0.1, 0.2, 0.7]), FINBERT_LABELS)
    with pytest.raises(ValueError):
        probs_to_results(np.array([[0.5, 0.5]]), {0: "pos", 1: "neg"})


def test_get_sentiment_empty_input_skips_model():
    assert get_sentiment([]) == []
    assert get_sentiment(["", "   ", None]) == []


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
