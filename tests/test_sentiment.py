import numpy as np
import pytest

from finpulse.sentiment import get_sentiment, probs_to_results

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
