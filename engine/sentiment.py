"""FinBERT inference.

The model is loaded lazily on first use, so importing this module is cheap
and does not require torch (useful for tests and CI).
"""
from __future__ import annotations

import functools

import numpy as np

from engine.config import FINBERT_MODEL

BATCH_SIZE = 16


def score_from_probs(p_pos: float, p_neg: float) -> float:
    """Signed sentiment in [-1, 1]: P(positive) - P(negative)."""
    return float(p_pos) - float(p_neg)


def probs_to_results(probs: np.ndarray, id2label: dict[int, str]) -> list[dict]:
    """Turn an (n, 3) probability matrix into labelled result dicts.

    Label order is read from the model config instead of being assumed.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2-D array")
    names = {i: str(lbl).lower() for i, lbl in id2label.items()}
    missing = {"positive", "negative", "neutral"} - set(names.values())
    if missing:
        raise ValueError(f"Model labels missing: {missing}")
    idx = {name: i for i, name in names.items()}

    out = []
    for row in probs:
        best = int(row.argmax())
        out.append({
            "label": names[best],
            "confidence": float(row[best]),
            "p_positive": float(row[idx["positive"]]),
            "p_negative": float(row[idx["negative"]]),
            "p_neutral": float(row[idx["neutral"]]),
            "score": score_from_probs(row[idx["positive"]], row[idx["negative"]]),
        })
    return out


@functools.lru_cache(maxsize=1)
def load_model(name: str = FINBERT_MODEL):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tokenizer, model


def get_sentiment(texts: list[str], batch_size: int = BATCH_SIZE) -> list[dict]:
    """Score a list of headlines with FinBERT. Empty input returns []."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    import torch

    tokenizer, model = load_model()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            logits = model(**enc).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return probs_to_results(np.vstack(chunks), model.config.id2label)
