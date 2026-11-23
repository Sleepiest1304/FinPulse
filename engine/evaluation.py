"""Classification metrics and dataset loading for the model comparison."""
from __future__ import annotations

import zipfile
from pathlib import Path

LABELS = ("negative", "neutral", "positive")

PHRASEBANK_REPO = "takala/financial_phrasebank"
PHRASEBANK_FILE = "data/FinancialPhraseBank-v1.0.zip"
PHRASEBANK_SUBSETS = {
    "allagree": "Sentences_AllAgree.txt",
    "75agree": "Sentences_75Agree.txt",
    "66agree": "Sentences_66Agree.txt",
    "50agree": "Sentences_50Agree.txt",
}


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equal length")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class_f1(y_true: list[str], y_pred: list[str], labels=LABELS) -> dict[str, float]:
    out = {}
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def macro_f1(y_true, y_pred, labels=LABELS) -> float:
    f1 = per_class_f1(y_true, y_pred, labels)
    return sum(f1.values()) / len(f1)


def weighted_f1(y_true, y_pred, labels=LABELS) -> float:
    f1 = per_class_f1(y_true, y_pred, labels)
    support = {c: sum(t == c for t in y_true) for c in labels}
    total = sum(support.values())
    return sum(f1[c] * support[c] for c in labels) / total if total else 0.0


def vader_label(compound: float, threshold: float = 0.05) -> str:
    """Standard VADER thresholds from the VADER authors."""
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"


def parse_phrasebank(lines: list[str]) -> tuple[list[str], list[str]]:
    """Each line is ``sentence@label``; the sentence itself may contain '@'."""
    texts, labels = [], []
    for line in lines:
        line = line.strip()
        if "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label in LABELS and text.strip():
            texts.append(text.strip())
            labels.append(label)
    return texts, labels


def load_phrasebank(subset: str = "allagree", cache_dir: Path | None = None):
    """Download (once) and parse a Financial PhraseBank subset."""
    if subset not in PHRASEBANK_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(PHRASEBANK_SUBSETS)}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(PHRASEBANK_REPO, PHRASEBANK_FILE, repo_type="dataset",
                           cache_dir=cache_dir)
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(PHRASEBANK_SUBSETS[subset]))
        raw = zf.read(name).decode("latin-1")
    return parse_phrasebank(raw.splitlines())
