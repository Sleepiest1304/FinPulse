"""Classification metrics and dataset loading for the model comparison."""
from __future__ import annotations

import zipfile
from collections import Counter
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

DATASET = "Financial PhraseBank v1.0 (Malo et al., 2014), CC BY-NC-SA 3.0"
CAVEAT = ("ProsusAI/finbert was fine-tuned on Financial PhraseBank, so its numbers "
          "are largely in-sample and overstate accuracy on unseen text.")


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


def metrics(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    return {"model": name, "n": len(y_true),
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "weighted_f1": weighted_f1(y_true, y_pred)}


def evaluate(subsets=("allagree", "50agree"), models=("majority", "vader", "finbert"),
             loader=None) -> list[dict]:
    """Score each model on each Financial PhraseBank subset.

    VADER needs ``vaderSentiment`` (requirements-dev.txt); FinBERT needs torch.
    ``loader`` defaults to ``load_phrasebank`` and is injectable for tests.
    """
    loader = loader or load_phrasebank
    if "vader" in models:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    if "finbert" in models:
        from finpulse.sentiment import get_sentiment

    rows = []
    for subset in subsets:
        texts, y = loader(subset)
        print(f"{subset}: {len(texts)} sentences {dict(Counter(y))}")
        majority = Counter(y).most_common(1)[0][0]
        if "majority" in models:
            rows.append({"subset": subset,
                         **metrics(f"majority ({majority})", y, [majority] * len(y))})
        if "vader" in models:
            preds = [vader_label(vader.polarity_scores(t)["compound"]) for t in texts]
            rows.append({"subset": subset, **metrics("VADER", y, preds)})
        if "finbert" in models:
            preds = [r["label"] for r in get_sentiment(texts)]
            rows.append({"subset": subset, **metrics("FinBERT", y, preds)})
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Subset | Model | n | Accuracy | Macro F1 | Weighted F1 |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['subset']} | {r['model']} | {r['n']} | {r['accuracy']:.3f} | "
              f"{r['macro_f1']:.3f} | {r['weighted_f1']:.3f} |" for r in rows]
    return "\n".join(lines)
