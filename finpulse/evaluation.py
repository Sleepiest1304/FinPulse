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


# ------------------------------------------------------------ hand-labelled set
ZEROSHOT_MODEL = "facebook/bart-large-mnli"
ZEROSHOT_TEMPLATE = "The sentiment of this financial news headline is {}."
LABELLED_CAVEAT = ("Hand-labelled by one person, so labels carry that person's judgement. "
                   "No model saw these headlines in training or tuning. Small n: read the "
                   "confidence intervals, not just the point estimates.")


def zeroshot_labels(texts: list[str], model: str = ZEROSHOT_MODEL, batch_size: int = 16) -> list[str]:
    """Zero-shot NLI classification into negative / neutral / positive (needs torch)."""
    if not texts:
        return []
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=model, device=-1)
    out = clf(list(texts), candidate_labels=list(LABELS),
              hypothesis_template=ZEROSHOT_TEMPLATE, batch_size=batch_size)
    out = [out] if isinstance(out, dict) else out
    return [r["labels"][0] for r in out]


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 2000, seed: int = 2026,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of ``metric`` over headlines."""
    import numpy as np

    rng = np.random.default_rng(seed)
    y_true, y_pred = list(y_true), list(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def load_labelled(path: Path, rebuild=None) -> tuple[list[str], list[str], int]:
    """Read ``record_id,label[,title]``. Missing titles are rebuilt from GDELT.

    Returns texts, labels and the number of rows dropped (bad label or no title).
    """
    import pandas as pd

    df = pd.read_csv(path, dtype=str)
    if "record_id" not in df or "label" not in df:
        raise ValueError("labelled file needs record_id and label columns")
    df["label"] = df["label"].str.strip().str.lower()
    if "title" not in df:
        df["title"] = None
    missing = df["title"].isna()
    if missing.any():
        if rebuild is None:
            from finpulse.gdelt import rebuild_titles as rebuild
        found = rebuild(df.loc[missing, "record_id"].tolist())
        df.loc[missing, "title"] = df.loc[missing, "record_id"].map(found)
    ok = df["label"].isin(LABELS) & df["title"].notna()
    return df.loc[ok, "title"].tolist(), df.loc[ok, "label"].tolist(), int((~ok).sum())


def evaluate_labelled(texts, y, models=("majority", "vader", "finbert", "zeroshot"),
                      predictors=None) -> list[dict]:
    """Compare models on a hand-labelled set, with bootstrap CIs.

    ``predictors`` maps model key -> callable(texts) -> labels, for tests.
    """
    predictors = dict(predictors or {})
    if "vader" in models and "vader" not in predictors:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        predictors["vader"] = lambda ts: [vader_label(sia.polarity_scores(t)["compound"]) for t in ts]
    if "finbert" in models and "finbert" not in predictors:
        from finpulse.sentiment import get_sentiment
        predictors["finbert"] = lambda ts: [r["label"] for r in get_sentiment(ts)]
    if "zeroshot" in models and "zeroshot" not in predictors:
        predictors["zeroshot"] = zeroshot_labels
    names = {"vader": "VADER", "finbert": "FinBERT", "zeroshot": f"Zero-shot ({ZEROSHOT_MODEL})"}

    rows = []
    majority = Counter(y).most_common(1)[0][0]
    for key in models:
        if key == "majority":
            name, preds = f"majority ({majority})", [majority] * len(y)
        else:
            name, preds = names.get(key, key), predictors[key](texts)
        row = {"subset": "indian_hand_labelled", **metrics(name, y, preds)}
        row["accuracy_ci"] = bootstrap_ci(y, preds, accuracy)
        row["macro_f1_ci"] = bootstrap_ci(y, preds, macro_f1)
        rows.append(row)
    return rows


def to_markdown_ci(rows: list[dict]) -> str:
    lines = ["| Model | n | Accuracy [95% CI] | Macro F1 [95% CI] | Weighted F1 |",
             "|---|---:|---|---|---:|"]
    for r in rows:
        a, f = r["accuracy_ci"], r["macro_f1_ci"]
        lines.append(f"| {r['model']} | {r['n']} | {r['accuracy']:.3f} [{a[0]:.3f}, {a[1]:.3f}] | "
                     f"{r['macro_f1']:.3f} [{f[0]:.3f}, {f[1]:.3f}] | {r['weighted_f1']:.3f} |")
    return "\n".join(lines)
