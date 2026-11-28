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


ZEROSHOT_MODEL = "facebook/bart-large-mnli"
ZEROSHOT_TEMPLATE = "The sentiment of this financial news headline is {}."
MODEL_NAMES = {"vader": "VADER", "finbert": "FinBERT", "zeroshot": f"Zero-shot ({ZEROSHOT_MODEL})"}


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


def predict(texts, models, predictors=None) -> dict[str, list[str]]:
    """Labels from each model in ``models`` except the majority baseline.

    VADER needs ``vaderSentiment`` (requirements-dev.txt); FinBERT and the
    zero-shot model need torch. ``predictors`` (key -> callable) is for tests.
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
    return {k: list(predictors[k](texts)) for k in models if k != "majority"}


def _named(key: str, y, predictions) -> tuple[str, list[str]]:
    if key == "majority":
        majority = Counter(y).most_common(1)[0][0]
        return f"majority ({majority})", [majority] * len(y)
    return MODEL_NAMES.get(key, key), predictions[key]


def evaluate(subsets=("allagree", "50agree"), models=("majority", "vader", "finbert"),
             loader=None, predictors=None) -> list[dict]:
    """Score each model on each Financial PhraseBank subset.

    ``loader`` defaults to ``load_phrasebank`` and is injectable for tests.
    """
    loader = loader or load_phrasebank
    rows = []
    for subset in subsets:
        texts, y = loader(subset)
        print(f"{subset}: {len(texts)} sentences {dict(Counter(y))}")
        preds = predict(texts, models, predictors)
        for key in models:
            name, p = _named(key, y, preds)
            rows.append({"subset": subset, **metrics(name, y, p)})
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Subset | Model | n | Accuracy | Macro F1 | Weighted F1 |",
             "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['subset']} | {r['model']} | {r['n']} | {r['accuracy']:.3f} | "
              f"{r['macro_f1']:.3f} | {r['weighted_f1']:.3f} |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------ labelled set (LLM-drafted, author-reviewed)
LABELLED_CAVEAT = ("Labels were drafted with an LLM and reviewed and confirmed by the "
                   "project owner, so they carry one reviewer's judgement and may be anchored on "
                   "the LLM draft. None of the evaluated models saw these headlines in training "
                   "or tuning. Small n: read the confidence intervals, not just the point estimates.")


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


def load_labelled(path: Path):
    """Read a ``record_id,title,label`` CSV (kept outside the repo).

    Returns the usable rows and the number dropped (unknown label or no title).
    """
    import pandas as pd

    df = pd.read_csv(path, dtype=str)
    missing = {"record_id", "title", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"labelled file is missing columns: {sorted(missing)}")
    df["label"] = df["label"].str.strip().str.lower()
    ok = df["label"].isin(LABELS) & df["title"].notna()
    return df.loc[ok, ["record_id", "title", "label"]].reset_index(drop=True), int((~ok).sum())


def apply_alternatives(df, audit, sheet: str, keys: list[str], label_col: str,
                       audit_keys: list[str] | None = None):
    """Swap in the ``plausible_alternative`` label for borderline items.

    ``audit`` has columns sheet, <audit_keys>, label_used, plausible_alternative.
    Every audit row for ``sheet`` must match exactly one row of ``df`` whose
    current label equals ``label_used``; otherwise this raises, so a stale
    audit file cannot silently change the wrong rows. Returns a copy of ``df``
    and the number of labels swapped.
    """
    audit_keys = audit_keys or keys
    rows = audit[audit["sheet"] == sheet]
    out = df.copy()
    for _, r in rows.iterrows():
        want = [str(r[k]) for k in audit_keys]
        hit = (out[keys].astype(str) == want).all(axis=1)
        if hit.sum() != 1:
            raise ValueError(f"audit row {want} matches {int(hit.sum())} rows")
        current = out.loc[hit, label_col].item()
        if current != r["label_used"]:
            raise ValueError(f"audit row {want}: label is {current!r}, audit says {r['label_used']!r}")
        out.loc[hit, label_col] = r["plausible_alternative"]
    return out, len(rows)


def score_labelled(y, models, predictions) -> list[dict]:
    """Metrics with bootstrap CIs; ``majority`` predicts the most common label in ``y``."""
    rows = []
    for key in models:
        name, preds = _named(key, y, predictions)
        row = {"subset": "indian_labelled", "key": key, **metrics(name, y, preds)}
        row["accuracy_ci"] = bootstrap_ci(y, preds, accuracy)
        row["macro_f1_ci"] = bootstrap_ci(y, preds, macro_f1)
        rows.append(row)
    return rows


def ranking(rows: list[dict], metric: str) -> list[str]:
    """Model keys from best to worst on ``metric`` (ties keep input order)."""
    return [r["key"] for r in sorted(rows, key=lambda r: -r[metric])]


def compare_rankings(rows: list[dict], alt_rows: list[dict]) -> dict:
    """Whether swapping labels changes the model order on accuracy or macro F1."""
    out = {}
    for m in ("accuracy", "macro_f1"):
        a, b = ranking(rows, m), ranking(alt_rows, m)
        out[m] = {"original": a, "alternative": b, "changed": a != b}
    out["changed"] = out["accuracy"]["changed"] or out["macro_f1"]["changed"]
    return out


def to_markdown_ci(rows: list[dict]) -> str:
    lines = ["| Model | n | Accuracy [95% CI] | Macro F1 [95% CI] | Weighted F1 |",
             "|---|---:|---|---|---:|"]
    for r in rows:
        a, f = r["accuracy_ci"], r["macro_f1_ci"]
        lines.append(f"| {r['model']} | {r['n']} | {r['accuracy']:.3f} [{a[0]:.3f}, {a[1]:.3f}] | "
                     f"{r['macro_f1']:.3f} [{f[0]:.3f}, {f[1]:.3f}] | {r['weighted_f1']:.3f} |")
    return "\n".join(lines)


# ------------------------------------------------------------ event-type tagger
TAGGER_THRESHOLD = 0.70   # docs/v3-preregistration.md, section 2


def precision_recall(y_true, y_pred, labels) -> dict[str, dict]:
    """Per-class precision, recall and counts. None where the class never occurs."""
    out = {}
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        n_pred = sum(p == c for p in y_pred)
        n_true = sum(t == c for t in y_true)
        out[c] = {"precision": tp / n_pred if n_pred else None,
                  "recall": tp / n_true if n_true else None,
                  "support": n_true, "predicted": n_pred}
    return out


def confusion(y_true, y_pred, labels) -> dict[str, dict[str, int]]:
    """Counts as {true label: {predicted label: n}}."""
    return {t: {p: sum(a == t and b == p for a, b in zip(y_true, y_pred)) for p in labels}
            for t in labels}


def evaluate_tagger(df, label_col: str = "event_type_hand") -> dict:
    """Compare ``events.tag_headline`` with the labels (LLM-drafted, author-reviewed).

    ``df`` needs title, multi_ticker and ``label_col``. Every label must be one
    of ``events.TYPE_ORDER``; anything else raises instead of being mapped.
    """
    from finpulse.events import TYPE_ORDER, tag_headline

    y = df[label_col].astype(str).str.strip().str.lower().tolist()
    unknown = sorted(set(y) - set(TYPE_ORDER))
    if unknown:
        raise ValueError(f"labels not in the tagger's categories {TYPE_ORDER}: {unknown}")
    multi = df["multi_ticker"].astype(str).str.strip().str.lower().isin(["true", "1"]).tolist()
    pred = [tag_headline(t, m) for t, m in zip(df["title"], multi)]
    acc = accuracy(y, pred)
    return {"n": len(y), "accuracy": acc, "accuracy_ci": bootstrap_ci(y, pred, accuracy),
            "macro_f1": macro_f1(y, pred, TYPE_ORDER),
            "threshold": TAGGER_THRESHOLD, "meets_threshold": acc >= TAGGER_THRESHOLD,
            "per_class": precision_recall(y, pred, TYPE_ORDER),
            "confusion": confusion(y, pred, TYPE_ORDER)}


def tagger_markdown(res: dict) -> str:
    def fmt(x):
        return "n/a" if x is None else f"{x:.2f}"
    a = res["accuracy_ci"]
    lines = [f"Accuracy {res['accuracy']:.3f} [95% CI {a[0]:.3f}, {a[1]:.3f}], macro F1 "
             f"{res['macro_f1']:.3f}, n = {res['n']}. Pre-registered threshold "
             f"{res['threshold']:.0%}: {'met' if res['meets_threshold'] else 'not met'}.", "",
             "| Type | Labelled | Tagged | Precision | Recall |", "|---|---:|---:|---:|---:|"]
    for c, v in res["per_class"].items():
        lines.append(f"| {c} | {v['support']} | {v['predicted']} | {fmt(v['precision'])} | "
                     f"{fmt(v['recall'])} |")
    return "\n".join(lines)
