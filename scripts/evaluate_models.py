"""Compare FinBERT with VADER and a majority-class baseline on Financial PhraseBank.

Usage:
    python scripts/evaluate_models.py [--subsets allagree 50agree]

Needs `vaderSentiment` (see requirements-dev.txt). Writes
results/model_eval.json and results/model_eval.md.

Caveat: ProsusAI/finbert was fine-tuned on Financial PhraseBank, so its
scores here are largely in-sample and overstate real-world accuracy on
unseen headlines. VADER and the majority baseline have not seen the data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import FINBERT_MODEL, RESULTS_DIR  # noqa: E402
from engine.evaluation import (accuracy, load_phrasebank, macro_f1,  # noqa: E402
                               vader_label, weighted_f1)

CAVEAT = ("ProsusAI/finbert was fine-tuned on Financial PhraseBank, so its numbers "
          "are largely in-sample and overstate accuracy on unseen text.")


def metrics(name, y_true, y_pred) -> dict:
    return {"model": name, "n": len(y_true),
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "weighted_f1": weighted_f1(y_true, y_pred)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subsets", nargs="+", default=["allagree", "50agree"])
    p.add_argument("--models", nargs="+", default=["majority", "vader", "finbert"],
                   choices=["majority", "vader", "finbert"])
    p.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)

    if "vader" in args.models:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    if "finbert" in args.models:
        from engine.sentiment import get_sentiment

    rows = []
    for subset in args.subsets:
        texts, y = load_phrasebank(subset)
        print(f"{subset}: {len(texts)} sentences {dict(Counter(y))}")
        majority = Counter(y).most_common(1)[0][0]
        if "majority" in args.models:
            rows.append({"subset": subset, **metrics(f"majority ({majority})", y,
                                                     [majority] * len(y))})
        if "vader" in args.models:
            rows.append({"subset": subset, **metrics("VADER", y, [
                vader_label(vader.polarity_scores(t)["compound"]) for t in texts])})
        if "finbert" in args.models:
            rows.append({"subset": subset, **metrics("FinBERT", y,
                         [r["label"] for r in get_sentiment(texts)])})

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dataset": "Financial PhraseBank v1.0 (Malo et al., 2014), CC BY-NC-SA 3.0",
           "finbert_model": FINBERT_MODEL, "caveat": CAVEAT, "results": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_eval.json").write_text(json.dumps(out, indent=2))

    md = ["| Subset | Model | n | Accuracy | Macro F1 | Weighted F1 |",
          "|---|---|---:|---:|---:|---:|"]
    md += [f"| {r['subset']} | {r['model']} | {r['n']} | {r['accuracy']:.3f} | "
           f"{r['macro_f1']:.3f} | {r['weighted_f1']:.3f} |" for r in rows]
    table = "\n".join(md)
    (args.out / "model_eval.md").write_text(table + f"\n\n> {CAVEAT}\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
