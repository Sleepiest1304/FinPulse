| Model | n | Accuracy [95% CI] | Macro F1 [95% CI] | Weighted F1 |
|---|---:|---|---|---:|
| majority (neutral) | 150 | 0.487 [0.407, 0.567] | 0.218 [0.193, 0.241] | 0.319 |
| VADER | 150 | 0.487 [0.407, 0.573] | 0.489 [0.403, 0.571] | 0.490 |
| FinBERT | 150 | 0.700 [0.620, 0.773] | 0.698 [0.618, 0.766] | 0.703 |
| Zero-shot (facebook/bart-large-mnli) | 150 | 0.500 [0.420, 0.580] | 0.456 [0.398, 0.507] | 0.355 |

n = 150 ({'positive': 40, 'neutral': 73, 'negative': 37}), 0 rows dropped.

### Sensitivity: 12 borderline labels swapped to their alternative

| Model | n | Accuracy [95% CI] | Macro F1 [95% CI] | Weighted F1 |
|---|---:|---|---|---:|
| majority (neutral) | 150 | 0.433 [0.353, 0.520] | 0.202 [0.174, 0.228] | 0.262 |
| VADER | 150 | 0.500 [0.420, 0.580] | 0.502 [0.420, 0.578] | 0.497 |
| FinBERT | 150 | 0.680 [0.600, 0.753] | 0.683 [0.601, 0.751] | 0.680 |
| Zero-shot (facebook/bart-large-mnli) | 150 | 0.553 [0.473, 0.633] | 0.485 [0.433, 0.535] | 0.414 |

Model ranking changes: accuracy

> Labels were drafted with an LLM and reviewed and confirmed by the project owner, so they carry one reviewer's judgement and may be anchored on the LLM draft. None of the evaluated models saw these headlines in training or tuning. Small n: read the confidence intervals, not just the point estimates.
