### Event-type tagger vs labels

Accuracy 0.750 [95% CI 0.670, 0.830], macro F1 0.583, n = 100. Pre-registered threshold 70%: met.

| Type | Labelled | Tagged | Precision | Recall |
|---|---:|---:|---:|---:|
| earnings | 26 | 27 | 0.93 | 0.96 |
| regulatory | 7 | 4 | 0.75 | 0.43 |
| rumour | 6 | 2 | 1.00 | 0.33 |
| management | 2 | 2 | 0.50 | 0.50 |
| macro | 10 | 9 | 0.22 | 0.20 |
| other | 49 | 56 | 0.75 | 0.86 |

### Sensitivity: 10 borderline labels swapped to their alternative

Accuracy 0.740 [95% CI 0.650, 0.830], macro F1 0.482, n = 100. Pre-registered threshold 70%: met.

| Type | Labelled | Tagged | Precision | Recall |
|---|---:|---:|---:|---:|
| earnings | 24 | 27 | 0.89 | 1.00 |
| regulatory | 7 | 4 | 0.75 | 0.43 |
| rumour | 2 | 2 | 0.00 | 0.00 |
| management | 3 | 2 | 0.50 | 0.33 |
| macro | 11 | 9 | 0.22 | 0.18 |
| other | 53 | 56 | 0.79 | 0.83 |

> Labels were drafted with an LLM and reviewed and confirmed by the project owner, so they carry one reviewer's judgement and may be anchored on the LLM draft. None of the evaluated models saw these headlines in training or tuning. Small n: read the confidence intervals, not just the point estimates.
