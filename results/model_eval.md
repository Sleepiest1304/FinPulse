| Subset | Model | n | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| allagree | majority (neutral) | 2264 | 0.614 | 0.254 | 0.468 |
| allagree | VADER | 2264 | 0.571 | 0.487 | 0.579 |
| allagree | FinBERT | 2264 | 0.972 | 0.963 | 0.972 |
| 50agree | majority (neutral) | 4846 | 0.594 | 0.248 | 0.443 |
| 50agree | VADER | 4846 | 0.543 | 0.489 | 0.549 |
| 50agree | FinBERT | 4846 | 0.890 | 0.883 | 0.891 |

> ProsusAI/finbert was fine-tuned on Financial PhraseBank, so its numbers are largely in-sample and overstate accuracy on unseen text.
