# Global Classification Metrics

All values use the same MIMIC-IV train/test split, five-client partition,
seed (`42`), 20 communication rounds, five local epochs, and a `0.5`
classification threshold.

| Configuration | Accuracy | F1 | Recall | Precision | AUROC |
|---|---:|---:|---:|---:|---:|
| FedAvg baseline | **0.8865** | 0.3722 | 0.3088 | **0.4684** | 0.7956 |
| MCFL personalization | 0.8684 | 0.3752 | 0.3627 | 0.3885 | 0.8025 |
| Full, lambda_ditto=0.01, mcfl_lr=0.10 | 0.8713 | 0.3975 | 0.3897 | 0.4056 | 0.8176 |
| Full, lambda_ditto=0.001, mcfl_lr=0.10 | 0.8654 | **0.4085** | **0.4265** | 0.3919 | **0.8252** |
| Full, lambda_ditto=0.10, mcfl_lr=0.10 | 0.8668 | 0.3966 | 0.4020 | 0.3914 | 0.8225 |
| Full, lambda_ditto=0.01, mcfl_lr=0.03 | 0.8694 | 0.4000 | 0.3995 | 0.4005 | 0.8209 |
| Full, lambda_ditto=0.01, mcfl_lr=0.30 | 0.8777 | 0.4067 | 0.3848 | 0.4313 | 0.8247 |

## Interpretation

Personalization improves AUROC, F1, and recall compared with FedAvg. It does
not improve raw accuracy or precision at the fixed `0.5` threshold. This is a
threshold trade-off on an imbalanced mortality outcome: higher recall catches
more deaths but also increases false positives. For the highest global AUROC,
F1, and recall, use `lambda_ditto=0.001` and `mcfl_lr=0.10`. For the strongest
fairness result, use `lambda_ditto=0.10` and `mcfl_lr=0.10`.
