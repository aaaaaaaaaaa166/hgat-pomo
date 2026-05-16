# ServicePolicy On-Time Failure Diagnosis

## Dataset balance

- samples: `1733`
- accept positives / negatives: `626` / `0`
- accept positive rate: `1.000000`
- route samples: `1052`
- on-time / late samples: `1678` / `0`
- late sample rate: `0.000000`

## Findings

- Accept head is strong/overactive when model_accept_rate is high. Current first model rate: `1.000000` against teacher positive rate `1.000000`.
- Route top-1 accuracy is `0.815589` and top-3 accuracy is `0.970532`; poor top-1 indicates the model is not reproducing teacher service order.
- Lateness risk MAE/RMSE are `9.234415` / `17.233464`; high error means the risk head is not calibrated enough for decoding.
- On-time class accuracy is `0.581645` and risk AUC is `n/a`.
- Predicted-late-but-accepted count is `106` and accepted-late count is `0`.
- Accept labels contain no reject examples, so accept accuracy/precision/recall are inflated and the accept head has no supervised signal for refusing risky orders.
- Lateness labels contain no late examples at the chosen threshold, so zero lateness error or perfect on-time classification is not evidence of calibrated real-world lateness risk.

## Diagnosis

- The main failure is the combination of incomplete negative/risk supervision and imperfect route sequencing, not total inability to imitate.
- The current dataset teaches the model to accept nearly everything, while giving little direct signal about which accepted orders become late under model-driven routing.
- The model tends to preserve high acceptance while missing the oracle's on-time ordering logic.
- Decoding can use lateness risk as a guard, but the guard is weak when the risk head is trained only on non-late labels.
- RL fine-tune remains blocked until imitation-only reaches the small gate.
