# ServicePolicy On-Time Failure Diagnosis

## Dataset balance

- samples: `1733`
- accept positives / negatives: `408` / `218`
- accept positive rate: `0.651757`
- route samples: `1052`
- on-time / late samples: `934` / `744`
- late sample rate: `0.443385`

## Findings

- Accept head is strong/overactive when model_accept_rate is high. Current first model rate: `0.686901` against teacher positive rate `0.651757`.
- Route top-1 accuracy is `0.829848` and top-3 accuracy is `0.974335`; poor top-1 indicates the model is not reproducing teacher service order.
- Assignment accuracy is `0.802281` with model drone assignment rate `0.125475` against teacher drone-positive rate `0.255703`.
- Lateness risk MAE/RMSE are `2.935264` / `6.693469`; high error means the risk head is not calibrated enough for decoding.
- On-time class accuracy is `0.906436` and risk AUC is `0.9351277025626856`.
- Predicted-late-but-accepted count is `0` and accepted-late count is `74`.

## Diagnosis

- The main failure is the combination of incomplete negative/risk supervision and imperfect route sequencing, not total inability to imitate.
- The current dataset teaches the model to accept nearly everything, while giving little direct signal about which accepted orders become late under model-driven routing.
- The model tends to preserve high acceptance while missing the oracle's on-time ordering logic.
- Decoding can use lateness risk as a guard, but the guard is weak when the risk head is trained only on non-late labels.
- RL fine-tune remains blocked until imitation-only reaches the small gate.
