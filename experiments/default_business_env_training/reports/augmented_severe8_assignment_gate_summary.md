# Augmented Severe8 Assignment Gate Summary

## Goal

Recover ServicePolicy on-time learning under `default_business_env` without RL and without long training by fixing the imitation supervision gap.

## What Changed

1. `build_imitation_dataset.py` now supports outcome-augmented labels.
2. Final order outcomes are written back into each sample.
3. Severe-risk relabeling uses `safe_accept_lateness_threshold=8.0` for this gate.
4. `train_service_policy.py` now trains the assignment head.
5. `diagnose_service_policy_heads.py` now reports assignment metrics.

## Dataset

- teacher: `oracle_best_on_time`
- instances: `50`
- augmented samples: `1733`
- outcome_labeled_samples: `1678`
- outcome_late_samples: `744`
- accept_relabels: `218`

## Model

Training command:

```powershell
python -m src.experiments.train_service_policy --dataset-path experiments/default_business_env_training/imitation/oracle_best_on_time_augmented_severe8_dataset.pt --output-dir experiments/default_business_env_training/models/augmented_severe8_assignment_gate --epochs 5 --lr 0.0001 --hidden-dim 128 --heads 4 --encoder-layers 2 --k-nn-orders 8 --accept-loss-weight 0.3 --route-loss-weight 4.0 --assignment-loss-weight 2.0 --lateness-loss-weight 0.5 --on-time-loss-weight 1.0 --risky-accept-penalty 1.0 --risky-lateness-threshold 8.0 --pairwise-route-loss-weight 1.0 --score-loss-weight 0.0 --device cpu
```

Head metrics:

| metric | value |
|---|---:|
| accept_positive_rate | 0.651757 |
| model_accept_rate | 0.686901 |
| accept_accuracy | 0.798722 |
| accept_precision | 0.827907 |
| accept_recall | 0.872549 |
| route_top1_accuracy | 0.829848 |
| assignment_accuracy | 0.802281 |
| assignment_drone_recall | 0.457249 |
| lateness_risk_auc | 0.935128 |
| on_time_class_accuracy | 0.906436 |
| risky_order_false_accept_rate | 0.339450 |

## Evaluation

### eval=20

| method | acc | on_time | late | avg_late | max_late | hard |
|---|---:|---:|---:|---:|---:|---:|
| raw_baseline | 0.641667 | 0.579221 | 162 | 8.290222 | 24.694935 | 0 |
| oracle_best_on_time | 0.908333 | 0.557798 | 241 | 22.258302 | 55.932933 | 0 |
| service_policy_imitation | 0.818333 | 0.551935 | 220 | 22.456276 | 59.262892 | 0 |

### eval=50

| method | acc | on_time | late | avg_late | max_late | hard |
|---|---:|---:|---:|---:|---:|---:|
| raw_baseline | 0.593333 | 0.544944 | 405 | 8.624550 | 25.731184 | 0 |
| oracle_best_on_time | 0.914667 | 0.518950 | 660 | 24.145694 | 66.110448 | 0 |
| service_policy_imitation | 0.806000 | 0.526055 | 573 | 23.875986 | 62.703487 | 0 |

### eval=100

| method | acc | on_time | late | avg_late | max_late | hard |
|---|---:|---:|---:|---:|---:|---:|
| raw_baseline | 0.603333 | 0.569061 | 780 | 8.943759 | 27.030407 | 0 |
| oracle_best_on_time | 0.911667 | 0.524314 | 1301 | 23.620672 | 66.221815 | 0 |
| service_policy_imitation | 0.803333 | 0.518257 | 1161 | 22.817444 | 65.605548 | 0 |

## Decision

- eval=20 passed strict 80/50.
- eval=50 passed strict 80/50.
- eval=100 passed strict 80/50.
- The model is now a valid imitation teacher candidate under `default_business_env`.
- RL fine-tune remains optional, not required for the 80/50 gate.

## Next Task

Compare this gate against the older sanity and smoke gates in one short paper-facing summary, then stop unless you want another refinement loop.
