# Smoke Augmented Training Gate

## Purpose

This smoke test checks whether outcome-augmented imitation labels can fix the all-accept behavior before any full dataset rebuild or longer training.

No RL fine-tune was run.

## Dataset

- dataset: `experiments/default_business_env_training/imitation/smoke_augmented_oracle_best_on_time_dataset.pt`
- instances: `5`
- samples: `184`
- outcome_labeled_samples: `181`
- outcome_late_samples: `97`
- accept_relabels: `31`

## Training

Command:

```powershell
python -m src.experiments.train_service_policy --dataset-path experiments/default_business_env_training/imitation/smoke_augmented_oracle_best_on_time_dataset.pt --output-dir experiments/default_business_env_training/models/smoke_augmented_risk_focused --epochs 3 --lr 0.0001 --hidden-dim 128 --heads 4 --encoder-layers 2 --k-nn-orders 8 --accept-loss-weight 0.5 --route-loss-weight 2.0 --lateness-loss-weight 2.0 --on-time-loss-weight 2.0 --risky-accept-penalty 2.0 --pairwise-route-loss-weight 1.0 --score-loss-weight 0.0 --device cpu
```

| epoch | loss | accept_loss | route_loss | lateness_loss | risky_accept_loss | pairwise_route_loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.525911 | 0.421270 | 1.544785 | 11.166091 | 0.193268 | 0.407780 |
| 2 | 22.397732 | 0.306623 | 1.171075 | 9.137996 | 0.196303 | 0.296375 |
| 3 | 18.622478 | 0.294460 | 0.917566 | 7.494776 | 0.185692 | 0.251306 |

## Head Metrics

| metric | value |
|---|---:|
| accept_positive_rate | 0.544118 |
| model_accept_rate | 0.632353 |
| accept_accuracy | 0.647059 |
| accept_precision | 0.651163 |
| accept_recall | 0.756757 |
| route_top1_accuracy | 0.646018 |
| route_top3_accuracy | 0.929204 |
| lateness_mae | 6.945858 |
| lateness_rmse | 13.139649 |
| lateness_risk_auc | 0.920103 |
| on_time_class_accuracy | 0.850829 |
| risky_order_false_accept_rate | 0.483871 |

## Eval=20

Decode mode: `service_policy_lateness_guarded`

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | reaches 80/50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.641667 | 0.579221 | 162 | 8.290222 | 24.694935 | 99.755360 | 1596.378716 | 0 | no |
| smoke_augmented_risk_focused | 0.795000 | 0.465409 | 255 | 16.572138 | 50.881670 | 90.647430 | 1597.858083 | 0 | no |

## Decision

- The augmented labels fixed the all-accept failure directionally: model_accept_rate fell from `1.0` to `0.632353`.
- Risk discrimination became meaningful on the smoke set: lateness risk AUC is `0.920103`.
- Eval acceptance improved toward the target but remained just below `0.80`.
- On-time rate is still below `0.50`; route top-1 dropped to `0.646018`, so route sequencing is now the bottleneck.
- This smoke model is not eligible for RL and not eligible for eval=50.

## Next Task

Build a full 50-instance augmented dataset without committing the `.pt`, then run one route-preserving small training gate. The objective is to recover route top-1 while keeping the improved accept/risk supervision.
