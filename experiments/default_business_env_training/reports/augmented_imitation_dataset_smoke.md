# Augmented Imitation Dataset Smoke

## Goal

Fix the supervision gap found in the first ServicePolicy gate:

- original imitation accept labels were all positive;
- original lateness labels did not reflect final episode outcomes for acceptance states;
- the model therefore learned high acceptance without calibrated on-time risk.

## Change

`build_imitation_dataset.py` now supports optional outcome augmentation:

- `--augment-outcome-labels true` writes final episode order outcomes back into each sample.
- `--relabel-risky-accepts true` converts accepted orders with final lateness above the threshold into risky reject supervision.
- `--safe-accept-lateness-threshold` controls that threshold.
- `--include-candidate-labels true` stores per-state candidate scores and predicted lateness for route/risk analysis.

The default behavior is unchanged unless these flags are enabled.

## Smoke Dataset

Command used:

```powershell
python -m src.experiments.build_imitation_dataset --env-profile default_business_env --teacher-method oracle_best_on_time --instances 5 --progress-every 1 --augment-outcome-labels true --relabel-risky-accepts true --safe-accept-lateness-threshold 1.0 --include-candidate-labels true --candidate-label-limit 12 --output-path experiments/default_business_env_training/imitation/smoke_augmented_oracle_best_on_time_dataset.pt --report-path experiments/default_business_env_training/reports/smoke_augmented_oracle_imitation_dataset_quality.md --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
```

| metric | value |
|---|---:|
| samples | 184 |
| teacher acceptance_rate | 0.953333 |
| teacher on_time_rate | 0.517483 |
| hard_constraint_violations | 0 |
| outcome_labeled_samples | 181 |
| outcome_late_samples | 97 |
| accept_relabels | 31 |
| teacher_reject_labels | 0 |

## Diagnostics On Augmented Labels

| model | accept_positive_rate | model_accept_rate | accept_acc | route_top1 | route_top3 | late_mae | late_rmse | late_auc | risky_false_accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 0.544118 | 1.000000 | 0.544118 | 0.796460 | 0.964602 | 3.646196 | 9.288375 | 0.947717 | 1.000000 |
| D_pairwise | 0.544118 | 1.000000 | 0.544118 | 0.814159 | 0.973451 | 12.273272 | 20.057657 | 0.500000 | 1.000000 |

## Interpretation

The augmented smoke dataset now exposes the failure clearly:

- accept supervision is no longer all-positive;
- late labels are present;
- existing ServicePolicy checkpoints still accept every risky order;
- route top-3 is high but route top-1 is still imperfect;
- D_pairwise collapses risk calibration on this augmented label set.

## Next Task

Run a tiny augmented-imitation training smoke test, still without RL or long training. The expected improvement is lower model_accept_rate and lower risky false accept rate; reaching 80/50 is not required at smoke scale.
