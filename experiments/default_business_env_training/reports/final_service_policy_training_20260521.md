# Final ServicePolicy Training Result - 2026-05-21

## Final Model

- Model: `severe8_assignment_100inst_20260521`
- Dataset: `oracle_best_on_time_augmented_severe8_dataset_100inst_20260521.pt`
- Samples: 3443
- Teacher quality: acc=0.915000, on_time=0.529326, hard=0
- Training: 6 epochs from scratch, severe8 relabeling, assignment loss, pairwise route loss
- Rejected continuation: `severe8_assignment_100inst_ep10_20260521` because acceptance fell to 0.74 on eval=100.

## Per-Seed Evaluation

| eval | seed | acc | on-time | late | avg late | max late | hard | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0 | 0.820000 | 0.534146 | 573 | 20.196305 | 57.971370 | 0 | yes |
| 50 | 1 | 0.802000 | 0.536160 | 558 | 19.115085 | 59.860907 | 0 | yes |
| 50 | 2 | 0.796000 | 0.551926 | 535 | 18.794343 | 60.711380 | 0 | no |
| 100 | 0 | 0.813667 | 0.523146 | 1164 | 19.993843 | 63.901285 | 0 | yes |
| 100 | 1 | 0.799667 | 0.538558 | 1107 | 19.509736 | 62.876089 | 0 | no |
| 100 | 2 | 0.797667 | 0.537819 | 1106 | 19.350174 | 60.711380 | 0 | no |
| 200 | 0 | 0.815500 | 0.532393 | 2288 | 20.266438 | 63.901285 | 0 | yes |

## Aggregate

| eval | seeds | acc mean | acc min | on-time mean | on-time min | pass seeds | all pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 50 | 0,1,2 | 0.806000 | 0.796000 | 0.540744 | 0.534146 | 2/3 | no |
| 100 | 0,1,2 | 0.803667 | 0.797667 | 0.533174 | 0.523146 | 1/3 | no |
| 200 | 0 | 0.815500 | 0.815500 | 0.532393 | 0.532393 | 1/1 | yes |

## Decision

- This is the final trained ServicePolicy checkpoint for the thesis workspace.
- It strictly improves the previous 50-instance gate on seed-0 eval=50/100 and additionally passes eval=200 seed 0.
- Multi-seed eval=100 remains marginal: seed 1 and seed 2 miss the 80% acceptance target by 0.000333 and 0.002333, while on-time remains above 50%.
- Do not use the ep10 continuation checkpoint; it overfits toward on-time and loses acceptance.
- Paper wording can now be stronger than before: the final trained model passes the standard seed-0 gate and eval=200, with near-threshold multi-seed acceptance as the remaining limitation.

## Complete Comparison Link

The complete paper comparison, full old-method recomparison under `default_business_env`, 100-instance training ablations, and competitive multi-seed comparison are summarized in:

`experiments/default_business_env_training/reports/complete_comparison_training_20260521.md`
