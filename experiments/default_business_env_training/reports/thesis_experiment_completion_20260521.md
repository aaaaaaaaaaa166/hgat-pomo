# Thesis Experiment Completion Summary - 2026-05-21

## Final Evidence Status

The experiments are now sufficient for a conservative thesis-level claim, but not for a strong robustness claim.

## Current Best Trained Model

- Model: `severe8_assignment_100inst_20260521`
- Checkpoint: `experiments/default_business_env_training/models/severe8_assignment_100inst_20260521/service_policy_imitation_best.pt`
- Dataset: `oracle_best_on_time_augmented_severe8_dataset_100inst_20260521.pt`
- Training recipe: outcome-augmented labels, severe-risk relabeling at lateness > 8, assignment loss for `(k,j)` decisions.

## Single-Seed Gate Result

The best checkpoint passes the original paper gate under eval seed 0:

| eval | acc | on-time | hard | status |
|---:|---:|---:|---:|---|
| 50 | 0.820000 | 0.534146 | 0 | pass |
| 100 | 0.813667 | 0.523146 | 0 | pass |
| 200 | 0.815500 | 0.532393 | 0 | pass |

This supports the claim that the adjusted `default_business_env` admits a trained ServicePolicy that reaches 80/50 without RL fine-tuning.

## Multi-Seed Stability Result

Additional eval=50 and eval=100 runs were added on 2026-05-21:

| eval | seeds | acc mean | acc min | on-time mean | on-time min | pass seeds |
|---:|---|---:|---:|---:|---:|---:|
| 50 | 0,1,2 | 0.806000 | 0.796000 | 0.540744 | 0.534146 | 2/3 |
| 100 | 0,1,2 | 0.803667 | 0.797667 | 0.533174 | 0.523146 | 1/3 |

Interpretation: on-time performance is stable above 50%, while acceptance is close to but not robustly above 80%.

## Eval=200 Boundary

The final 100-instance model passes the eval=200 seed-0 probe:

| model | decode | acc | on-time | hard | status |
|---|---|---:|---:|---:|---|
| severe8_assignment_100inst_20260521 | guarded severe8 | 0.815500 | 0.532393 | 0 | pass |

The older 50-instance model missed eval=200 by about 0.2 percentage points in acceptance; the 100-instance retraining closes that boundary for seed 0.

## Quick Continuation Training Check

Several continuation-training attempts were run and rejected:

| candidate | eval=100 acc | eval=100 on-time | decision |
|---|---:|---:|---|
| quick_accept_calib_20260521 | 0.787000 | 0.529860 | reject |
| quick_accept_loose_20260521 | 0.774000 | 0.546081 | reject |
| severe8_assignment_100inst_ep10_20260521 | 0.741667 mean | 0.589886 mean | reject |

These runs improved on-time behavior but hurt acceptance. They should not replace the 100-instance 6-epoch checkpoint.

## Baseline Evidence Closure

The May 1 long-run service-baseline report has been closed conservatively:

- Rainbow-DQN seed0 has a 20000-episode result.
- AM-REINFORCE and POMO-REINFORCE 20000-episode service artifacts were not found in the current workspace.
- Therefore, do not claim that all external service baselines were fully retrained to 20000 episodes.

Use the Rainbow-DQN long-run result only as a limited check against the "under-trained baseline" concern.

## Thesis-Safe Claims

Safe:

1. The strict original business setting does not reliably admit the 80/50 target under the tested policies.
2. Under `default_business_env` (`response_window=5.0`, `delivery_window_extension=+3.0`, `resources=2`), the oracle confirms feasibility.
3. Outcome-augmented imitation with assignment supervision can train a ServicePolicy that reaches 80/50 on the seed-0 eval=50/100/200 gate.
4. Further blind continuation training is not effective; the remaining issue is acceptance calibration.

Not safe:

1. Do not claim full multi-seed robust 80/50 success.
2. Do not claim eval=200 passes.
3. Do not claim all external RL baselines completed 20000-episode service training.

## Recommended Thesis Wording

> In the adjusted default business environment, the proposed outcome-augmented ServicePolicy trained on 100 oracle instances reaches the 80% acceptance and 50% on-time targets on the eval=50/100/200 gate under the primary evaluation seed, with no hard constraint violations. Additional multi-seed probes show that on-time performance remains above target, while acceptance is close to the 80% threshold and is the main remaining stability bottleneck. Thus, the trained policy demonstrates learnability of the feasible oracle behavior, while robust multi-seed 80/50 guarantees remain future work focused on acceptance calibration.
