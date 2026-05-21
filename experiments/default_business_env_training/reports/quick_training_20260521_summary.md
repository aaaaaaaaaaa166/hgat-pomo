# Quick Training And Stability Check - 2026-05-21

## Purpose

Time-limited check of whether a very short continuation training run can improve the current `augmented_severe8_assignment_gate` ServicePolicy, especially the narrow acceptance margin observed at eval=100/200.

## Short Training Attempts

Both attempts resumed from:

`experiments/default_business_env_training/models/augmented_severe8_assignment_gate/service_policy_imitation_best.pt`

| candidate | epochs | lr | accept loss | risky accept penalty | eval=100 acc | eval=100 on-time | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| quick_accept_calib_20260521 | 1 | 0.00005 | 0.45 | 0.8 | 0.787000 | 0.529860 | 0 | no |
| quick_accept_loose_20260521 | 1 | 0.00003 | 0.10 | 0.0 | 0.774000 | 0.546081 | 0 | no |

Both quick continuation runs reduced acceptance below the 80% target. They should not replace the current severe8 assignment-aware checkpoint.

## Current Best Checkpoint Multi-Seed Eval=100

Checkpoint:

`experiments/default_business_env_training/models/augmented_severe8_assignment_gate/service_policy_imitation_best.pt`

| eval seed | acc | on-time | late orders | avg late | max late | hard | pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.803333 | 0.518257 | 1161 | 22.817444 | 65.605548 | 0 | yes |
| 1 | 0.797333 | 0.530518 | 1123 | 22.571076 | 74.348933 | 0 | no |
| 2 | 0.788000 | 0.541455 | 1084 | 22.435019 | 67.991005 | 0 | no |

Aggregate over seeds 0/1/2:

| metric | value |
|---|---:|
| acceptance mean | 0.796222 |
| acceptance std | 0.006309 |
| minimum acceptance | 0.788000 |
| on-time mean | 0.530077 |
| on-time std | 0.009476 |
| minimum on-time | 0.518257 |

## Decision

- Do not use the quick continuation checkpoints.
- Keep `augmented_severe8_assignment_gate` as the best trained checkpoint for the single-seed gate.
- Do not claim multi-seed stable 80/50 at eval=100.
- The remaining gap is acceptance calibration, not on-time recovery.
- If more work is possible, the next experiment should target acceptance calibration with validation selection, not another blind continuation-training pass.
