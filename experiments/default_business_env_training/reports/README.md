# Default Business Environment Training Reports

This directory summarizes the trained ServicePolicy work under `default_business_env`.

## Main Result

Use these files first:

- `augmented_severe8_assignment_gate_summary.md`
- `service_policy_default_env_progress_summary.md`
- `paper_service_policy_results_table.md`
- `paper_service_policy_results_table.tex`
- `service_policy_reproduction_commands.md`

Confirmed ServicePolicy gate:

| Eval | Acceptance | On-time | Hard violations |
|---:|---:|---:|---:|
| 20 | 0.818333 | 0.551935 | 0 |
| 50 | 0.806000 | 0.526055 | 0 |
| 100 | 0.803333 | 0.518257 | 0 |

## Key Reports

- `service_policy_on_time_failure_diagnosis.md`: why the first ServicePolicy sanity models failed.
- `service_policy_head_metrics.md`: head-level metrics for early models.
- `service_policy_sanity_gate.md`: A/B/C/D small training gate before augmentation.
- `augmented_imitation_dataset_smoke.md`: smoke test proving outcome-augmented labels fixed the missing-risk signal.
- `smoke_augmented_training_gate.md`: first augmented smoke training result.
- `augmented_severe8_assignment_gate_summary.md`: final passing gate summary.
- `service_policy_eval200_and_threshold_notes.md`: eval=200 probe and severe10 threshold check.
- `service_policy_default_env_progress_summary.md`: paper-facing narrative summary.
- `service_policy_reproduction_commands.md`: exact commands for dataset, training, diagnostics, eval, and plotting.

## Tables

- `paper_service_policy_results_table.md`
- `paper_service_policy_results_table.tex`
- `../metrics/augmented_severe8_assignment_gate_summary.csv`
- `../metrics/service_policy_default_env_progress_summary.csv`
- `../metrics/service_policy_eval200_and_threshold_notes.csv`

## Local Artifacts Not Committed

The following are intentionally local only:

- `../imitation/*.pt`
- `../models/`
- `../evaluation/`

They are ignored by `.gitignore` and should not be committed.
