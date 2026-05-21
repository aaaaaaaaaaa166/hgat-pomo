# Default Business Environment Training Reports

This directory summarizes the trained ServicePolicy work under `default_business_env`.

## Main Result

Use these files first:

- `complete_comparison_training_20260521.md`
- `final_service_policy_training_20260521.md`
- `thesis_experiment_completion_20260521.md`
- `service_policy_multiseed_gate_20260521.md`
- `quick_training_20260521_summary.md`
- `augmented_severe8_assignment_gate_summary.md`
- `service_policy_default_env_progress_summary.md`
- `paper_service_policy_results_table.md`
- `paper_service_policy_results_table.tex`
- `service_policy_reproduction_commands.md`

Confirmed ServicePolicy gate:

| Eval | Acceptance | On-time | Hard violations |
|---:|---:|---:|---:|
| 50 | 0.820000 | 0.534146 | 0 |
| 100 | 0.813667 | 0.523146 | 0 |
| 200 | 0.815500 | 0.532393 | 0 |

Multi-seed follow-up:

| Eval | Seeds | Acceptance mean | Acceptance min | On-time mean | Pass seeds |
|---:|---|---:|---:|---:|---:|
| 50 | 0,1,2 | 0.806000 | 0.796000 | 0.540744 | 2/3 |
| 100 | 0,1,2 | 0.803667 | 0.797667 | 0.533174 | 1/3 |

Interpretation: the single-seed gate is confirmed, but acceptance is not robustly above 80% across eval seeds. Do not claim multi-seed stable 80/50.

## Key Reports

- `service_policy_on_time_failure_diagnosis.md`: why the first ServicePolicy sanity models failed.
- `service_policy_head_metrics.md`: head-level metrics for early models.
- `service_policy_sanity_gate.md`: A/B/C/D small training gate before augmentation.
- `augmented_imitation_dataset_smoke.md`: smoke test proving outcome-augmented labels fixed the missing-risk signal.
- `smoke_augmented_training_gate.md`: first augmented smoke training result.
- `augmented_severe8_assignment_gate_summary.md`: final passing gate summary.
- `service_policy_eval200_and_threshold_notes.md`: eval=200 probe and severe10 threshold check.
- `service_policy_multiseed_gate_20260521.md`: eval=50/100 multi-seed stability check.
- `quick_training_20260521_summary.md`: short continuation-training attempts and rejection.
- `final_service_policy_training_20260521.md`: final 100-instance trained model result.
- `complete_comparison_training_20260521.md`: final paper comparison, ablation training, and competitive multi-seed summary.
- `thesis_experiment_completion_20260521.md`: paper-facing final evidence status and safe wording.
- `service_policy_default_env_progress_summary.md`: paper-facing narrative summary.
- `service_policy_reproduction_commands.md`: exact commands for dataset, training, diagnostics, eval, and plotting.

Complete old-method recomparison under `default_business_env`:

- `../evaluation/complete_old_methods_default_env_20260521/full_old_methods_default_env_eval50_20260521.csv`
- `../evaluation/complete_old_methods_default_env_20260521/full_old_methods_default_env_eval100_20260521.csv`
- `../evaluation/complete_old_methods_default_env_20260521/full_old_methods_default_env_eval50_100_20260521.csv`

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
