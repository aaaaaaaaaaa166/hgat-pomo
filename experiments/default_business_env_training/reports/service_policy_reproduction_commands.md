# ServicePolicy Reproduction Commands

## Environment

- Branch: `experiment/business-constraint-sensitivity-80-50`
- Environment profile: `default_business_env`
- Business constraints:
  - `response_window=5.0`
  - `delivery_window_extension=+3.0`
  - `resources=2`

## Build Severe8 Outcome-Augmented Dataset

This creates a local `.pt` dataset. Do not commit the dataset file.

```powershell
python -m src.experiments.build_imitation_dataset --env-profile default_business_env --teacher-method oracle_best_on_time --instances 50 --progress-every 10 --augment-outcome-labels true --relabel-risky-accepts true --safe-accept-lateness-threshold 8.0 --include-candidate-labels false --output-path experiments/default_business_env_training/imitation/oracle_best_on_time_augmented_severe8_dataset.pt --report-path experiments/default_business_env_training/reports/oracle_augmented_severe8_dataset_quality.md --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
```

## Train Assignment-Aware ServicePolicy

This creates local checkpoints. Do not commit `.pt` files.

```powershell
python -m src.experiments.train_service_policy --dataset-path experiments/default_business_env_training/imitation/oracle_best_on_time_augmented_severe8_dataset.pt --output-dir experiments/default_business_env_training/models/augmented_severe8_assignment_gate --epochs 5 --lr 0.0001 --hidden-dim 128 --heads 4 --encoder-layers 2 --k-nn-orders 8 --accept-loss-weight 0.3 --route-loss-weight 4.0 --assignment-loss-weight 2.0 --lateness-loss-weight 0.5 --on-time-loss-weight 1.0 --risky-accept-penalty 1.0 --risky-lateness-threshold 8.0 --pairwise-route-loss-weight 1.0 --score-loss-weight 0.0 --device cpu
```

## Head Diagnostics

```powershell
python -m src.experiments.diagnose_service_policy_heads --dataset-path experiments/default_business_env_training/imitation/oracle_best_on_time_augmented_severe8_dataset.pt --model-paths experiments/default_business_env_training/models/augmented_severe8_assignment_gate/service_policy_imitation_best.pt --model-names augmented_severe8_assignment_gate --metrics-csv experiments/default_business_env_training/metrics/augmented_severe8_assignment_head_metrics.csv --head-report-path experiments/default_business_env_training/reports/augmented_severe8_assignment_head_metrics.md --diagnosis-report-path experiments/default_business_env_training/reports/augmented_severe8_assignment_diagnosis.md --lateness-threshold 8.0 --device cpu
```

## Eval Gate

Run eval=20:

```powershell
python -m src.experiments.run_business_constraint_sensitivity_80_50 --output-dir experiments/default_business_env_training/evaluation/augmented_severe8_assignment_eval20 --env-profile default_business_env --service-model-path experiments/default_business_env_training/models/augmented_severe8_assignment_gate/service_policy_imitation_best.pt --eval-instances 20 --eval-progress-every 5 --methods raw_baseline,oracle_best_on_time,service_policy_imitation --decode-mode service_policy_lateness_guarded --lateness-risk-threshold 8.0 --max-predicted-lateness 20.0 --accept-risk-penalty 1.0 --on-time-priority-bonus 0.5 --service-device cpu
```

Run eval=50:

```powershell
python -m src.experiments.run_business_constraint_sensitivity_80_50 --output-dir experiments/default_business_env_training/evaluation/augmented_severe8_assignment_eval50 --env-profile default_business_env --service-model-path experiments/default_business_env_training/models/augmented_severe8_assignment_gate/service_policy_imitation_best.pt --eval-instances 50 --eval-progress-every 10 --methods raw_baseline,oracle_best_on_time,service_policy_imitation --decode-mode service_policy_lateness_guarded --lateness-risk-threshold 8.0 --max-predicted-lateness 20.0 --accept-risk-penalty 1.0 --on-time-priority-bonus 0.5 --service-device cpu
```

Run eval=100:

```powershell
python -m src.experiments.run_business_constraint_sensitivity_80_50 --output-dir experiments/default_business_env_training/evaluation/augmented_severe8_assignment_eval100 --env-profile default_business_env --service-model-path experiments/default_business_env_training/models/augmented_severe8_assignment_gate/service_policy_imitation_best.pt --eval-instances 100 --eval-progress-every 20 --methods raw_baseline,oracle_best_on_time,service_policy_imitation --decode-mode service_policy_lateness_guarded --lateness-risk-threshold 8.0 --max-predicted-lateness 20.0 --accept-risk-penalty 1.0 --on-time-priority-bonus 0.5 --service-device cpu
```

## Plot

The MATLAB plotter now draws both business-constraint oracle figures and ServicePolicy gate figures:

```matlab
run('experiments/business_constraint_sensitivity_80_50_stability/plot_business_constraint_comparison.m')
```

Expected additional figures:

- `fig5_service_policy_gate_rates.png`
- `fig6_service_policy_lateness.png`

## Confirmed Results

| eval | ServicePolicy acc | ServicePolicy on-time | hard |
|---:|---:|---:|---:|
| 20 | 0.818333 | 0.551935 | 0 |
| 50 | 0.806000 | 0.526055 | 0 |
| 100 | 0.803333 | 0.518257 | 0 |
