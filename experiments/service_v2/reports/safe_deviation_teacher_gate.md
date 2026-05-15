# Safe Deviation Teacher Gate

## Decision

The safe-deviation teacher is safe but not valuable yet.

It passed the safety part of the 5/10/20 checks by falling back to raw baseline, but it found zero accepted safe deviations:

- eval 5: `safe_deviation_rate = 0.0`
- eval 10: `safe_deviation_rate = 0.0`
- eval 20: `safe_deviation_rate = 0.0`

Therefore it does not pass the value gate, should not enter 30/50, should not generate an imitation dataset, and should not be used to train `ServicePolicy`.

## Eval 5

| method | acc | on_time | late | avg_late | max_late | energy | distance | safe_dev_rate | value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0.000000 | baseline |
| tail_risk_constrained_joint_beam | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0.000000 | safety only |
| tail_risk_constrained_joint_beam_safe_deviation | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0.000000 | fail |
| joint_accept_route_beam_guarded | 0.506667 | 0.368421 | 48 | 14.617627 | 39.300615 | 19.284999 | 282.883406 | 0.000000 | fail |

## Eval 10

| method | acc | on_time | late | avg_late | max_late | energy | distance | safe_dev_rate | value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0.000000 | baseline |
| tail_risk_constrained_joint_beam | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0.000000 | safety only |
| tail_risk_constrained_joint_beam_safe_deviation | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0.000000 | fail |
| joint_accept_route_beam_guarded | 0.506667 | 0.322368 | 103 | 13.737646 | 39.918412 | 35.710149 | 546.862686 | 0.000000 | fail |

## Eval 20

| method | acc | on_time | late | avg_late | max_late | energy | distance | safe_dev_rate | value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0.000000 | baseline |
| tail_risk_constrained_joint_beam | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0.000000 | safety only |
| tail_risk_constrained_joint_beam_safe_deviation | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0.000000 | fail |
| joint_accept_route_beam_guarded | 0.503333 | 0.298013 | 212 | 16.008799 | 49.885210 | 77.487682 | 1183.589142 | 0.000000 | fail |

## Commands

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/safe_deviation_teacher_gate_eval5 --methods raw_baseline,tail_risk_constrained_joint_beam,tail_risk_constrained_joint_beam_safe_deviation,joint_accept_route_beam_guarded --eval-instances 5 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/safe_deviation_teacher_gate_eval10 --methods raw_baseline,tail_risk_constrained_joint_beam,tail_risk_constrained_joint_beam_safe_deviation,joint_accept_route_beam_guarded --eval-instances 10 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/safe_deviation_teacher_gate_eval20 --methods raw_baseline,tail_risk_constrained_joint_beam,tail_risk_constrained_joint_beam_safe_deviation,joint_accept_route_beam_guarded --eval-instances 20 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
```

