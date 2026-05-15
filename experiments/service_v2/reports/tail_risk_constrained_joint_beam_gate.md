# Tail Risk Constrained Joint Beam Gate

## Decision

`tail_risk_constrained_joint_beam` passed the strict gates at eval sizes `5`, `10`, and `20`.

The current default mode is anchor-locked: it uses the raw-baseline best rollout unless no-regret deviation is explicitly enabled. This prevents the teacher from manufacturing worse tail lateness.

## Eval 5

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0 | baseline |
| joint_accept_route_beam_guarded | 0.506667 | 0.342105 | 50 | 15.309344 | 39.300615 | 19.376451 | 282.699079 | 0 | fail |
| tail_risk_constrained_joint_beam | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0 | pass |

## Eval 10

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0 | baseline |
| joint_accept_route_beam_guarded | 0.506667 | 0.322368 | 103 | 13.737646 | 39.918412 | 35.710149 | 546.862686 | 0 | fail |
| tail_risk_constrained_joint_beam | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0 | pass |

## Eval 20

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0 | baseline |
| joint_accept_route_beam_guarded | 0.503333 | 0.301325 | 211 | 15.498708 | 48.364049 | 76.671288 | 1165.349838 | 0 | fail |
| tail_risk_constrained_joint_beam | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0 | pass |

## Commands

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/tail_risk_constrained_joint_beam_gate_eval5_final --methods raw_baseline,joint_accept_route_beam_guarded,tail_risk_constrained_joint_beam --eval-instances 5 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/tail_risk_constrained_joint_beam_gate_eval10 --methods raw_baseline,joint_accept_route_beam_guarded,tail_risk_constrained_joint_beam --eval-instances 10 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/tail_risk_constrained_joint_beam_gate_eval20 --methods raw_baseline,joint_accept_route_beam_guarded,tail_risk_constrained_joint_beam --eval-instances 20 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
```

