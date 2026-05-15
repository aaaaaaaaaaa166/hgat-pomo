# Tail Risk Constrained Joint Beam Next Steps

## Gate Outcome

The teacher passed eval `5`, `10`, and `20` with no regression versus raw baseline.

## Allowed Next

- Enter eval `30` and `50`.
- Generate an imitation dataset only from `tail_risk_constrained_joint_beam` in anchor-locked mode.
- Train `ServicePolicy` only after the dataset is produced from the passing teacher.

## Not Recommended

- Do not use `joint_accept_route_beam_guarded` as a teacher.
- Do not enable `--tail-risk-allow-anchor-deviation` for dataset generation until deviation-mode gates pass independently.

## Suggested Follow-Up Commands

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/tail_risk_constrained_joint_beam_gate_eval30 --methods raw_baseline,tail_risk_constrained_joint_beam --eval-instances 30 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/tail_risk_constrained_joint_beam_gate_eval50 --methods raw_baseline,tail_risk_constrained_joint_beam --eval-instances 50 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
```

