# Joint Accept-Route Beam Gate

## Decision

`joint_accept_route_beam` did **not** pass the 5/10/20 small-data gate.

Do not run 30/50 validation, do not generate a formal imitation dataset, and do not train `ServicePolicy` from this teacher.

## Commands

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/joint_accept_route_beam_gate_eval5 --methods raw_baseline,v2_repair_only,policy_accept_ontime_beam,guarded_ontime_beam,joint_accept_route_beam --eval-instances 5 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/joint_accept_route_beam_gate_eval10 --methods raw_baseline,v2_repair_only,policy_accept_ontime_beam,guarded_ontime_beam,joint_accept_route_beam --eval-instances 10 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/joint_accept_route_beam_gate_eval20 --methods raw_baseline,v2_repair_only,policy_accept_ontime_beam,guarded_ontime_beam,joint_accept_route_beam --eval-instances 20 --eval-progress-every 5 --decision-mode accept_then_route --feature-mode legacy --N 30 --K 8 --eval-seed 0 --edge-mode road --time-dependent --energy-cost-weight 0.08 --drone-takeoff-landing-energy 0.01
```

## eval_instances=5

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | strict_gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0 | baseline |
| v2_repair_only | 0.526667 | 0.303797 | 55 | 13.665207 | 51.748751 | 20.956155 | 312.798919 | 0 | fail |
| policy_accept_ontime_beam | 0.520000 | 0.384615 | 48 | 17.458257 | 48.041712 | 21.301283 | 315.062065 | 0 | fail-strict |
| guarded_ontime_beam | 0.506667 | 0.342105 | 50 | 14.948441 | 44.011625 | 18.881094 | 279.866718 | 0 | fail |
| joint_accept_route_beam | 0.580000 | 0.149425 | 74 | 20.086043 | 65.033543 | 20.699020 | 337.696110 | 0 | fail |

## eval_instances=10

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | strict_gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0 | baseline |
| v2_repair_only | 0.513333 | 0.298701 | 108 | 12.329668 | 51.748751 | 38.200877 | 588.272216 | 0 | fail |
| policy_accept_ontime_beam | 0.533333 | 0.343750 | 105 | 15.609612 | 51.123086 | 38.804550 | 591.945369 | 0 | fail |
| guarded_ontime_beam | 0.506667 | 0.322368 | 103 | 15.862412 | 51.942725 | 37.526817 | 577.665728 | 0 | fail |
| joint_accept_route_beam | 0.576667 | 0.104046 | 155 | 18.194940 | 65.033543 | 39.183574 | 657.989428 | 0 | fail |

## eval_instances=20

| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | strict_gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0 | baseline |
| v2_repair_only | 0.506667 | 0.299342 | 213 | 13.121789 | 51.748751 | 76.269763 | 1155.775931 | 0 | fail |
| policy_accept_ontime_beam | 0.521667 | 0.319489 | 213 | 17.194818 | 51.123086 | 83.611342 | 1271.325650 | 0 | fail |
| guarded_ontime_beam | 0.506667 | 0.302632 | 212 | 19.146384 | 56.324121 | 81.430350 | 1240.533588 | 0 | fail |
| joint_accept_route_beam | 0.563333 | 0.150888 | 287 | 20.253662 | 96.393187 | 84.143669 | 1406.873292 | 0 | fail |

## Gate Result

The new teacher preserves hard constraints, but it fails the business gate at every scale:

- acceptance improves, but on-time rate drops sharply;
- late orders increase substantially;
- average and max lateness worsen;
- total distance worsens, especially at 20 instances;
- eval=20 clearly degrades, so medium validation is blocked.
