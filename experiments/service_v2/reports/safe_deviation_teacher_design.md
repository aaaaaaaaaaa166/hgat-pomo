# Safe Deviation Teacher Design

## Motivation

The anchor-locked `tail_risk_constrained_joint_beam` is safe because it reproduces raw baseline metrics. That is useful as a guardrail, but it is not enough for a valuable teacher: a student trained on it would mostly learn baseline behavior.

## Safe Deviation Rule

The new safe-deviation mode keeps raw baseline as the default action source. It first tries a tail-risk candidate rollout, then accepts that rollout only if the final episode metrics are non-worse than raw baseline and at least one key metric improves.

Hard acceptance criteria:

- `hard_constraint_violations = 0`
- `acceptance_rate >= raw_baseline`
- `on_time_rate >= raw_baseline`
- `late_orders <= raw_baseline`
- `average_lateness <= raw_baseline`
- `max_lateness <= raw_baseline`
- `total_energy <= raw_baseline * 1.01`
- `total_distance <= raw_baseline * 1.01`

Nontrivial improvement is required by default. Improvements can be higher acceptance/on-time rate, fewer late orders, lower average/max lateness, lower energy, or lower distance.

## Interfaces

- Method alias: `tail_risk_constrained_joint_beam_safe_deviation`
- Flag: `--tail-risk-allow-safe-deviation`
- Default remains anchor-locked.

New controls:

- `--tail-risk-min-improvement`
- `--tail-risk-max-acceptance-drop`
- `--tail-risk-max-on-time-drop`
- `--tail-risk-max-avg-late-ratio`
- `--tail-risk-max-max-late-ratio`
- `--tail-risk-max-energy-ratio`
- `--tail-risk-max-distance-ratio`
- `--tail-risk-require-nontrivial-improvement`

