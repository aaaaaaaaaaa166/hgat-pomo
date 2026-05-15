# Safe Deviation Teacher Next Steps

## Current Decision

Stop at eval 20.

Do not run 30/50, do not generate an imitation dataset, and do not train `ServicePolicy`.

## Why

The teacher is safe but has no learning value yet:

- `safe_deviation_rate = 0.0` at eval 5, 10, and 20
- no accepted candidate improves any core metric
- all selected outputs are raw-baseline fallbacks

## Recommended Next Work

Improve the candidate generator before trying dataset generation:

- use raw-baseline route as the primary route skeleton;
- search only small local swaps or service-pair substitutions;
- require candidate rollouts to improve one metric while preserving all tail-risk budgets;
- add per-instance Pareto candidate logging to identify where safe improvements might exist;
- keep `joint_accept_route_beam_guarded` out of teacher generation until its tail-risk regressions are fixed.

## Not Allowed Yet

- 30/50 escalation
- formal imitation dataset
- `ServicePolicy` training

