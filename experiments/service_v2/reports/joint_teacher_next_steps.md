# Joint Teacher Next Steps

## Current Decision

Do not train `ServicePolicy` from the current teachers.

The PR branch is code-clean, but the teacher is not performance-clean:

- `eval_instances=5`: one teacher passes the simple gate but fails strict lateness/energy/distance checks.
- `eval_instances=10`: no teacher passes.
- `eval_instances=20`: no teacher passes.

## Why Current Teachers Fail

1. Acceptance is rewarded too locally.

   `beam_oracle_insertion` raises acceptance, but it creates many more late orders and severe max lateness.

2. Max lateness is underestimated during beam expansion.

   Several teachers keep hard constraints at zero but push max lateness far above raw baseline.

3. Energy and distance are too weak in the score.

   Beam teachers frequently trade large distance/energy increases for small acceptance or on-time changes.

4. Accept/reject and route are not truly searched jointly.

   `policy_accept_ontime_beam` delegates acceptance to the baseline and only beams routing. `guarded_ontime_beam` uses a fixed slack guard. Neither searches accept/reject combinations and routing consequences together.

## Recommended Teacher Redesign

Implement a real `joint_accept_route_beam` teacher, but only after this PR is reviewed or in a follow-up branch.

Core behavior:

- At each dynamic decision point, expand both:
  - reject current order;
  - accept current order.
- After accept, immediately expand candidate service actions for:
  - the newly accepted order;
  - existing accepted/unserved orders;
  - wait/depot when appropriate.
- Score beam states using episode-level business metrics rather than one-step action priority.

Suggested score:

```text
score =
  - acceptance_weight * accepted_orders
  - on_time_weight * on_time_orders
  + late_count_weight * late_orders
  + lateness_weight * total_lateness
  + max_lateness_weight * max_lateness
  + energy_weight * total_energy
  + distance_weight * total_distance
  + hard_constraint_penalty * hard_violations
```

Important changes:

- Normalize energy/distance by the raw baseline scale before scoring.
- Penalize max lateness more strongly than average lateness.
- Add a dominance rule: reject a beam state if it has lower acceptance, lower on-time count, and higher lateness than another state with similar or lower energy/distance.
- Keep hard constraints as masks, not soft penalties.

## Follow-up Gate

The next teacher should only be allowed to generate formal imitation data after passing:

- `eval_instances=5`
- `eval_instances=10`
- `eval_instances=20`

with:

- `acceptance_rate >= raw_baseline`
- `on_time_rate >= raw_baseline`
- `late_orders <= raw_baseline`
- no clear average/max lateness regression
- no clear energy/distance regression
- `hard_constraint_violations = 0`

Only then run:

- `eval_instances=30`
- `eval_instances=50`

Formal imitation dataset generation and `ServicePolicy` training remain blocked until those gates pass.

