# Beam Teacher Small-Gate Report

## Scope

This report covers the second teacher-design attempt after the first one-step insertion teachers failed.

No model training was started. The frozen baseline was not modified:

`experiments/frozen_models_20260419/model_main_ep200.pt`

## New Teacher Variants

Added to `src/experiments/eval_acceptance_insertion.py`:

- `beam_oracle_insertion`
- `deadline_beam_oracle`
- `conservative_deadline_beam`
- `ontime_beam_oracle`
- `guarded_ontime_beam`
- `policy_accept_ontime_beam`

The strongest two-stage variants split the problem as follows:

- acceptance decision:
  - `guarded_ontime_beam`: explicit high-slack acceptance guard;
  - `policy_accept_ontime_beam`: baseline policy accepts/rejects, beam handles routing.
- service decision:
  - V2 beam search with stronger on-time and lateness penalties.

## Results

### eval_instances=5

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0 | true |
| policy_accept_ontime_beam | 0.520000 | 0.371795 | 49 | 14.988338 | 49.701626 | 20.432552 | 303.366171 | 0 | true |
| guarded_ontime_beam | 0.506667 | 0.368421 | 48 | 15.681360 | 40.043612 | 19.296540 | 284.696450 | 0 | true |

### eval_instances=10

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0 | true |
| policy_accept_ontime_beam | 0.543333 | 0.349693 | 106 | 13.911818 | 49.701626 | 39.067687 | 582.985901 | 0 | false |
| guarded_ontime_beam | 0.513333 | 0.331169 | 103 | 14.239666 | 40.043612 | 36.992627 | 551.284582 | 0 | false |

### eval_instances=20

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0 | true |
| policy_accept_ontime_beam | 0.528333 | 0.324921 | 214 | 16.155948 | 52.024166 | 84.906828 | 1269.581887 | 0 | false |
| guarded_ontime_beam | 0.508333 | 0.311475 | 210 | 16.069716 | 48.767459 | 80.705141 | 1209.690757 | 0 | false |

## Diagnosis

The beam variants are materially better than the first insertion-only teachers. On `eval_instances=5`, both `policy_accept_ontime_beam` and `guarded_ontime_beam` pass the small gate.

However, they do not remain stable at `eval_instances=10` and `20`:

- `policy_accept_ontime_beam` keeps acceptance closer to baseline, but increases max lateness, average lateness, energy, and distance.
- `guarded_ontime_beam` reduces late orders at 10/20 but loses too much acceptance and on-time rate.
- Both keep hard constraint violations at `0`.

## Gate Decision

Do not start ServicePolicy imitation training from these teachers yet.

The current best teacher passes `5` but fails `10/20`, so it does not satisfy the requested small-data gate.

## Next Useful Direction

The likely next improvement is a stronger acceptance planner that optimizes per-instance acceptance count and route impact jointly, instead of using a fixed slack threshold or one-step baseline accept decision.

Recommended next teacher design:

1. Generate a short candidate set of dynamic accepts during each response window.
2. Run beam search over accept/reject combinations and route actions together.
3. Score the full episode-level business metrics:
   - acceptance rate;
   - on-time rate;
   - late orders;
   - average/max lateness;
   - energy and distance;
   - hard constraints.

Until that teacher passes `5/10/20`, formal imitation training should remain blocked.

