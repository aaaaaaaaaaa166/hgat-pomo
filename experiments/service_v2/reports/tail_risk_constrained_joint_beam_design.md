# Tail Risk Constrained Joint Beam Design

## Goal

Create a baseline-anchored joint teacher that does not trade a few severe late orders for a better-looking on-time rate.

## What Failed Before

`joint_accept_route_beam_guarded` improved on-time rate on the 5-instance gate, but it still worsened tail risk:

- average lateness increased from `12.203287` to `14.617627`
- max lateness increased from `31.124878` to `39.300615`
- energy and distance also increased

That means it is not a safe teacher.

## New Teacher

`tail_risk_constrained_joint_beam`

## Design

- Build a raw-baseline budget for each eval instance.
- Enforce caps on:
  - max lateness
  - average lateness
  - total energy
  - total distance
- Use severe lateness as a hard guard.
- Keep the beam objective lexicographic:
  - hard constraints
  - tail risk
  - service quality
  - acceptance
  - cost
- Default eval mode is baseline-anchored, so the teacher falls back to the raw best rollout unless a no-regret deviation is explicitly enabled.

## Implementation Notes

- `src/schedulers/joint_accept_route_beam.py`
- `src/schedulers/joint_beam_objective.py`
- `src/experiments/eval_acceptance_insertion.py`

