# Joint Accept-Route Beam Design

## Scope

This teacher is implemented as a real `joint_accept_route_beam` method rather than an alias for the earlier beam-like teachers.

New code:

- `src/schedulers/joint_accept_route_beam.py`
- `src/schedulers/joint_beam_objective.py`
- `src/experiments/eval_acceptance_insertion.py`

No `ServicePolicy` training was run, and no imitation dataset was generated.

## Difference From Earlier Teachers

Earlier teachers such as `guarded_ontime_beam` and `policy_accept_ontime_beam` split the decision into an acceptance guard and a route beam. They did not search accept/reject decisions and route actions in one beam state.

`joint_accept_route_beam` keeps a beam state containing:

- current time and position
- pending orders
- accepted but unserved orders
- rejected, expired, and served orders
- route sequence
- response and delivery deadlines
- accepted orders, on-time orders, late orders
- total lateness, max lateness
- total energy, total distance
- hard constraint violations
- score and risk components

Each expansion can consider:

- `reject(order_id)`
- `accept_only(order_id)`
- `accept_and_serve(order_id)`
- `serve_accepted(order_id)`
- `wait` / depot return where feasible

Hard infeasible actions are masked through the environment masks before expansion.

## Objective

The score is lower-is-better:

```text
- accept_weight * accepted_orders
- on_time_weight * on_time_orders
+ late_weight * late_orders
+ lateness_weight * total_lateness
+ max_lateness_weight * max_lateness
+ energy_weight * total_energy
+ distance_weight * total_distance
+ hard_violation_weight * hard_constraint_violations
```

Default weights:

```text
accept=20
on_time=30
late=40
lateness=3
max_lateness=8
energy=0.08
distance=0.04
hard_violation=1000000
```

The implementation also adds an accepted-but-unserved lateness proxy so `accept_only` cannot collect acceptance reward while pushing lateness outside the beam horizon.

## Dominance Pruning

A state is pruned if another state is no worse on:

- accepted orders
- on-time orders
- late orders
- max lateness
- total lateness
- total distance
- total energy
- hard constraint violations

and at least one of those dimensions is strictly better.
