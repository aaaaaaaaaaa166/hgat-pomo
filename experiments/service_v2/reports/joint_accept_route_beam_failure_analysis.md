# Joint Accept-Route Beam Failure Analysis

## Summary

The new `joint_accept_route_beam` is a true joint accept-route beam, but the current objective is still not a viable teacher.

It improves acceptance rate on 5/10/20, but does so by accepting too many orders that later become late. This violates the gate and makes it unsafe for imitation learning.

## Failure Type

Primary failure:

- acceptance is still too strong relative to sequence-level service capacity;
- max lateness is underestimated inside the beam horizon;
- route quality is not strong enough after accepting extra dynamic orders;
- distance/energy penalties are too weak to prevent long detours.

Not observed:

- hard constraint violations. Hard violations remained 0 in all 5/10/20 runs.

## Evidence

At `eval_instances=20`:

- raw baseline: acc 0.533333, on_time 0.328125, late 215, max_late 35.936546, distance 1115.095666
- joint beam: acc 0.563333, on_time 0.150888, late 287, max_late 96.393187, distance 1406.873292

The teacher gains 3.0 percentage points acceptance but loses 17.7 percentage points on-time rate and adds 72 late orders.

## Interpretation

The beam is now structurally correct, but the environment is highly dynamic and response windows are short. A short lookahead beam can accept a feasible-looking order, then discover later that the accepted backlog forces multiple deadline misses.

The accepted-but-unserved risk proxy reduced one obvious loophole but was not sufficient. The route sequence still lacks a strong global constraint that says: do not accept an order unless the resulting accepted backlog can be served with no worse lateness than raw baseline.

## Blocked Actions

- Do not run 30/50 validation.
- Do not generate `joint_teacher_dataset.pt`.
- Do not train `ServicePolicy`.
- Do not replace baseline weights.
