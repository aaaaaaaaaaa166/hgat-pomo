# Joint Accept-Route Beam Next Steps

## Current Decision

Stop at the small-data gate. The teacher is not good enough for imitation learning.

## Recommended Fixes

1. Add an acceptance safety test before rewarding `accept_only`.

   A dynamic order should not receive the full acceptance benefit unless the accepted backlog can be greedily served without increasing late orders or max lateness beyond a guard threshold.

2. Use a baseline-relative route guard.

   Keep a per-instance raw-baseline reference or fast local surrogate and reject candidate states whose predicted late orders, max lateness, distance, or energy exceed that reference by more than the gate tolerance.

3. Strengthen max-lateness dominance.

   The current dominance pruning allows states with more accepted orders to survive even when max lateness is much worse. Add epsilon dominance that treats large max-lateness regressions as non-negotiable.

4. Add route compaction after each accepted order.

   After an accept action, immediately re-rank the accepted backlog by deadline and insertion regret before scoring the beam state.

5. Calibrate distance and energy penalties by observed scale.

   Distance increased by 291.78 at eval=20. The default distance weight is too small relative to acceptance and on-time weights for this dataset scale.

## Next Experiment

Do not train. The next code-only experiment should be a guarded variant:

```text
joint_accept_route_beam_guarded
```

The guard should accept only if a fast deterministic backlog simulation satisfies:

- accepted_orders does not cause late_orders to increase;
- max_lateness does not increase beyond 5%;
- total_distance does not increase beyond 10%;
- total_energy does not increase beyond 10%;
- hard_constraint_violations stays 0.

Only after that variant passes 5/10/20 should 30/50 or imitation generation be reconsidered.
