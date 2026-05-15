# Recommendation For 80/50

- Best acceptance observed: `response_window_10.0 / oracle_best_acceptance` (response=10.0, due+0.0, resources=1, density=1.0, acc=0.999, on_time=0.267)
- Best on-time observed: `combined_G / v2_repair_only` (response=8.0, due+5.0, resources=3, density=1.0, acc=0.621, on_time=0.746)
- Closest joint target gap: `combined_D / oracle_best_on_time` (response=5.0, due+3.0, resources=2, density=1.0, acc=0.912, on_time=0.533)
- Any configuration reached both targets: `True`

## Interpretation

- These runs do not train or alter model weights.
- Resource counts above 1 use a compatible order-partition simulation because the original environment exposes one truck/drone resource.
- Under original constraints, the 0.80 acceptance / 0.50 on-time target is not reachable in these eval=30 runs.
- Response-window relaxation alone can reach the acceptance target from `3.0`, but on-time rate remains far below `0.50`.
- Delivery-window relaxation alone can reach the on-time target from `+5.0`, but acceptance rate remains far below `0.80`.
- More resources alone can reach the on-time target from `3` resources, but acceptance rate remains far below `0.80`.
- The smallest tested dual-target combination is `combined_D`: response window `5.0`, delivery window extension `+3.0`, resources `2`, with `oracle_best_on_time` reaching acc `0.912` and on_time `0.533`.
- Further model training is not the recommended next lever until business constraints are relaxed; the dominant blockers are response time, delivery time, and parallel capacity.

## Configurations reaching both targets

- `combined_D / oracle_best_on_time` (response=5.0, due+3.0, resources=2, density=1.0, acc=0.912, on_time=0.533)
- `combined_E / oracle_best_acceptance` (response=5.0, due+4.0, resources=3, density=1.0, acc=0.916, on_time=0.606)
- `combined_E / oracle_best_on_time` (response=5.0, due+4.0, resources=3, density=1.0, acc=0.887, on_time=0.652)
- `combined_F / oracle_best_acceptance` (response=8.0, due+5.0, resources=2, density=1.0, acc=0.994, on_time=0.564)
- `combined_F / oracle_best_on_time` (response=8.0, due+5.0, resources=2, density=1.0, acc=0.987, on_time=0.569)
- `combined_G / oracle_best_acceptance` (response=8.0, due+5.0, resources=3, density=1.0, acc=0.980, on_time=0.615)
- `combined_G / oracle_best_on_time` (response=8.0, due+5.0, resources=3, density=1.0, acc=0.971, on_time=0.640)
