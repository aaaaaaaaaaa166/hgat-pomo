# ServicePolicy On-Time Sanity Gate

## Scope

- Environment: `default_business_env`
- Constraints: `response_window=5.0`, `delivery_window_extension=+3.0`, `resources=2`
- Teacher target: `oracle_best_on_time`
- Dataset: `experiments/default_business_env_training/imitation/oracle_best_on_time_dataset.pt`
- Rule: no RL fine-tune and no long training unless imitation-only reaches or closely approaches the small gate.

## Reference

Prior default-business eval at 20 instances:

| method | acc | on_time | late | avg_late | max_late | hard |
|---|---:|---:|---:|---:|---:|---:|
| raw_baseline | 0.641667 | 0.579221 | 162 | 8.290222 | 24.694935 | 0 |
| v2_repair_only | 0.606667 | 0.596154 | 147 | 8.445696 | 25.640610 | 0 |
| oracle_best_on_time | 0.908333 | 0.557798 | 241 | 22.258302 | 55.932933 | 0 |
| ServicePolicy previous raw decode | 0.951667 | 0.434326 | 323 | 23.155230 | 68.531271 | 0 |

## Head Diagnostics

The diagnostic data show an important supervision gap:

- Accept labels: `626` positive, `0` negative.
- Lateness labels: `1678` on-time, `0` late at threshold `1.0`.
- The accept head therefore learns an all-accept policy.
- The lateness head is not calibrated on real late/risky examples; zero lateness error in B/C/D is not evidence of real on-time control.
- Route top-1 remains around `0.80-0.82`, while top-3 is around `0.97`, so the model often finds the right neighborhood but does not reliably pick the teacher's exact next service order.

## Small Training Gate

All service-policy rows below use `service_policy_lateness_guarded` decoding.

| group | setting | eval | acc | on_time | late | avg_late | max_late | energy | distance | hard | reaches 80/50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_current | 5 epochs, accept=0.5, route=2.0, lateness=1.0 | 20 | 0.898333 | 0.460111 | 291 | 21.873191 | 54.862255 | 111.186165 | 1906.179037 | 0 | no |
| B_route_focused | 5 epochs, accept=0.2, route=3.0, lateness=2.0, on_time=2.0 | 20 | 0.941667 | 0.438938 | 317 | 22.960943 | 60.739584 | 118.258124 | 2003.569811 | 0 | no |
| C_lateness_focused | 5 epochs, accept=0.1, route=3.0, lateness=3.0, on_time=2.0, risky_penalty=2.0 | 20 | 0.948333 | 0.430580 | 324 | 25.194179 | 67.268131 | 122.744251 | 2119.884061 | 0 | no |
| D_pairwise | 5 epochs, accept=0.1, route=2.0, lateness=2.0, on_time=2.0, risky_penalty=2.0, pairwise=1.0 | 20 | 0.938333 | 0.456483 | 306 | 22.629398 | 59.327634 | 126.917877 | 2073.564862 | 0 | no |

## Decision

- eval=20 did not reach `acc>=0.80`, `on_time>=0.50`, `hard=0` for any A/B/C/D group.
- eval=50 was not run.
- RL fine-tune is not allowed yet.
- Continue with imitation and decoding fixes, but the immediate priority is to regenerate or augment the imitation dataset with reject examples, risky accepted examples, and late outcomes under model-like rollouts.

## Recommended Next Fix

1. Rebuild the dataset so it includes teacher rejects and non-priority accepted orders, not only accepted/on-time oracle steps.
2. Add rollout-derived labels: actual lateness after model-like decoding, not just teacher trajectory lateness.
3. Train route ranking on pairwise comparisons among feasible candidates at each step.
4. Keep lateness-guarded decoding, but make the guard depend on calibrated risk labels before using it as a gate for RL.
