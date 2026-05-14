# Joint Teacher Gate Rerun

## Important Note

The current PR branch does not contain a method literally named `joint_accept_route_beam`.

Rerun therefore used the implemented teacher variants in `src/experiments/eval_acceptance_insertion.py`:

- `beam_oracle_insertion`
- `deadline_beam_oracle`
- `conservative_deadline_beam`
- `ontime_beam_oracle`
- `guarded_ontime_beam`
- `policy_accept_ontime_beam`

These are beam/joint-like teachers, but none is a full joint accept-route beam over accept/reject combinations and route actions.

## Gate Criteria

A teacher must satisfy all of:

- `acceptance_rate >= raw_baseline`
- `on_time_rate >= raw_baseline`
- `late_orders <= raw_baseline`
- `hard_constraint_violations = 0`
- average/max lateness should not clearly worsen
- energy/distance should not clearly worsen

## eval_instances=5

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | strict_gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0 | baseline |
| v2_repair_only | 0.526667 | 0.303797 | 55 | 13.665207 | 51.748751 | 20.956155 | 312.798919 | 0 | fail |
| beam_oracle_insertion | 0.600000 | 0.400000 | 54 | 31.333198 | 81.969071 | 25.553820 | 406.388820 | 0 | fail |
| deadline_beam_oracle | 0.533333 | 0.350000 | 52 | 17.466975 | 49.489309 | 22.025979 | 319.798685 | 0 | fail |
| conservative_deadline_beam | 0.540000 | 0.370370 | 51 | 19.844368 | 56.700752 | 20.937891 | 318.002896 | 0 | fail |
| ontime_beam_oracle | 0.540000 | 0.358025 | 52 | 18.618621 | 55.005030 | 20.765554 | 313.064268 | 0 | fail |
| guarded_ontime_beam | 0.506667 | 0.342105 | 50 | 14.954062 | 38.115986 | 19.051686 | 278.266665 | 0 | fail |
| policy_accept_ontime_beam | 0.526667 | 0.379747 | 49 | 17.130856 | 48.041712 | 21.419397 | 316.733385 | 0 | fail-strict |

`policy_accept_ontime_beam` satisfies the simple script gate, but fails the stricter PR gate because average/max lateness, energy, and distance worsen substantially.

## eval_instances=10

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | strict_gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0 | baseline |
| v2_repair_only | 0.513333 | 0.298701 | 108 | 12.329668 | 51.748751 | 38.200877 | 588.272216 | 0 | fail |
| beam_oracle_insertion | 0.620000 | 0.338710 | 123 | 32.145680 | 131.566870 | 49.779217 | 804.912659 | 0 | fail |
| deadline_beam_oracle | 0.533333 | 0.337500 | 106 | 16.055863 | 49.489309 | 36.483663 | 561.404249 | 0 | fail |
| conservative_deadline_beam | 0.533333 | 0.343750 | 105 | 16.882466 | 50.984605 | 36.563038 | 565.366259 | 0 | fail |
| ontime_beam_oracle | 0.530000 | 0.358491 | 102 | 16.154914 | 50.375124 | 37.298836 | 572.921256 | 0 | fail |
| guarded_ontime_beam | 0.506667 | 0.328947 | 102 | 15.794932 | 51.942725 | 37.163737 | 567.206023 | 0 | fail |
| policy_accept_ontime_beam | 0.533333 | 0.368750 | 101 | 16.732739 | 48.041712 | 40.230705 | 606.913543 | 0 | fail |

No teacher passes at 10 instances.

## eval_instances=20

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | strict_gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0 | baseline |
| v2_repair_only | 0.506667 | 0.299342 | 213 | 13.121789 | 51.748751 | 76.269763 | 1155.775931 | 0 | fail |
| beam_oracle_insertion | 0.581667 | 0.329513 | 234 | 31.091093 | 131.566870 | 103.351520 | 1700.250261 | 0 | fail |
| deadline_beam_oracle | 0.538333 | 0.315789 | 221 | 19.655685 | 80.846565 | 85.534080 | 1302.227587 | 0 | fail |
| conservative_deadline_beam | 0.540000 | 0.314815 | 222 | 19.766451 | 65.013330 | 83.522227 | 1270.214668 | 0 | fail |
| ontime_beam_oracle | 0.548333 | 0.319149 | 224 | 19.913732 | 73.984805 | 86.013043 | 1303.923544 | 0 | fail |
| guarded_ontime_beam | 0.511667 | 0.299674 | 215 | 16.391783 | 49.952632 | 78.987881 | 1193.820340 | 0 | fail |
| policy_accept_ontime_beam | 0.521667 | 0.313099 | 215 | 17.079856 | 51.261140 | 83.731022 | 1270.166623 | 0 | fail |

No teacher passes at 20 instances.

## Decision

- Do not run 30/50 medium validation.
- Do not generate a formal imitation dataset.
- Do not train `ServicePolicy`.

## Failure Pattern

- High-acceptance beam variants over-reward acceptance and underestimate sequence-level max lateness.
- On-time-oriented beam variants reduce late orders in some cases but lose acceptance and on-time rate versus raw baseline.
- Energy and distance terms are too weak relative to acceptance/on-time terms, causing longer routes and higher max lateness.
- Current teachers do not jointly optimize accept/reject and routing over the whole response-window horizon.

