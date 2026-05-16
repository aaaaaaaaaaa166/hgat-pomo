# Business Constraint Stability Validation

## Scope

- No model training, ServicePolicy training, imitation dataset generation, baseline weight replacement, or joint-teacher tuning was performed.
- The validation is limited to combined_D, combined_E, combined_F, and combined_G.
- Methods: raw_baseline, v2_repair_only, tail_risk_constrained_joint_beam, oracle_best_acceptance, oracle_best_on_time.
- The tail_risk_constrained_joint_beam row is used as a safe baseline reference in this runner; it is anchor-locked to raw_baseline.
- Eval scales: 50, 100.

## Stability Matrix

| configuration | method | eval=50 | eval=100 | stable |
|---|---|---|---|---|
| combined_D | raw_baseline | fail | fail | no |
| combined_D | v2_repair_only | fail | fail | no |
| combined_D | tail_risk_constrained_joint_beam | fail | fail | no |
| combined_D | oracle_best_acceptance | fail | fail | no |
| combined_D | oracle_best_on_time | pass | pass | yes |
| combined_E | raw_baseline | fail | fail | no |
| combined_E | v2_repair_only | fail | fail | no |
| combined_E | tail_risk_constrained_joint_beam | fail | fail | no |
| combined_E | oracle_best_acceptance | pass | pass | yes |
| combined_E | oracle_best_on_time | pass | pass | yes |
| combined_F | raw_baseline | fail | fail | no |
| combined_F | v2_repair_only | fail | fail | no |
| combined_F | tail_risk_constrained_joint_beam | fail | fail | no |
| combined_F | oracle_best_acceptance | pass | pass | yes |
| combined_F | oracle_best_on_time | pass | pass | yes |
| combined_G | raw_baseline | fail | fail | no |
| combined_G | v2_repair_only | fail | fail | no |
| combined_G | tail_risk_constrained_joint_beam | fail | fail | no |
| combined_G | oracle_best_acceptance | pass | pass | yes |
| combined_G | oracle_best_on_time | pass | pass | yes |

## Recommended Stable Pair

- Recommended: `combined_D / oracle_best_on_time` (response=5.0, due+3.0, resources=2).

## Stable Oracle Pairs

- `combined_D / oracle_best_on_time`: response=5.0, due+3.0, resources=2, worst_avg_late=24.145694, worst_max_late=66.221815.
- `combined_F / oracle_best_acceptance`: response=8.0, due+5.0, resources=2, worst_avg_late=21.549208, worst_max_late=71.931582.
- `combined_F / oracle_best_on_time`: response=8.0, due+5.0, resources=2, worst_avg_late=24.699792, worst_max_late=71.806114.
- `combined_E / oracle_best_acceptance`: response=5.0, due+4.0, resources=3, worst_avg_late=14.975779, worst_max_late=50.832591.
- `combined_E / oracle_best_on_time`: response=5.0, due+4.0, resources=3, worst_avg_late=16.163347, worst_max_late=53.213413.
- `combined_G / oracle_best_acceptance`: response=8.0, due+5.0, resources=3, worst_avg_late=15.854506, worst_max_late=54.819866.
- `combined_G / oracle_best_on_time`: response=8.0, due+5.0, resources=3, worst_avg_late=17.683158, worst_max_late=52.522568.
