# Service Policy Head Metrics

- dataset: `experiments/default_business_env_training/imitation/oracle_best_on_time_dataset.pt`
- lateness_threshold: `1.0`

| model | accept_acc | accept_prec | accept_rec | model_accept_rate | route_top1 | route_top3 | late_mae | late_rmse | late_auc | on_time_cls_acc | risky_false_accept | teacher_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.815589 | 0.970532 | 9.234415 | 17.233464 | n/a | 0.581645 | 0.000000 | 0.884386 |
| B_route_focused | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.801331 | 0.975285 | 0.000000 | 0.000000 | n/a | 1.000000 | 0.000000 | 0.875447 |
| C_lateness_focused | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.815589 | 0.970532 | 0.000000 | 0.000000 | n/a | 1.000000 | 0.000000 | 0.884386 |
| D_pairwise | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.816540 | 0.974335 | 0.000000 | 0.000000 | n/a | 1.000000 | 0.000000 | 0.884982 |
