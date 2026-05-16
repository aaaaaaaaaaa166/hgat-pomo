# Service Policy Head Metrics

- dataset: `experiments/default_business_env_training/imitation/oracle_best_on_time_augmented_severe8_dataset.pt`
- lateness_threshold: `8.0`

| model | accept_acc | accept_prec | accept_rec | model_accept_rate | route_top1 | route_top3 | assign_acc | model_drone_rate | late_mae | late_rmse | late_auc | on_time_cls_acc | risky_false_accept | teacher_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| augmented_severe8_assignment_gate | 0.798722 | 0.827907 | 0.872549 | 0.686901 | 0.829848 | 0.974335 | 0.802281 | 0.125475 | 2.935264 | 6.693469 | 0.9351277025626856 | 0.906436 | 0.339450 | 0.818236 |
