# Imitation Quality

- teacher_method: `oracle_best_on_time`
- instances: `50`
- samples: `1733`

| metric | value |
|---|---:|
| acceptance_rate | 0.917333 |
| on_time_rate | 0.528343 |
| late_orders | 649 |
| average_lateness | 24.071630 |
| max_lateness | 68.142227 |
| hard_constraint_violations | 0 |
| soft_time_window_violations | 649 |
| outcome_labeled_samples | 1678 |
| outcome_late_samples | 685 |
| accept_relabels | 191 |
| teacher_reject_labels | 0 |

This dataset should not be used for long training unless the teacher passes the small-data gate.
