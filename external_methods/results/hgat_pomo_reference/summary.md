# HGAT-POMO Reference Summary

## Frozen models
- Main model: `experiments/frozen_models_20260419/model_main_ep200.pt`
- Backup model: `experiments/frozen_models_20260419/model_backup_ep140.pt`

## Open-data training
- Logged epochs: 200
- Best `cost_best` epoch: ep0031 (12.159)
- Best `cost_mean` epoch: ep0020 (73.810)
- Final epoch: ep0200 (cost_mean=211.435, cost_best=127.681)
- Checkpoints: 10

## Open split coverage
- Eligible instances: 67
- Families: A, B, P
- Split sizes: train=47, val=10, test=10

## Formal protocol
- Total runs: 90
- Seeds: 0, 1, 2
- Scenarios: full_main, full_backup, ablate_no_accept_reject, ablate_no_pickup_capacity, ablate_no_time_traffic

### Full Main cost ranking
| Rank | Method | cost_mean_mean | cost_best_mean | cost_worst_mean |
| --- | --- | ---: | ---: | ---: |
| 1 | HGAT-POMO | 145.255 | 118.779 | 177.633 |
| 2 | Truck Only | 194.817 | 156.133 | 247.044 |
| 3 | Truck Local Search | 231.389 | 181.191 | 299.401 |
| 4 | Profit Accept | 336.965 | 298.278 | 377.871 |
| 5 | Heuristic | 360.777 | 301.514 | 460.800 |
| 6 | Random | 415.485 | 290.367 | 593.977 |

### Full Main on-time ranking
| Rank | Method | on_time_rate_mean | accept_rate_mean | total_energy_mean | total_revenue_mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | HGAT-POMO | 0.3033 | 0.0211 | 3.654 | 15.875 |
| 2 | Truck Only | 0.2450 | 0.0547 | 2.239 | 16.466 |
| 3 | Truck Local Search | 0.1992 | 0.0771 | 2.528 | 17.009 |
| 4 | Random | 0.1402 | 0.0142 | 6.089 | 15.794 |
| 5 | Profit Accept | 0.0726 | 0.0009 | 6.408 | 15.523 |
| 6 | Heuristic | 0.0719 | 0.0016 | 6.421 | 15.545 |

### Full Main accept-rate ranking
| Rank | Method | accept_rate_mean | on_time_rate_mean | avg_lateness_mean |
| --- | --- | ---: | ---: | ---: |
| 1 | Truck Local Search | 0.0771 | 0.1992 | 13.620 |
| 2 | Truck Only | 0.0547 | 0.2450 | 12.068 |
| 3 | HGAT-POMO | 0.0211 | 0.3033 | 8.973 |
| 4 | Random | 0.0142 | 0.1402 | 23.208 |
| 5 | Heuristic | 0.0016 | 0.0719 | 25.005 |
| 6 | Profit Accept | 0.0009 | 0.0726 | 24.724 |

## Source files
- Frozen manifest: `experiments/frozen_models_20260419/manifest.json`
- Train log: `experiments/retrain_open_20260419/logs/train_stdout.log`
- Formal summary: `experiments/thesis_protocol_20260420_formal/summary.json`
- Cost table: `experiments/thesis_protocol_20260420_formal/table_cost.csv`
- Ops table: `experiments/thesis_protocol_20260420_formal/table_ops.csv`
- Ablation table: `experiments/thesis_protocol_20260420_formal/table_ablation.csv`
