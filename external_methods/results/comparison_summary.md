# Workspace Comparison Summary

## HGAT-POMO dynamic thesis protocol
| Rank | Method | full_main cost_mean_mean | on_time_rate_mean | accept_rate_mean |
| --- | --- | ---: | ---: | ---: |
| 1 | HGAT-POMO | 145.255 | 0.3033 | 0.0211 |
| 2 | Truck Only | 194.817 | 0.2450 | 0.0547 |
| 3 | Truck Local Search | 231.389 | 0.1992 | 0.0771 |
| 4 | Profit Accept | 336.965 | 0.0726 | 0.0009 |
| 5 | Heuristic | 360.777 | 0.0719 | 0.0016 |
| 6 | Random | 415.485 | 0.1402 | 0.0142 |

## External static CVRP runs
These runs use the same open CVRPLIB split files and the same sampler, but on a frozen static CVRP proxy.

| Method | Run Path | Key Test Metric | Extra |
| --- | --- | --- | --- |
| attention_learn_to_route | `external_methods/results/attention_learn_to_route/formal_round1_20260420/metrics.json` | test_avg_cost=97.21420288085938 | best_val_cost=90.23748779296875 |
| attention_learn_to_route | `external_methods/results/attention_learn_to_route/formal_round2_20260420_20k50ep/metrics.json` | test_avg_cost=83.75250244140625 | best_val_cost=87.3285903930664 |
| attention_learn_to_route | `external_methods/results/attention_learn_to_route/smoke_20260420/metrics.json` | test_avg_cost=214.42803955078125 | best_val_cost=169.83636474609375 |
| pomo | `external_methods/results/pomo/smoke_20260420/metrics.json` | aug_avg_cost=86.35505676269531 | no_aug_avg_cost=91.55126953125 |
