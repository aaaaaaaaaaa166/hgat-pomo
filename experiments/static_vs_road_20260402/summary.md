# Static vs Road-Aware Comparison

## Setup

- Shared training scale:
  - `epochs=40`
  - `batch-size=8`
  - `N=30`
  - `K=8`
  - `use-curriculum`
- Static model:
  - train with `--edge-mode static`
- Road-aware model:
  - train with `--edge-mode road --time-dependent`

## Raw Results In Their Own Evaluation Environments

These values are useful for sanity checks, but the absolute costs are not directly comparable because the objectives differ.

| Model | Eval environment | best(K) mean | mean(K) mean | worst(K) mean |
|---|---|---:|---:|---:|
| Static HGAT-POMO | static | 232.153 | 282.330 | 347.911 |
| Road-aware HGAT-POMO | road + time-dependent | 337.240 | 395.360 | 472.857 |

## Fair Comparison On The Same Road-Aware Evaluation Environment

This is the main result to cite.

| Model | Eval environment | best(K) mean | mean(K) mean | worst(K) mean |
|---|---|---:|---:|---:|
| Static-trained HGAT-POMO | road + time-dependent | 317.422 | 381.597 | 455.327 |
| Road-aware HGAT-POMO | road + time-dependent | 337.240 | 395.360 | 472.857 |

## Takeaway

- On the current formal experiment, the road-aware model did **not** outperform the static-trained model under the same road-aware evaluation.
- Under the road-aware evaluation objective, the static-trained model is lower-cost by `19.818` on `best(K) mean`, which is about `5.9%` better than the current road-aware model.
- This suggests the road-aware feature integration is runnable and learnable, but the present hyperparameters and training recipe are not yet enough to beat the stronger static baseline.

## Metric Files

- Static model in static environment:
  - `experiments/static_vs_road_20260402/metrics/eval_static_metrics.json`
- Static model in road-aware environment:
  - `experiments/static_vs_road_20260402/metrics/eval_static_model_on_road_env.json`
- Road-aware model in road-aware environment:
  - `experiments/road_formal_20260402/metrics/eval_metrics.json`
