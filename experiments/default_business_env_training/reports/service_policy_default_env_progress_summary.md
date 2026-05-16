# ServicePolicy Default-Business Environment Progress Summary

## Context

The original strict environment is not the main training target because 80% acceptance and 50% on-time rate were not reachable there. The validated training target is now:

- `default_business_env`
- `response_window=5.0`
- `delivery_window_extension=+3.0`
- `resources=2`

Oracle remains the upper reference:

- eval=50: `oracle_best_on_time` acc `0.914667`, on-time `0.518950`, hard `0`
- eval=100: `oracle_best_on_time` acc `0.911667`, on-time `0.524314`, hard `0`

## Failure Progression

| stage | key change | eval | acc | on_time | hard | diagnosis |
|---|---|---:|---:|---:|---:|---|
| original sanity B | route/lateness loss, no outcome labels | 20 | 0.951667 | 0.434326 | 0 | learned high acceptance, missed on-time sequencing |
| guarded A | lateness-guarded decode | 20 | 0.898333 | 0.460111 | 0 | guard helped but labels were still all-positive for accept |
| augmented threshold=1 | outcome labels, risky relabel at lateness > 1 | 20 | 0.748333 | 0.576837 | 0 | on-time recovered, acceptance became too conservative |
| severe8 + assignment | outcome labels, severe relabel at lateness > 8, assignment loss | 20 | 0.818333 | 0.551935 | 0 | first ServicePolicy pass |
| severe8 + assignment | same model | 50 | 0.806000 | 0.526055 | 0 | stable pass |
| severe8 + assignment | same model | 100 | 0.803333 | 0.518257 | 0 | stable pass |

## Why The Final Gate Worked

The decisive fixes were not RL or longer training. They were data and supervised-head corrections:

1. Outcome labels replaced step-local lateness labels, so the model saw final late outcomes.
2. Severe-risk relabeling avoided the two extremes:
   - no relabeling: model accepts too much and misses on-time;
   - threshold `1.0`: model becomes too conservative.
3. Assignment loss trained the `(k, j)` service pairing instead of only the next order `j`.
4. Lateness-guarded decoding remained useful, but decoding alone could not fix missing supervision.

## Final Model Status

The current best ServicePolicy gate is:

- model directory: `experiments/default_business_env_training/models/augmented_severe8_assignment_gate`
- dataset: `experiments/default_business_env_training/imitation/oracle_best_on_time_augmented_severe8_dataset.pt`
- decode mode: `service_policy_lateness_guarded`
- threshold settings:
  - `lateness-risk-threshold=8.0`
  - `max-predicted-lateness=20.0`
  - `accept-risk-penalty=1.0`
  - `on-time-priority-bonus=0.5`

It passes 80/50 at eval=20, eval=50, and eval=100.

## RL Decision

RL fine-tune is not required to prove the 80/50 feasibility of a trained policy under `default_business_env`.

If RL is attempted later, it should be a light fine-tune from the severe8 assignment-aware checkpoint, with rollback to imitation-only if either acceptance or on-time drops below the eval=50/100 gate.

## Thesis-Ready Conclusion

在默认业务环境 `response_window=5.0`、`delivery_window_extension=+3.0`、`resources=2` 下，单纯延长训练或直接引入 RL 并不是 ServicePolicy 达成 80/50 的关键。实验表明，早期模型失败主要来自 imitation 数据缺少最终迟到结果、accept 标签全为正例以及 assignment head 未被监督，导致模型学会高接单但不能稳定复现 oracle 的准时服务策略。通过将最终订单迟到结果回写到 imitation 样本、仅对严重迟到订单进行风险重标记，并增加 `(k,j)` assignment loss 后，ServicePolicy 在 eval=20、50、100 上分别达到 `0.818/0.552`、`0.806/0.526`、`0.803/0.518`，且 hard violations 均为 0。因此，在调整后的业务约束下，监督式 imitation 已经可以学习到接近 oracle 的有效策略，后续 RL 只应作为可选微调，而不是达成 80/50 的必要条件。
