# Complete Comparison Training Summary - 2026-05-21

## Scope

Completed comparison set for the current thesis target: `default_business_env`, 80% acceptance, 50% on-time, zero hard violations. After the environment changed, all previously compared runnable methods were re-evaluated under the same new environment profile: response window `5.0`, delivery window extension `+3.0`, resource count `2`.

Raw artifacts:

- `experiments/default_business_env_training/evaluation/complete_old_methods_default_env_20260521/full_old_methods_default_env_eval50_20260521.csv`
- `experiments/default_business_env_training/evaluation/complete_old_methods_default_env_20260521/full_old_methods_default_env_eval100_20260521.csv`
- `experiments/default_business_env_training/evaluation/complete_old_methods_default_env_20260521/full_old_methods_default_env_eval50_100_20260521.csv`

## Full Old-Method Recomparison, Eval=100 Seed 0

| method | acc | on-time | late | avg late | max late | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.603333 | 0.569061 | 780 | 8.943759 | 27.030407 | 0 | no |
| v2_repair_only | 0.598333 | 0.596100 | 725 | 9.231487 | 32.251497 | 0 | no |
| edd_insertion | 0.568667 | 0.563306 | 745 | 14.044912 | 45.159086 | 0 | no |
| regret_insertion | 0.565333 | 0.541863 | 777 | 31.116151 | 124.110140 | 0 | no |
| min_lateness_insertion | 0.567000 | 0.555556 | 756 | 30.145738 | 124.110140 | 0 | no |
| hybrid_score_insertion | 0.566667 | 0.557059 | 753 | 30.124537 | 124.110140 | 0 | no |
| beam_oracle_insertion | 0.828667 | 0.502816 | 1236 | 18.070489 | 75.436083 | 0 | yes |
| deadline_beam_oracle | 0.729667 | 0.534947 | 1018 | 12.795685 | 45.372674 | 0 | no |
| conservative_deadline_beam | 0.681333 | 0.536693 | 947 | 12.304481 | 49.794250 | 0 | no |
| ontime_beam_oracle | 0.687333 | 0.541222 | 946 | 12.254988 | 49.794250 | 0 | no |
| guarded_ontime_beam | 0.547000 | 0.517977 | 791 | 10.953185 | 37.252650 | 0 | no |
| policy_accept_ontime_beam | 0.607333 | 0.548299 | 823 | 11.416559 | 42.633784 | 0 | no |
| joint_accept_route_beam | 0.903333 | 0.411808 | 1594 | 17.702433 | 75.093013 | 0 | no |
| joint_accept_route_beam_guarded | 0.533000 | 0.513446 | 778 | 11.060149 | 44.410710 | 0 | no |
| tail_risk_constrained_joint_beam | 0.601333 | 0.569845 | 776 | 9.249591 | 31.364686 | 0 | no |
| tail_risk_constrained_joint_beam_safe_deviation | 0.599000 | 0.566500 | 779 | 8.942271 | 28.287932 | 0 | no |
| oracle_best_acceptance | 0.936667 | 0.485765 | 1445 | 20.713926 | 67.483751 | 0 | no |
| oracle_best_on_time | 0.911667 | 0.524314 | 1301 | 23.620672 | 66.221815 | 0 | yes |
| service_policy_imitation | 0.813667 | 0.523146 | 1164 | 19.993843 | 63.901285 | 0 | yes |

## Full Old-Method Recomparison, Eval=50 Seed 0

| method | acc | on-time | late | avg late | max late | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.593333 | 0.544944 | 405 | 8.624550 | 25.731184 | 0 | no |
| v2_repair_only | 0.581333 | 0.577982 | 368 | 8.870040 | 27.031143 | 0 | no |
| edd_insertion | 0.560000 | 0.569048 | 362 | 13.941557 | 45.159086 | 0 | no |
| regret_insertion | 0.558667 | 0.544153 | 382 | 30.131783 | 100.179368 | 0 | no |
| min_lateness_insertion | 0.558667 | 0.565632 | 364 | 29.500404 | 108.129318 | 0 | no |
| hybrid_score_insertion | 0.558000 | 0.565114 | 364 | 29.302765 | 108.129318 | 0 | no |
| beam_oracle_insertion | 0.848000 | 0.503145 | 632 | 18.969939 | 75.436083 | 0 | yes |
| deadline_beam_oracle | 0.736000 | 0.530797 | 518 | 13.062708 | 44.738559 | 0 | no |
| conservative_deadline_beam | 0.684667 | 0.530672 | 482 | 12.555381 | 49.794250 | 0 | no |
| ontime_beam_oracle | 0.694000 | 0.535062 | 484 | 12.794918 | 49.794250 | 0 | no |
| guarded_ontime_beam | 0.547333 | 0.515225 | 398 | 10.862840 | 36.089831 | 0 | no |
| policy_accept_ontime_beam | 0.601333 | 0.544346 | 411 | 11.574815 | 42.633784 | 0 | no |
| joint_accept_route_beam | 0.910000 | 0.417582 | 795 | 17.763125 | 64.265702 | 0 | no |
| joint_accept_route_beam_guarded | 0.539333 | 0.517923 | 390 | 11.627542 | 44.410710 | 0 | no |
| tail_risk_constrained_joint_beam | 0.593333 | 0.544944 | 405 | 8.624550 | 25.731184 | 0 | no |
| tail_risk_constrained_joint_beam_safe_deviation | 0.587333 | 0.555051 | 392 | 8.793353 | 26.635047 | 0 | no |
| oracle_best_acceptance | 0.954667 | 0.479749 | 745 | 21.206119 | 63.515615 | 0 | no |
| oracle_best_on_time | 0.914667 | 0.518950 | 660 | 24.145694 | 66.110448 | 0 | yes |
| service_policy_imitation | 0.820000 | 0.534146 | 573 | 20.196305 | 57.971370 | 0 | yes |

## 100-Instance Training Ablation At Eval=100 Seed 0

| variant | acc | on-time | late | avg late | max late | hard | pass | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|---|
| final_full | 0.813667 | 0.523146 | 1164 | 19.993843 | 63.901285 | 0 | yes | Selected risk-balanced final model. |
| no_assignment_loss | 0.816333 | 0.467946 | 1303 | 25.014387 | 73.008002 | 0 | no | Keeps acceptance but loses on-time performance. |
| no_pairwise_loss | 0.752000 | 0.570479 | 969 | 21.046027 | 59.816686 | 0 | no | Improves on-time but becomes too conservative in acceptance. |
| no_risky_accept_penalty | 0.814000 | 0.542179 | 1118 | 21.879403 | 65.293611 | 0 | yes | Competitive on 80/50 but worse lateness intensity. |
| ep10_continuation_rejected | 0.750000 | 0.582667 | 939 | 19.975020 | 58.969518 | 0 | no | Over-trained toward on-time; acceptance collapses. |

## Competitive Multi-Seed Eval=100

| variant | seeds | acc mean | acc min | on-time mean | on-time min | avg late mean | max late max | pass seeds | all pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| final_full | 0,1,2 | 0.803667 | 0.797667 | 0.533174 | 0.523146 | 19.617918 | 63.901285 | 1/3 | no |
| no_risky_accept_penalty | 0,1,2 | 0.803333 | 0.794333 | 0.542056 | 0.538877 | 21.363969 | 66.401204 | 2/3 | no |

## Final Decision

- The old comparison methods have now all been re-run under `default_business_env`; this closes the earlier gap where only raw/oracle/ServicePolicy were in the main comparison.
- At eval=100 seed 0, the methods that reach the 80/50 gate are `beam_oracle_insertion`, `oracle_best_on_time`, and the final `service_policy_imitation`.
- `service_policy_imitation` is weaker than `oracle_best_on_time` on acceptance, but it is competitive with `beam_oracle_insertion` while using the learned ServicePolicy checkpoint instead of a beam teacher.
- Many heuristics have higher on-time than 50% but fail acceptance badly; `joint_accept_route_beam` has high acceptance but misses the on-time target.
- Keep `severe8_assignment_100inst_20260521` as the selected final ServicePolicy because it passes eval=50/100/200 seed 0 and is supported by the complete old-method recomparison.
- External AM/POMO 20000 service runs remain not closed in the workspace and should not be claimed as completed.
