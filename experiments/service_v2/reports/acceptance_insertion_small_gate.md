# Acceptance Insertion Small-Gate Report

## Scope

This run implemented and evaluated the first Service V2 step only:

- acceptance-aware environment state fields;
- hard/soft feasibility classification;
- acceptance-insertion schedulers;
- business metric evaluation with rejection reason output.

No ServicePolicy training was started. The frozen baseline weight was not modified:

`experiments/frozen_models_20260419/model_main_ep200.pt`

## Implemented Files

- `src/schedulers/feasibility.py`
- `src/schedulers/insertion_objective.py`
- `src/schedulers/acceptance_insertion.py`
- `src/experiments/eval_acceptance_insertion.py`
- compatible state/feature extensions in `src/env/td_env.py`
- optional `service_v2` graph feature mode in `src/graph/build_graph_pyg.py`

## Validation Commands

Static check:

```powershell
python -m py_compile src/env/td_env.py src/graph/build_graph_pyg.py src/schedulers/__init__.py src/schedulers/feasibility.py src/schedulers/insertion_objective.py src/schedulers/acceptance_insertion.py src/experiments/eval_acceptance_insertion.py
```

Smoke check:

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/smoke_guarded_acceptance_insertion --eval-instances 1 --eval-progress-every 1 --methods raw_baseline,edd_insertion,regret_insertion,min_lateness_insertion,hybrid_score_insertion --baseline-model-path experiments/frozen_models_20260419/model_main_ep200.pt --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
```

Small-gate checks:

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/small_gate_eval5_guarded --eval-instances 5 --eval-progress-every 5 --methods raw_baseline,v2_repair_only,edd_insertion,regret_insertion,min_lateness_insertion,hybrid_score_insertion --baseline-model-path experiments/frozen_models_20260419/model_main_ep200.pt --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/small_gate_eval10_guarded --eval-instances 10 --eval-progress-every 10 --methods raw_baseline,v2_repair_only,edd_insertion,regret_insertion,min_lateness_insertion,hybrid_score_insertion --baseline-model-path experiments/frozen_models_20260419/model_main_ep200.pt --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/small_gate_eval20_guarded --eval-instances 20 --eval-progress-every 20 --methods raw_baseline,v2_repair_only,edd_insertion,regret_insertion,min_lateness_insertion,hybrid_score_insertion --baseline-model-path experiments/frozen_models_20260419/model_main_ep200.pt --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
```

## Results

### eval_instances=5

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.506667 | 0.355263 | 49 | 12.203287 | 31.124878 | 18.077902 | 261.992519 | 0 | true |
| v2_repair_only | 0.526667 | 0.303797 | 55 | 13.665207 | 51.748751 | 20.956155 | 312.798919 | 0 | false |
| edd_insertion | 0.506667 | 0.144737 | 65 | 18.936190 | 61.098158 | 16.583107 | 274.077197 | 0 | false |
| regret_insertion | 0.500000 | 0.133333 | 65 | 30.271056 | 112.389536 | 26.210147 | 481.835832 | 0 | false |
| min_lateness_insertion | 0.506667 | 0.131579 | 66 | 28.695886 | 112.389536 | 26.379232 | 477.637534 | 0 | false |
| hybrid_score_insertion | 0.506667 | 0.131579 | 66 | 28.688955 | 112.389536 | 26.388936 | 477.637534 | 0 | false |

### eval_instances=10

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.560000 | 0.375000 | 105 | 10.989798 | 33.856089 | 35.569443 | 522.175673 | 0 | true |
| v2_repair_only | 0.513333 | 0.298701 | 108 | 12.329668 | 51.748751 | 38.200877 | 588.272216 | 0 | false |
| edd_insertion | 0.503333 | 0.145695 | 129 | 18.178693 | 61.098158 | 33.651774 | 565.788840 | 0 | false |
| regret_insertion | 0.503333 | 0.119205 | 133 | 28.135617 | 118.551805 | 51.087096 | 973.646012 | 0 | false |
| min_lateness_insertion | 0.503333 | 0.132450 | 131 | 27.233169 | 121.046376 | 50.963065 | 956.428895 | 0 | false |
| hybrid_score_insertion | 0.503333 | 0.139073 | 130 | 27.071535 | 118.551805 | 51.085919 | 956.723207 | 0 | false |

### eval_instances=20

| method | acceptance_rate | on_time_rate | late_orders | avg_late | max_late | energy | distance | hard | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw_baseline | 0.533333 | 0.328125 | 215 | 12.883776 | 35.936546 | 75.931073 | 1115.095666 | 0 | true |
| v2_repair_only | 0.506667 | 0.299342 | 213 | 13.121789 | 51.748751 | 76.269763 | 1155.775931 | 0 | false |
| edd_insertion | 0.501667 | 0.199336 | 241 | 20.076130 | 63.607843 | 70.388189 | 1177.808528 | 0 | false |
| regret_insertion | 0.501667 | 0.159468 | 253 | 30.674021 | 118.551805 | 104.618772 | 1941.867348 | 0 | false |
| min_lateness_insertion | 0.501667 | 0.192691 | 243 | 29.838133 | 121.046376 | 103.227806 | 1908.048004 | 0 | false |
| hybrid_score_insertion | 0.501667 | 0.192691 | 243 | 29.573031 | 118.551805 | 103.003104 | 1898.726277 | 0 | false |

## Diagnosis

The guarded acceptance logic fixed the most obvious failure mode from the first smoke run: it no longer blindly accepts many dynamic orders that are likely to become late after queueing.

However, the route sequencing part of the insertion heuristic is still weaker than the learned baseline. Even with acceptance held near baseline, all insertion variants reduce on-time rate and increase late orders. Regret/min-lateness/hybrid variants also substantially increase max lateness, energy, and distance on 10/20 instance tests.

## Decision

The small-data gate failed. Do not start ServicePolicy imitation learning or RL fine-tuning from these heuristic labels yet.

Recommended next engineering step is not long training. The next useful step is to replace the one-step insertion route selector with a stronger oracle/beam route generator before producing imitation labels.

