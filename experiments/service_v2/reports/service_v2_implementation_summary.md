# Service V2 Implementation Summary

## What Changed

Implemented the Service V2 scaffolding without changing the frozen baseline model:

- Environment state now carries explicit acceptance lifecycle fields:
  - `accepted`
  - `rejected`
  - `expired`
  - `served`
  - `accept_time`
  - `finish_time`
  - `reject_reason`
- `EnvConfig` includes response-window and business reward/cost parameters.
- `feature_mode=legacy` remains the compatible default.
- `feature_mode=service_v2` adds response-window, estimated-arrival, arrival-slack, predicted-lateness, accepted/rejected/expired, tight-window, and urgency features.
- Added hard/soft feasibility classification in `src/schedulers/feasibility.py`.
- Added acceptance-aware insertion schedulers:
  - `edd_insertion`
  - `regret_insertion`
  - `min_lateness_insertion`
  - `hybrid_score_insertion`
- Added business metric evaluation:
  - `src/experiments/eval_acceptance_insertion.py`
- Added ServicePolicy architecture:
  - `src/models/service_policy.py`
- Added imitation dataset builder:
  - `src/experiments/build_imitation_dataset.py`
- Added supervised warm-start trainer:
  - `src/experiments/train_service_policy.py`
- Added ServicePolicy evaluator:
  - `src/experiments/eval_service_policy.py`

## Validation

Static compile passed:

```powershell
python -m py_compile src/models/service_policy.py src/experiments/build_imitation_dataset.py src/experiments/train_service_policy.py src/experiments/eval_service_policy.py src/env/td_env.py src/graph/build_graph_pyg.py src/schedulers/feasibility.py src/schedulers/insertion_objective.py src/schedulers/acceptance_insertion.py src/experiments/eval_acceptance_insertion.py
```

Imitation dataset smoke passed:

```powershell
python -m src.experiments.build_imitation_dataset --instances 1 --progress-every 1 --teacher-method edd_insertion --output-path experiments/service_v2/imitation/smoke_imitation_dataset.pt --report-path experiments/service_v2/reports/smoke_imitation_quality.md --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
```

Smoke output:

- samples: `20`
- acceptance_rate: `0.533333`
- on_time_rate: `0.250000`
- late_orders: `12`

## Gate Decision

Do not start long training yet.

The insertion teachers failed the 5/10/20 small-data gate in `acceptance_insertion_small_gate.md`. The dataset builder and trainer are ready, but using the current teacher for imitation would likely train the ServicePolicy to reproduce poor routing decisions.

## Recommended Next Step

Improve the teacher first:

1. Replace one-step insertion routing with a stronger beam/oracle route generator.
2. Re-run `eval_acceptance_insertion.py` on 5/10/20.
3. Only build the full imitation dataset after the teacher has:
   - acceptance_rate >= raw_baseline;
   - on_time_rate >= raw_baseline;
   - late_orders <= raw_baseline;
   - hard_constraint_violations = 0.

