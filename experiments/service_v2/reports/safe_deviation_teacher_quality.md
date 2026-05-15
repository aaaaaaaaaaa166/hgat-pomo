# Safe Deviation Teacher Quality

## Summary

The safe-deviation teacher did not produce useful imitation signal in this run.

Across eval sizes 5/10/20:

- safe deviation sample ratio: `0%`
- fallback ratio: `100%`
- nontrivial improvement instances: `0`
- teacher behavior: raw baseline fallback only

## Rejected Deviation Reasons

Eval 20 rejected candidate deviations for:

- `average_lateness_regression`: 18
- `max_lateness_regression`: 18
- `distance_regression`: 16
- `energy_regression`: 15
- `no_nontrivial_improvement`: 15
- `on_time_rate_drop`: 12
- `late_orders_increase`: 9
- `acceptance_rate_drop`: 6

This means the candidate search still tends to buy local service changes by worsening tail risk or cost, so the safety filter correctly rejects it.

## Dataset Recommendation

Do not generate a formal imitation dataset from this teacher yet.

Even if generated, it would contain only baseline fallback samples and no safe-deviation samples. That would train a baseline distillation model rather than a better service policy.

## Training Recommendation

Do not train `ServicePolicy`.

The safe-deviation rate is below the useful threshold of 1%-3%, and there are no nontrivial improvements to learn from.

