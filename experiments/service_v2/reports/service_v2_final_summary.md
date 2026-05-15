# Service V2 Final Summary

## Scope

This PR implements the service-v2 experimentation framework and a sequence of teacher candidates for acceptance/routing under strict business gates.

Implemented teacher families:

- acceptance insertion teachers;
- beam and deadline/on-time oracle variants;
- joint accept-route beam;
- guarded joint teacher;
- baseline-anchored tail-risk constrained teacher;
- safe-deviation tail-risk teacher.

## Gate Results

The early insertion and beam teachers did not provide a deployable teacher under the strict gate. The joint and guarded teachers also failed because they could improve one headline metric while worsening tail risk or cost.

Key guarded failure at eval 5:

- raw baseline: acc `0.506667`, on_time `0.355263`, late `49`, avg_late `12.203287`, max_late `31.124878`
- guarded joint: acc `0.506667`, on_time `0.368421`, late `48`, avg_late `14.617627`, max_late `39.300615`

The baseline-anchored tail-risk teacher passed 5/10/20 safety gates by reproducing raw baseline exactly. That proves safety, but not value.

The safe-deviation teacher also stayed safe on 5/10/20, but found no accepted safe deviations:

- eval 5: `safe_deviation_rate = 0.0`
- eval 10: `safe_deviation_rate = 0.0`
- eval 20: `safe_deviation_rate = 0.0`

Common rejected-deviation reasons were `average_lateness_regression`, `max_lateness_regression`, `distance_regression`, `energy_regression`, `on_time_rate_drop`, and `late_orders_increase`.

## Dataset And Training Decision

No imitation dataset was generated.

Reason: the only safe teacher behavior is raw-baseline fallback. A dataset from this teacher would be baseline distillation, not a source of improved service decisions.

`ServicePolicy` was not trained.

Reason: the safe-deviation rate is `0%`, below the useful 1%-3% threshold, and there are no nontrivial improvements for a student policy to learn.

## PR Value

The value of this PR is the framework and the negative result:

- the service-v2 evaluation and teacher infrastructure is now in place;
- strict gates now catch false improvements that worsen tail lateness or cost;
- the experiments show that, under the current business constraints, teacher search cannot safely improve over raw baseline;
- the next useful direction is not more teacher variants or longer training, but business-constraint sensitivity analysis.

This PR does not claim a replacement strategy for raw baseline.

