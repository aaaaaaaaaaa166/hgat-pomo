# ServicePolicy Eval=200 And Threshold Notes

## Best Confirmed Gate

The best confirmed model remains:

- `experiments/default_business_env_training/models/augmented_severe8_assignment_gate`
- severe-risk relabel threshold: `8.0`
- decode: `service_policy_lateness_guarded`

It passes:

| eval | acc | on_time | hard | status |
|---:|---:|---:|---:|---|
| 20 | 0.818333 | 0.551935 | 0 | pass |
| 50 | 0.806000 | 0.526055 | 0 | pass |
| 100 | 0.803333 | 0.518257 | 0 | pass |

## eval=200 Probe

Service-only eval=200 was run as an additional stability probe:

| decode | acc | on_time | hard | status |
|---|---:|---:|---:|---|
| guarded severe8 | 0.798167 | 0.526624 | 0 | misses acceptance by 0.001833 |
| raw | 0.798000 | 0.526525 | 0 | misses acceptance by 0.002000 |
| loose guarded | 0.797667 | 0.526536 | 0 | misses acceptance by 0.002333 |

Interpretation: eval=200 is not an on-time failure. It is a tiny acceptance-margin failure. Decode loosening does not recover it, so this is not primarily a decoding-threshold issue.

## Severe10 Check

To test whether slightly fewer risky accept relabels improve acceptance, a severe10 dataset/model was built:

- threshold: `10.0`
- accept relabels: `191` versus `218` for severe8

eval=100 result:

| model | acc | on_time | hard | status |
|---|---:|---:|---:|---|
| severe10 assignment-aware | 0.794333 | 0.495174 | 0 | fail |

Interpretation: relaxing the relabel threshold to `10.0` degraded both acceptance and on-time. The severe8 threshold is the current best operating point.

## Decision

- Use severe8 assignment-aware ServicePolicy as the trained model result for default_business_env.
- Claim stable pass at eval=20/50/100.
- Do not claim eval=200 pass.
- Do not run RL from severe10.
- If eval=200 must pass later, the next targeted fix should be acceptance calibration, not another broad training run.
