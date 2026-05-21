# ServicePolicy Multi-Seed Gate - 2026-05-21

## Per-Seed Results

| eval | seed | acc | on-time | late | avg late | max late | hard | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0 | 0.806000 | 0.526055 | 573 | 23.875986 | 62.703487 | 0 | yes |
| 50 | 1 | 0.802667 | 0.523256 | 574 | 22.106350 | 61.732244 | 0 | yes |
| 50 | 2 | 0.787333 | 0.556308 | 524 | 21.680931 | 63.327138 | 0 | no |
| 100 | 0 | 0.803333 | 0.518257 | 1161 | 22.817444 | 65.605548 | 0 | yes |
| 100 | 1 | 0.797333 | 0.530518 | 1123 | 22.571076 | 74.348933 | 0 | no |
| 100 | 2 | 0.788000 | 0.541455 | 1084 | 22.435019 | 67.991005 | 0 | no |

## Aggregate

| eval | seeds | acc mean | acc std | acc min | on-time mean | on-time std | on-time min | pass seeds | all pass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 0,1,2 | 0.798667 | 0.008129 | 0.787333 | 0.535206 | 0.014965 | 0.523256 | 2/3 | no |
| 100 | 0,1,2 | 0.796222 | 0.006309 | 0.788000 | 0.530077 | 0.009476 | 0.518257 | 1/3 | no |

## Interpretation

- The best checkpoint remains useful and consistently keeps on-time rate above 50%.
- Acceptance is the unstable side of the 80/50 gate: eval=50 passes 2/3 seeds and eval=100 passes only seed 0.
- The model should be reported as a strong single-seed gate result plus a partial multi-seed validation, not as fully robust across seeds.
