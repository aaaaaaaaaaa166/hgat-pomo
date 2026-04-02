# Ablation Runner

Use this script to run thesis-style ablations and export result tables.

## Quick Start

```bash
python -m src.experiments.run_ablation ^
  --epochs 40 ^
  --batch-size 16 ^
  --N 30 ^
  --K 8 ^
  --eval-instances 100 ^
  --seeds 0,1,2 ^
  --variants full,no_traffic,no_time_window,no_soc,no_curriculum ^
  --eval-no-store-traj
```

If you also want stronger eval baselines (`truck_only`, `heuristic`) in each evaluation:

```bash
python -m src.experiments.run_ablation ^
  --eval-extra-baselines
```

Outputs are written to `experiments/ablation/`:

- `runs.csv`: per-run records (`variant x seed`)
- `summary.csv`: aggregated statistics by variant
- `summary.md`: markdown table for direct use in the thesis
- `models/`: trained checkpoints
- `metrics/`: evaluation JSON files
- `logs/`: train/eval logs

## Export Three-line Table (LaTeX)

```bash
python -m src.experiments.export_threeline_table ^
  --summary-csv experiments/ablation/summary.csv ^
  --out experiments/ablation/paper_table.tex ^
  --caption "Ablation Study Results" ^
  --label "tab:ablation_results" ^
  --print
```
