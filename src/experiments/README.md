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

## Open Dataset Protocol (CVRPLIB)

1) Prepare open-source instances and deterministic train/val/test split:

```bash
python -m src.experiments.setup_open_cvrplib ^
  --dataset-dir datasets/cvrplib ^
  --include-families A,B,P ^
  --min-customers 30
```

This now writes a fuller evidence bundle under `datasets/cvrplib/splits/`:

- `train.txt`, `val.txt`, `test.txt`: deterministic split lists
- `instance_manifest.csv`: per-instance family / size / split / source trace
- `family_summary.csv`: family-level coverage counts for A/B/P
- `summary.json`: machine-readable dataset coverage summary

2) Run fixed thesis protocol (full + ablations, 3 seeds, n_instances>=100) and export three core tables:

```bash
python -m src.experiments.run_thesis_protocol ^
  --model-main experiments/frozen_models_20260419/model_main_ep200.pt ^
  --model-backup experiments/frozen_models_20260419/model_backup_ep140.pt ^
  --dataset-path datasets/cvrplib ^
  --dataset-split-file datasets/cvrplib/splits/test.txt ^
  --seeds 0,1,2 ^
  --n-instances 100 ^
  --edge-mode road ^
  --time-dependent
```

Outputs under `experiments/thesis_protocol_20260420_formal/`:

- `config.json`: frozen protocol config, dataset summary, and model manifest snapshot
- `runs.csv`: per-seed raw results in long format (`scenario x seed x method`)
- `table_cost.csv`: overall cost table (`best/mean/worst`) for model plus all baselines
- `table_ops.csv`: operations KPI table for model plus all baselines
- `table_ablation.csv`: ablation deltas against `full_main`
- `metrics/` and `logs/`: detailed artifacts

## Export Three-line Table (LaTeX)

```bash
python -m src.experiments.export_threeline_table ^
  --summary-csv experiments/ablation/summary.csv ^
  --out experiments/ablation/paper_table.tex ^
  --caption "Ablation Study Results" ^
  --label "tab:ablation_results" ^
  --print
```
