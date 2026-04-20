# External Methods Workspace

This folder keeps third-party routing baselines, shared open-data exports, and comparison results separate from the main HGAT-POMO code.

## Layout

- `repos/`: cloned upstream repositories
- `data/`: exported datasets derived from the repo's open CVRPLIB split protocol
- `results/`: training logs, checkpoints, metrics, and consolidated summaries
- `scripts/`: adapters and wrappers so upstream repos can use the same frozen data protocol
- `docs/`: manifests, notes, and dependency instructions

## Current scope

The main HGAT-POMO thesis protocol is a dynamic truck-drone setting. Most downloaded open-source repos are static CVRP solvers. To avoid misleading apples-to-oranges claims, this workspace separates two comparison layers:

1. `HGAT-POMO thesis formal`
   Uses the repo's existing `main_eval.py` protocol and its dynamic operating metrics.

2. `External static CVRP proxy`
   Uses the same open CVRPLIB split files and the same `sample_open_vrp_base(...)` sampler, but freezes the stream into deterministic static train/val/test datasets that external CVRP methods can consume.

## Recommended workflow

1. Export shared data

```powershell
python external_methods/scripts/export_open_cvrplib_static.py
```

2. Consolidate current HGAT-POMO reference results

```powershell
python external_methods/scripts/collect_hgat_reference.py
```

3. Train an external method on the exported data

```powershell
python external_methods/scripts/train_attention_cvrplib.py
python external_methods/scripts/train_pomo_cvrplib.py
python external_methods/scripts/train_rl4co_cvrplib.py
```

4. Rebuild the workspace summary after new runs finish

```powershell
python external_methods/scripts/build_comparison_summary.py
```

## Existing organized outputs

- HGAT reference summary:
  - `external_methods/results/hgat_pomo_reference/summary.json`
  - `external_methods/results/hgat_pomo_reference/summary.md`
- Shared exported data protocol:
  - `external_methods/data/open_cvrplib_n30/protocol.json`
- Workspace comparison rollup:
  - `external_methods/results/comparison_summary.json`
  - `external_methods/results/comparison_summary.md`
