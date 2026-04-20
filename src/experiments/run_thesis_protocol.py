from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


METHOD_ORDER = [
    "model",
    "random",
    "truck_only",
    "heuristic",
    "local_search_truck",
    "profit_accept",
]
METHOD_LABELS = {
    "model": "HGAT-POMO",
    "random": "Random",
    "truck_only": "Truck Only",
    "heuristic": "Heuristic",
    "local_search_truck": "Truck Local Search",
    "profit_accept": "Profit Accept",
}
OPS_KEYS = [
    "accept_rate",
    "reject_rate",
    "on_time_rate",
    "avg_lateness",
    "total_revenue",
    "total_energy",
]


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    if len(out) == 0:
        raise ValueError("seeds must not be empty")
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / float(len(values))
    v = sum((x - m) * (x - m) for x in values) / float(len(values))
    return m, math.sqrt(max(v, 0.0))


def _build_cmd(python_exe: str, args_dict: Dict[str, object]) -> List[str]:
    cmd = [python_exe, "-m", "src.main_eval"]
    for key, value in args_dict.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    return cmd


def _run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd))
        f.write("\n\nOUTPUT:\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _maybe_load_json(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_dataset_summary(repo_root: Path, dataset_split_file: str, explicit_path: str) -> Optional[Path]:
    if explicit_path.strip():
        candidate = _resolve_path(repo_root, explicit_path.strip())
        return candidate if candidate.exists() else None
    if not dataset_split_file.strip():
        return None
    candidate = _resolve_path(repo_root, dataset_split_file.strip()).parent / "summary.json"
    return candidate if candidate.exists() else None


def _detect_model_manifest(repo_root: Path, model_main: str, explicit_path: str) -> Optional[Path]:
    if explicit_path.strip():
        candidate = _resolve_path(repo_root, explicit_path.strip())
        return candidate if candidate.exists() else None
    candidate = _resolve_path(repo_root, model_main).parent / "manifest.json"
    return candidate if candidate.exists() else None


def _method_cost_block(metrics: Dict[str, object], method: str) -> Optional[Dict[str, object]]:
    if method == "model":
        return metrics.get("model")  # type: ignore[return-value]
    baselines = metrics.get("baselines", {})
    if not isinstance(baselines, dict):
        return None
    block = baselines.get(method)
    return block if isinstance(block, dict) else None


def _method_ops_block(metrics: Dict[str, object], method: str) -> Optional[Dict[str, object]]:
    if method == "model":
        return metrics.get("model_ops")  # type: ignore[return-value]
    baseline_ops = metrics.get("baseline_ops", {})
    if not isinstance(baseline_ops, dict):
        return None
    block = baseline_ops.get(method)
    return block if isinstance(block, dict) else None


def _collect_rows(metrics_path: Path, scenario: str, model_tag: str, seed: int) -> List[Dict[str, object]]:
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        cost_block = _method_cost_block(d, method)
        ops_block = _method_ops_block(d, method)
        if not isinstance(cost_block, dict) or not isinstance(ops_block, dict):
            continue
        row: Dict[str, object] = {
            "scenario": scenario,
            "model_tag": model_tag,
            "seed": int(seed),
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "cost_best_mean": float(cost_block["best"]["mean"]),
            "cost_mean_mean": float(cost_block["mean"]["mean"]),
            "cost_worst_mean": float(cost_block["worst"]["mean"]),
            "metrics_path": str(metrics_path.resolve()),
        }
        for key in OPS_KEYS:
            row[key] = float(ops_block.get(key, {}).get("mean", 0.0))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis fixed protocol eval and export summary tables.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-dir", type=str, default="experiments/thesis_protocol_20260419")
    parser.add_argument("--model-main", type=str, required=True)
    parser.add_argument("--model-backup", type=str, default="")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k-nn-orders", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--edge-mode", type=str, default="road", choices=["static", "road"])
    parser.add_argument("--time-dependent", action="store_true")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-format", type=str, default="cvrplib", choices=["cvrplib"])
    parser.add_argument("--dataset-split-file", type=str, default="")
    parser.add_argument("--dataset-summary-json", type=str, default="")
    parser.add_argument("--model-manifest-json", type=str, default="")
    parser.add_argument("--dataset-demand-scale", type=float, default=1.0)
    parser.add_argument("--dataset-no-normalize-coords", action="store_true")
    parser.add_argument("--skip-existing-metrics", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.output_dir).resolve()
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.seeds)
    dataset_summary_path = _detect_dataset_summary(repo_root, args.dataset_split_file, args.dataset_summary_json)
    model_manifest_path = _detect_model_manifest(repo_root, args.model_main, args.model_manifest_json)
    dataset_summary = _maybe_load_json(dataset_summary_path)
    model_manifest = _maybe_load_json(model_manifest_path)
    scenarios: List[Dict[str, object]] = [
        {"scenario": "full_main", "model_tag": "main", "model_path": args.model_main, "flags": {}},
    ]
    if args.model_backup.strip():
        scenarios.append(
            {"scenario": "full_backup", "model_tag": "backup", "model_path": args.model_backup.strip(), "flags": {}}
        )
    scenarios.extend(
        [
            {
                "scenario": "ablate_no_accept_reject",
                "model_tag": "main",
                "model_path": args.model_main,
                "flags": {"ablate_no_accept_reject": True},
            },
            {
                "scenario": "ablate_no_pickup_capacity",
                "model_tag": "main",
                "model_path": args.model_main,
                "flags": {"ablate_no_pickup_capacity": True},
            },
            {
                "scenario": "ablate_no_time_traffic",
                "model_tag": "main",
                "model_path": args.model_main,
                "flags": {"ablate_no_time_traffic": True},
            },
        ]
    )

    base_eval: Dict[str, object] = {
        "N": args.N,
        "K": args.K,
        "n_instances": args.n_instances,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "k_nn_orders": args.k_nn_orders,
        "encoder_layers": args.encoder_layers,
        "tanh_clipping": args.tanh_clipping,
        "temperature": args.temperature,
        "edge_mode": args.edge_mode,
        "time_dependent": args.time_dependent,
        "dataset_path": args.dataset_path,
        "dataset_format": args.dataset_format,
        "dataset_split_file": args.dataset_split_file.strip() if args.dataset_split_file.strip() else None,
        "dataset_demand_scale": args.dataset_demand_scale,
        "dataset_no_normalize_coords": args.dataset_no_normalize_coords,
        "extra_baselines": True,
        "no_store_traj": True,
    }

    protocol_config = {
        "python_exe": args.python_exe,
        "output_dir": str(out_dir),
        "models": {
            "main": str(_resolve_path(repo_root, args.model_main)),
            "backup": str(_resolve_path(repo_root, args.model_backup)) if args.model_backup.strip() else "",
        },
        "seeds": seeds,
        "scenarios": [str(x["scenario"]) for x in scenarios],
        "n_instances": int(args.n_instances),
        "dataset": {
            "path": str(_resolve_path(repo_root, args.dataset_path)),
            "split_file": str(_resolve_path(repo_root, args.dataset_split_file))
            if args.dataset_split_file.strip()
            else "",
            "summary_json": str(dataset_summary_path.resolve()) if dataset_summary_path else "",
            "format": args.dataset_format,
            "demand_scale": float(args.dataset_demand_scale),
            "normalize_coords": bool(not args.dataset_no_normalize_coords),
        },
        "evaluation": {
            "N": int(args.N),
            "K": int(args.K),
            "hidden_dim": int(args.hidden_dim),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "k_nn_orders": int(args.k_nn_orders),
            "encoder_layers": int(args.encoder_layers),
            "tanh_clipping": float(args.tanh_clipping),
            "temperature": float(args.temperature),
            "edge_mode": args.edge_mode,
            "time_dependent": bool(args.time_dependent),
            "methods": METHOD_ORDER,
        },
        "dataset_summary": dataset_summary,
        "model_manifest": model_manifest,
    }
    (out_dir / "config.json").write_text(json.dumps(protocol_config, ensure_ascii=False, indent=2), encoding="utf-8")

    run_rows: List[Dict[str, object]] = []
    total = len(scenarios) * len(seeds)
    run_idx = 0
    t0 = time.time()
    for sc in scenarios:
        for seed in seeds:
            run_idx += 1
            scenario = str(sc["scenario"])
            model_tag = str(sc["model_tag"])
            model_path = str(sc["model_path"])
            flags = dict(sc["flags"])

            metrics_path = metrics_dir / f"{scenario}_seed{seed}.json"
            log_path = logs_dir / f"{scenario}_seed{seed}.log"

            eval_cfg = dict(base_eval)
            eval_cfg.update(flags)
            eval_cfg["model_path"] = model_path
            eval_cfg["eval_seed"] = int(seed)
            eval_cfg["metrics_json"] = str(metrics_path)

            print("\n" + "=" * 88)
            print(f"[{run_idx}/{total}] scenario={scenario} seed={seed}")
            print("=" * 88)
            if args.skip_existing_metrics and metrics_path.exists():
                print(f"Skip eval (metrics exists): {metrics_path}")
            else:
                cmd = _build_cmd(args.python_exe, eval_cfg)
                _run_cmd(cmd, cwd=repo_root, log_path=log_path)

            run_rows.extend(_collect_rows(metrics_path, scenario=scenario, model_tag=model_tag, seed=int(seed)))

    runs_csv = out_dir / "runs.csv"
    run_headers = [
        "scenario",
        "model_tag",
        "seed",
        "method",
        "method_label",
        "cost_best_mean",
        "cost_mean_mean",
        "cost_worst_mean",
        "accept_rate",
        "reject_rate",
        "on_time_rate",
        "avg_lateness",
        "total_revenue",
        "total_energy",
        "metrics_path",
    ]
    _write_csv(runs_csv, run_rows, run_headers)

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for r in run_rows:
        grouped.setdefault((str(r["scenario"]), str(r["method"])), []).append(r)

    cost_rows: List[Dict[str, object]] = []
    ops_rows: List[Dict[str, object]] = []
    for scenario, method in sorted(grouped.keys()):
        rows = grouped[(scenario, method)]
        model_tag = str(rows[0]["model_tag"])
        method_label = str(rows[0]["method_label"])

        cb_m, cb_s = _mean_std([float(x["cost_best_mean"]) for x in rows])
        cm_m, cm_s = _mean_std([float(x["cost_mean_mean"]) for x in rows])
        cw_m, cw_s = _mean_std([float(x["cost_worst_mean"]) for x in rows])
        cost_rows.append(
            {
                "scenario": scenario,
                "model_tag": model_tag,
                "method": method,
                "method_label": method_label,
                "seeds": len(rows),
                "cost_best_mean": cb_m,
                "cost_best_std": cb_s,
                "cost_mean_mean": cm_m,
                "cost_mean_std": cm_s,
                "cost_worst_mean": cw_m,
                "cost_worst_std": cw_s,
            }
        )

        ac_m, ac_s = _mean_std([float(x["accept_rate"]) for x in rows])
        rc_m, rc_s = _mean_std([float(x["reject_rate"]) for x in rows])
        ot_m, ot_s = _mean_std([float(x["on_time_rate"]) for x in rows])
        la_m, la_s = _mean_std([float(x["avg_lateness"]) for x in rows])
        rv_m, rv_s = _mean_std([float(x["total_revenue"]) for x in rows])
        en_m, en_s = _mean_std([float(x["total_energy"]) for x in rows])
        ops_rows.append(
            {
                "scenario": scenario,
                "model_tag": model_tag,
                "method": method,
                "method_label": method_label,
                "seeds": len(rows),
                "accept_rate_mean": ac_m,
                "accept_rate_std": ac_s,
                "reject_rate_mean": rc_m,
                "reject_rate_std": rc_s,
                "on_time_rate_mean": ot_m,
                "on_time_rate_std": ot_s,
                "avg_lateness_mean": la_m,
                "avg_lateness_std": la_s,
                "total_revenue_mean": rv_m,
                "total_revenue_std": rv_s,
                "total_energy_mean": en_m,
                "total_energy_std": en_s,
            }
        )

    _write_csv(
        out_dir / "table_cost.csv",
        cost_rows,
        [
            "scenario",
            "model_tag",
            "method",
            "method_label",
            "seeds",
            "cost_best_mean",
            "cost_best_std",
            "cost_mean_mean",
            "cost_mean_std",
            "cost_worst_mean",
            "cost_worst_std",
        ],
    )
    _write_csv(
        out_dir / "table_ops.csv",
        ops_rows,
        [
            "scenario",
            "model_tag",
            "method",
            "method_label",
            "seeds",
            "accept_rate_mean",
            "accept_rate_std",
            "reject_rate_mean",
            "reject_rate_std",
            "on_time_rate_mean",
            "on_time_rate_std",
            "avg_lateness_mean",
            "avg_lateness_std",
            "total_revenue_mean",
            "total_revenue_std",
            "total_energy_mean",
            "total_energy_std",
        ],
    )

    cost_lookup = {(str(x["scenario"]), str(x["method"])): x for x in cost_rows}
    ops_lookup = {(str(x["scenario"]), str(x["method"])): x for x in ops_rows}
    full_main_cost = cost_lookup.get(("full_main", "model"))
    full_main_ops = ops_lookup.get(("full_main", "model"))
    ablation_rows: List[Dict[str, object]] = []
    for key in sorted(cost_lookup.keys()):
        scenario, method = key
        if not scenario.startswith("ablate_"):
            continue
        if method != "model":
            continue
        cost_row = dict(cost_lookup[key])
        ops_row = ops_lookup.get(key, {})
        row = {
            "scenario": scenario,
            "model_tag": cost_row["model_tag"],
            "seeds": cost_row["seeds"],
            "cost_best_mean": cost_row["cost_best_mean"],
            "cost_best_std": cost_row["cost_best_std"],
            "cost_mean_mean": cost_row["cost_mean_mean"],
            "cost_mean_std": cost_row["cost_mean_std"],
            "cost_worst_mean": cost_row["cost_worst_mean"],
            "cost_worst_std": cost_row["cost_worst_std"],
            "accept_rate_mean": ops_row.get("accept_rate_mean", 0.0),
            "accept_rate_std": ops_row.get("accept_rate_std", 0.0),
            "on_time_rate_mean": ops_row.get("on_time_rate_mean", 0.0),
            "on_time_rate_std": ops_row.get("on_time_rate_std", 0.0),
            "avg_lateness_mean": ops_row.get("avg_lateness_mean", 0.0),
            "avg_lateness_std": ops_row.get("avg_lateness_std", 0.0),
            "total_revenue_mean": ops_row.get("total_revenue_mean", 0.0),
            "total_revenue_std": ops_row.get("total_revenue_std", 0.0),
            "total_energy_mean": ops_row.get("total_energy_mean", 0.0),
            "total_energy_std": ops_row.get("total_energy_std", 0.0),
        }
        if full_main_cost is not None:
            row["delta_cost_best_vs_full_main"] = float(cost_row["cost_best_mean"]) - float(full_main_cost["cost_best_mean"])
            row["delta_cost_mean_vs_full_main"] = float(cost_row["cost_mean_mean"]) - float(full_main_cost["cost_mean_mean"])
            row["delta_cost_worst_vs_full_main"] = float(cost_row["cost_worst_mean"]) - float(full_main_cost["cost_worst_mean"])
        else:
            row["delta_cost_best_vs_full_main"] = 0.0
            row["delta_cost_mean_vs_full_main"] = 0.0
            row["delta_cost_worst_vs_full_main"] = 0.0
        if full_main_ops is not None:
            row["delta_accept_rate_vs_full_main"] = float(row["accept_rate_mean"]) - float(full_main_ops["accept_rate_mean"])
            row["delta_on_time_rate_vs_full_main"] = float(row["on_time_rate_mean"]) - float(full_main_ops["on_time_rate_mean"])
            row["delta_avg_lateness_vs_full_main"] = float(row["avg_lateness_mean"]) - float(
                full_main_ops["avg_lateness_mean"]
            )
            row["delta_total_revenue_vs_full_main"] = float(row["total_revenue_mean"]) - float(
                full_main_ops["total_revenue_mean"]
            )
            row["delta_total_energy_vs_full_main"] = float(row["total_energy_mean"]) - float(
                full_main_ops["total_energy_mean"]
            )
        else:
            row["delta_accept_rate_vs_full_main"] = 0.0
            row["delta_on_time_rate_vs_full_main"] = 0.0
            row["delta_avg_lateness_vs_full_main"] = 0.0
            row["delta_total_revenue_vs_full_main"] = 0.0
            row["delta_total_energy_vs_full_main"] = 0.0
        ablation_rows.append(row)
    _write_csv(
        out_dir / "table_ablation.csv",
        ablation_rows,
        [
            "scenario",
            "model_tag",
            "seeds",
            "cost_best_mean",
            "cost_best_std",
            "cost_mean_mean",
            "cost_mean_std",
            "cost_worst_mean",
            "cost_worst_std",
            "delta_cost_best_vs_full_main",
            "delta_cost_mean_vs_full_main",
            "delta_cost_worst_vs_full_main",
            "accept_rate_mean",
            "accept_rate_std",
            "delta_accept_rate_vs_full_main",
            "on_time_rate_mean",
            "on_time_rate_std",
            "delta_on_time_rate_vs_full_main",
            "avg_lateness_mean",
            "avg_lateness_std",
            "delta_avg_lateness_vs_full_main",
            "total_revenue_mean",
            "total_revenue_std",
            "delta_total_revenue_vs_full_main",
            "total_energy_mean",
            "total_energy_std",
            "delta_total_energy_vs_full_main",
        ],
    )

    summary = {
        "output_dir": str(out_dir),
        "elapsed_sec": float(time.time() - t0),
        "runs": len(run_rows),
        "seeds": seeds,
        "scenarios": [str(x["scenario"]) for x in scenarios],
        "methods": METHOD_ORDER,
        "files": {
            "config_json": str((out_dir / "config.json").resolve()),
            "runs_csv": str((out_dir / "runs.csv").resolve()),
            "table_cost_csv": str((out_dir / "table_cost.csv").resolve()),
            "table_ops_csv": str((out_dir / "table_ops.csv").resolve()),
            "table_ablation_csv": str((out_dir / "table_ablation.csv").resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nProtocol complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
