from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_csv_floats(text: str) -> List[float]:
    out: List[float] = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(float(s))
    if len(out) == 0:
        raise ValueError("float grid list must not be empty")
    return out


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    if len(out) == 0:
        raise ValueError("int grid list must not be empty")
    return out


def _build_cmd(python_exe: str, module: str, args_dict: Dict[str, object]) -> List[str]:
    cmd = [python_exe, "-m", module]
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


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / float(len(values))
    v = sum((x - m) * (x - m) for x in values) / float(len(values))
    return m, math.sqrt(max(v, 0.0))


def _write_csv(path: Path, rows: List[Dict[str, object]], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune acceptance-related parameters with open dataset protocol.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-dir", type=str, default="experiments/accept_tuning_20260419")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--reject-penalties", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--response-slack-highs", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--scheduled-ratios", type=str, default="0.4,0.5,0.6")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-instances", type=int, default=100)
    parser.add_argument("--edge-mode", type=str, default="road", choices=["static", "road"])
    parser.add_argument("--time-dependent", action="store_true")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--train-split-file", type=str, required=True)
    parser.add_argument("--val-split-file", type=str, required=True)
    parser.add_argument("--dataset-demand-scale", type=float, default=1.0)
    parser.add_argument("--dataset-no-normalize-coords", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    reject_penalties = _parse_csv_floats(args.reject_penalties)
    response_slack_highs = _parse_csv_floats(args.response_slack_highs)
    scheduled_ratios = _parse_csv_floats(args.scheduled_ratios)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.output_dir).resolve()
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    combos = list(itertools.product(reject_penalties, response_slack_highs, scheduled_ratios))
    total = len(combos) * len(seeds)
    idx = 0
    t0 = time.time()
    for rp, rsh, sr in combos:
        tag = f"rp{rp:g}_rsh{rsh:g}_sr{sr:g}"
        for seed in seeds:
            idx += 1
            model_path = models_dir / f"{tag}_seed{seed}.pt"
            metrics_path = metrics_dir / f"{tag}_seed{seed}.json"
            train_log = logs_dir / f"{tag}_seed{seed}_train.log"
            eval_log = logs_dir / f"{tag}_seed{seed}_eval.log"

            print("\n" + "=" * 88)
            print(f"[{idx}/{total}] {tag} seed={seed}")
            print("=" * 88)

            if not (args.skip_existing and model_path.exists()):
                train_cfg: Dict[str, object] = {
                    "seed": seed,
                    "N": args.N,
                    "K": args.K,
                    "batch_size": args.batch_size,
                    "epochs": args.train_epochs,
                    "lr": args.lr,
                    "save_path": str(model_path),
                    "init_model_path": args.base_model,
                    "edge_mode": args.edge_mode,
                    "time_dependent": args.time_dependent,
                    "dataset_path": args.dataset_path,
                    "dataset_format": "cvrplib",
                    "dataset_split_file": args.train_split_file,
                    "dataset_demand_scale": args.dataset_demand_scale,
                    "dataset_no_normalize_coords": args.dataset_no_normalize_coords,
                    "reject_penalty": rp,
                    "response_slack_high": rsh,
                    "scheduled_ratio": sr,
                    "use_curriculum": True,
                    "curriculum_start_n": 10,
                }
                train_cmd = _build_cmd(args.python_exe, "src.main_train", train_cfg)
                _run_cmd(train_cmd, cwd=repo_root, log_path=train_log)
            else:
                print(f"Skip training (model exists): {model_path}")

            if not (args.skip_existing and metrics_path.exists()):
                eval_cfg: Dict[str, object] = {
                    "model_path": str(model_path),
                    "N": args.N,
                    "K": args.K,
                    "n_instances": args.eval_instances,
                    "edge_mode": args.edge_mode,
                    "time_dependent": args.time_dependent,
                    "dataset_path": args.dataset_path,
                    "dataset_format": "cvrplib",
                    "dataset_split_file": args.val_split_file,
                    "dataset_demand_scale": args.dataset_demand_scale,
                    "dataset_no_normalize_coords": args.dataset_no_normalize_coords,
                    "eval_seed": seed,
                    "no_store_traj": True,
                    "extra_baselines": True,
                    "metrics_json": str(metrics_path),
                    "reject_penalty": rp,
                    "response_slack_high": rsh,
                    "scheduled_ratio": sr,
                }
                eval_cmd = _build_cmd(args.python_exe, "src.main_eval", eval_cfg)
                _run_cmd(eval_cmd, cwd=repo_root, log_path=eval_log)
            else:
                print(f"Skip eval (metrics exists): {metrics_path}")

            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_best = float(m["model"]["best"]["mean"])
            accept_rate = float(m["model_ops"]["accept_rate"]["mean"])
            on_time_rate = float(m["model_ops"]["on_time_rate"]["mean"])
            avg_lateness = float(m["model_ops"]["avg_lateness"]["mean"])
            score = 2.0 * accept_rate + 1.0 * on_time_rate - 0.01 * avg_lateness - 0.001 * model_best
            runs.append(
                {
                    "tag": tag,
                    "seed": int(seed),
                    "reject_penalty": float(rp),
                    "response_slack_high": float(rsh),
                    "scheduled_ratio": float(sr),
                    "model_best_mean": model_best,
                    "accept_rate": accept_rate,
                    "on_time_rate": on_time_rate,
                    "avg_lateness": avg_lateness,
                    "score": score,
                    "model_path": str(model_path),
                    "metrics_path": str(metrics_path),
                }
            )

    run_headers = [
        "tag",
        "seed",
        "reject_penalty",
        "response_slack_high",
        "scheduled_ratio",
        "model_best_mean",
        "accept_rate",
        "on_time_rate",
        "avg_lateness",
        "score",
        "model_path",
        "metrics_path",
    ]
    _write_csv(out_dir / "runs.csv", runs, run_headers)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in runs:
        grouped.setdefault(str(r["tag"]), []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for tag, rows in grouped.items():
        cfg = rows[0]
        best_m, best_s = _mean_std([float(x["model_best_mean"]) for x in rows])
        acc_m, acc_s = _mean_std([float(x["accept_rate"]) for x in rows])
        ot_m, ot_s = _mean_std([float(x["on_time_rate"]) for x in rows])
        lat_m, lat_s = _mean_std([float(x["avg_lateness"]) for x in rows])
        sc_m, sc_s = _mean_std([float(x["score"]) for x in rows])
        summary_rows.append(
            {
                "tag": tag,
                "seeds": len(rows),
                "reject_penalty": float(cfg["reject_penalty"]),
                "response_slack_high": float(cfg["response_slack_high"]),
                "scheduled_ratio": float(cfg["scheduled_ratio"]),
                "model_best_mean_mean": best_m,
                "model_best_mean_std": best_s,
                "accept_rate_mean": acc_m,
                "accept_rate_std": acc_s,
                "on_time_rate_mean": ot_m,
                "on_time_rate_std": ot_s,
                "avg_lateness_mean": lat_m,
                "avg_lateness_std": lat_s,
                "score_mean": sc_m,
                "score_std": sc_s,
            }
        )
    summary_rows.sort(key=lambda x: float(x["score_mean"]), reverse=True)
    _write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "tag",
            "seeds",
            "reject_penalty",
            "response_slack_high",
            "scheduled_ratio",
            "model_best_mean_mean",
            "model_best_mean_std",
            "accept_rate_mean",
            "accept_rate_std",
            "on_time_rate_mean",
            "on_time_rate_std",
            "avg_lateness_mean",
            "avg_lateness_std",
            "score_mean",
            "score_std",
        ],
    )

    best_cfg = summary_rows[0] if summary_rows else {}
    summary = {
        "output_dir": str(out_dir),
        "elapsed_sec": float(time.time() - t0),
        "runs": len(runs),
        "combos": len(combos),
        "seeds": seeds,
        "best_config": best_cfg,
        "files": {
            "runs_csv": str((out_dir / "runs.csv").resolve()),
            "summary_csv": str((out_dir / "summary.csv").resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAcceptance tuning complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
