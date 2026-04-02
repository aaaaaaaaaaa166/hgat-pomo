from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_VARIANTS = [
    "full",
    "no_traffic",
    "no_time_window",
    "no_soc",
    "no_curriculum",
]


def parse_csv_items(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    items = parse_csv_items(text)
    return [int(x) for x in items]


def format_float(x: float) -> str:
    return f"{x:.6f}"


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / float(len(values))
    var = sum((v - m) * (v - m) for v in values) / float(len(values))
    return m, math.sqrt(max(0.0, var))


def build_cmd(python_exe: str, module: str, args_dict: Dict[str, object]) -> List[str]:
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


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("COMMAND:\n")
        log_f.write(" ".join(cmd))
        log_f.write("\n\nOUTPUT:\n")
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
            log_f.write(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def variant_overrides(name: str) -> Dict[str, object]:
    if name == "full":
        return {}
    if name == "no_traffic":
        return {"traffic_sigma": 0.0}
    if name == "no_time_window":
        return {"tw_mode": "none", "lateness_penalty": 0.0}
    if name == "no_soc":
        return {"soc_reserve": 0.0, "energy_per_dist": 0.0}
    if name == "no_curriculum":
        return {"use_curriculum": False}
    raise ValueError(f"Unknown variant: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations and export aggregate tables.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--output-dir", type=str, default="experiments/ablation")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-train-if-exists", action="store_true")

    # training scale
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-curriculum", type=int, default=1, choices=[0, 1])
    parser.add_argument("--curriculum-start-n", type=int, default=10)

    # evaluation scale
    parser.add_argument("--eval-instances", type=int, default=100)
    parser.add_argument("--eval-no-store-traj", action="store_true")
    parser.add_argument("--eval-extra-baselines", action="store_true")

    # model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k-nn-orders", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.0)

    # data generation
    parser.add_argument("--coord-scale", type=float, default=10.0)
    parser.add_argument("--release-mode", type=str, default="batches", choices=["batches", "uniform", "poisson"])
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--max-release", type=float, default=10.0)
    parser.add_argument("--poisson-rate", type=float, default=1.0)
    parser.add_argument("--tw-mode", type=str, default="relative", choices=["relative", "mixed", "none"])
    parser.add_argument("--tw-slack-low", type=float, default=4.0)
    parser.add_argument("--tw-slack-high", type=float, default=14.0)
    parser.add_argument("--tw-active-prob", type=float, default=0.8)

    # env
    parser.add_argument("--vT", type=float, default=1.0)
    parser.add_argument("--vD", type=float, default=1.5)
    parser.add_argument("--QD", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=6.0)
    parser.add_argument("--traffic-sigma", type=float, default=0.15)
    parser.add_argument("--lateness-penalty", type=float, default=0.5)
    parser.add_argument("--soc-init", type=float, default=1.0)
    parser.add_argument("--soc-reserve", type=float, default=0.10)
    parser.add_argument("--energy-per-dist", type=float, default=0.08)
    parser.add_argument("--recharge-rate", type=float, default=0.25)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.output_dir).resolve()
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    variants = parse_csv_items(args.variants)
    for v in variants:
        if v not in DEFAULT_VARIANTS:
            raise ValueError(f"Unsupported variant: {v}. Allowed={DEFAULT_VARIANTS}")

    base_train = {
        "N": args.N,
        "K": args.K,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "k_nn_orders": args.k_nn_orders,
        "encoder_layers": args.encoder_layers,
        "tanh_clipping": args.tanh_clipping,
        "temperature": args.temperature,
        "temperature_end": args.temperature_end,
        "entropy_coef": args.entropy_coef,
        "entropy_coef_end": args.entropy_coef_end,
        "coord_scale": args.coord_scale,
        "release_mode": args.release_mode,
        "n_batches": args.n_batches,
        "max_release": args.max_release,
        "poisson_rate": args.poisson_rate,
        "tw_mode": args.tw_mode,
        "tw_slack_low": args.tw_slack_low,
        "tw_slack_high": args.tw_slack_high,
        "tw_active_prob": args.tw_active_prob,
        "vT": args.vT,
        "vD": args.vD,
        "QD": args.QD,
        "B": args.B,
        "traffic_sigma": args.traffic_sigma,
        "lateness_penalty": args.lateness_penalty,
        "soc_init": args.soc_init,
        "soc_reserve": args.soc_reserve,
        "energy_per_dist": args.energy_per_dist,
        "recharge_rate": args.recharge_rate,
        "use_curriculum": bool(args.use_curriculum),
        "curriculum_start_n": args.curriculum_start_n,
    }
    base_eval = {
        "N": args.N,
        "K": args.K,
        "n_instances": args.eval_instances,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "dropout": args.dropout,
        "k_nn_orders": args.k_nn_orders,
        "encoder_layers": args.encoder_layers,
        "tanh_clipping": args.tanh_clipping,
        "temperature": args.temperature,
        "coord_scale": args.coord_scale,
        "release_mode": args.release_mode,
        "n_batches": args.n_batches,
        "max_release": args.max_release,
        "poisson_rate": args.poisson_rate,
        "tw_mode": args.tw_mode,
        "tw_slack_low": args.tw_slack_low,
        "tw_slack_high": args.tw_slack_high,
        "tw_active_prob": args.tw_active_prob,
        "vT": args.vT,
        "vD": args.vD,
        "QD": args.QD,
        "B": args.B,
        "traffic_sigma": args.traffic_sigma,
        "lateness_penalty": args.lateness_penalty,
        "soc_init": args.soc_init,
        "soc_reserve": args.soc_reserve,
        "energy_per_dist": args.energy_per_dist,
        "recharge_rate": args.recharge_rate,
    }
    if args.eval_no_store_traj:
        base_eval["no_store_traj"] = True
    if args.eval_extra_baselines:
        base_eval["extra_baselines"] = True

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_root": str(repo_root),
                "args": vars(args),
                "variants": variants,
                "seeds": seeds,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    run_rows: List[Dict[str, object]] = []
    fail_rows: List[Dict[str, object]] = []

    total_runs = len(variants) * len(seeds)
    run_idx = 0
    for variant in variants:
        overrides = variant_overrides(variant)
        for seed in seeds:
            run_idx += 1
            print("\n" + "=" * 80)
            print(f"[{run_idx}/{total_runs}] variant={variant} seed={seed}")
            print("=" * 80)

            train_cfg = dict(base_train)
            eval_cfg = dict(base_eval)
            train_cfg.update(overrides)
            eval_cfg.update(overrides)

            model_path = models_dir / f"{variant}_seed{seed}.pt"
            metrics_path = metrics_dir / f"{variant}_seed{seed}.json"
            train_log = logs_dir / f"{variant}_seed{seed}_train.log"
            eval_log = logs_dir / f"{variant}_seed{seed}_eval.log"

            train_cfg["seed"] = seed
            train_cfg["save_path"] = str(model_path)
            eval_cfg["model_path"] = str(model_path)
            eval_cfg["metrics_json"] = str(metrics_path)
            eval_cfg["traj_out"] = str(metrics_dir / f"{variant}_seed{seed}_trajs.txt")

            try:
                if args.skip_train_if_exists and model_path.exists():
                    print(f"Skip training (model exists): {model_path}")
                    train_elapsed = 0.0
                else:
                    train_cmd = build_cmd(args.python_exe, "src.main_train", train_cfg)
                    t_train = time.time()
                    run_cmd(train_cmd, cwd=repo_root, log_path=train_log)
                    train_elapsed = float(time.time() - t_train)

                eval_cmd = build_cmd(args.python_exe, "src.main_eval", eval_cfg)
                t_eval = time.time()
                run_cmd(eval_cmd, cwd=repo_root, log_path=eval_log)
                eval_elapsed = float(time.time() - t_eval)

                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)

                model_best_mean = float(m["model"]["best"]["mean"])
                random_best_mean = float(m["random"]["best"]["mean"])
                improve_pct = (random_best_mean - model_best_mean) / max(1e-9, random_best_mean) * 100.0
                row = {
                    "variant": variant,
                    "seed": seed,
                    "train_sec": train_elapsed,
                    "eval_sec": eval_elapsed,
                    "model_best_mean": model_best_mean,
                    "model_mean_mean": float(m["model"]["mean"]["mean"]),
                    "model_worst_mean": float(m["model"]["worst"]["mean"]),
                    "random_best_mean": random_best_mean,
                    "random_mean_mean": float(m["random"]["mean"]["mean"]),
                    "random_worst_mean": float(m["random"]["worst"]["mean"]),
                    "improve_best_pct": improve_pct,
                    "metrics_path": str(metrics_path),
                    "model_path": str(model_path),
                }
                run_rows.append(row)
                print(
                    f"Completed variant={variant} seed={seed} "
                    f"| model_best={model_best_mean:.3f} rand_best={random_best_mean:.3f} "
                    f"| improve={improve_pct:.2f}%"
                )
            except Exception as exc:
                fail_info = {
                    "variant": variant,
                    "seed": seed,
                    "error": str(exc),
                }
                fail_rows.append(fail_info)
                print(f"FAILED variant={variant} seed={seed}: {exc}")
                if not args.continue_on_error:
                    raise

    runs_csv = out_dir / "runs.csv"
    run_fields = [
        "variant",
        "seed",
        "train_sec",
        "eval_sec",
        "model_best_mean",
        "model_mean_mean",
        "model_worst_mean",
        "random_best_mean",
        "random_mean_mean",
        "random_worst_mean",
        "improve_best_pct",
        "metrics_path",
        "model_path",
    ]
    with open(runs_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        for row in run_rows:
            out = dict(row)
            for k in [
                "train_sec",
                "eval_sec",
                "model_best_mean",
                "model_mean_mean",
                "model_worst_mean",
                "random_best_mean",
                "random_mean_mean",
                "random_worst_mean",
                "improve_best_pct",
            ]:
                out[k] = format_float(float(out[k]))
            writer.writerow(out)

    summary_rows: List[Dict[str, object]] = []
    numeric_keys = [
        "train_sec",
        "eval_sec",
        "model_best_mean",
        "model_mean_mean",
        "model_worst_mean",
        "random_best_mean",
        "random_mean_mean",
        "random_worst_mean",
        "improve_best_pct",
    ]
    for variant in variants:
        subset = [r for r in run_rows if r["variant"] == variant]
        if not subset:
            continue
        row = {"variant": variant, "n_runs": len(subset)}
        for key in numeric_keys:
            vals = [float(x[key]) for x in subset]
            m, s = mean_std(vals)
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = s
        summary_rows.append(row)

    summary_csv = out_dir / "summary.csv"
    summary_fields = ["variant", "n_runs"] + [f"{k}_{p}" for k in numeric_keys for p in ("mean", "std")]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            out = dict(row)
            for key in summary_fields:
                if key in ("variant", "n_runs"):
                    continue
                out[key] = format_float(float(out[key]))
            writer.writerow(out)

    summary_md = out_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Ablation Summary\n\n")
        f.write(f"- Total successful runs: {len(run_rows)}\n")
        f.write(f"- Total failed runs: {len(fail_rows)}\n\n")
        f.write("| Variant | Runs | Model Best (mean+/-std) | Random Best (mean+/-std) | Improve % (mean+/-std) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['variant']} | {row['n_runs']} | "
                f"{row['model_best_mean_mean']:.3f}+/-{row['model_best_mean_std']:.3f} | "
                f"{row['random_best_mean_mean']:.3f}+/-{row['random_best_mean_std']:.3f} | "
                f"{row['improve_best_pct_mean']:.2f}+/-{row['improve_best_pct_std']:.2f} |\n"
            )
        if fail_rows:
            f.write("\n## Failures\n\n")
            for fr in fail_rows:
                f.write(f"- variant={fr['variant']} seed={fr['seed']} error={fr['error']}\n")

    if fail_rows:
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(fail_rows, f, ensure_ascii=False, indent=2)

    print("\nFinished ablation suite.")
    print(f"runs: {runs_csv}")
    print(f"summary: {summary_csv}")
    print(f"table: {summary_md}")
    if fail_rows:
        print(f"failures: {out_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
