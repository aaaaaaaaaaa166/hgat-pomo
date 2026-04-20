from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

from common import dump_json, dump_text, load_json, repo_relative


TRAIN_LOG_PATTERN = re.compile(
    r"\[ep=(?P<epoch>\d+)\]\s+loss=(?P<loss>-?\d+\.\d+)\s+"
    r"cost_mean=(?P<cost_mean>-?\d+\.\d+)\s+cost_best=(?P<cost_best>-?\d+\.\d+)\s+"
    r"entropy=(?P<entropy>-?\d+\.\d+).+N=(?P<N>\d+)\s+B=(?P<B>\d+)\s+K=(?P<K>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect current HGAT-POMO results into one reference summary.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_methods/results/hgat_pomo_reference",
        help="Directory to write summary.json and summary.md.",
    )
    parser.add_argument(
        "--frozen-manifest",
        type=str,
        default="experiments/frozen_models_20260419/manifest.json",
    )
    parser.add_argument(
        "--train-log",
        type=str,
        default="experiments/retrain_open_20260419/logs/train_stdout.log",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="experiments/retrain_open_20260419/models",
    )
    parser.add_argument(
        "--formal-summary",
        type=str,
        default="experiments/thesis_protocol_20260420_formal/summary.json",
    )
    parser.add_argument(
        "--cost-table",
        type=str,
        default="experiments/thesis_protocol_20260420_formal/table_cost.csv",
    )
    parser.add_argument(
        "--ops-table",
        type=str,
        default="experiments/thesis_protocol_20260420_formal/table_ops.csv",
    )
    parser.add_argument(
        "--ablation-table",
        type=str,
        default="experiments/thesis_protocol_20260420_formal/table_ablation.csv",
    )
    parser.add_argument(
        "--split-summary",
        type=str,
        default="datasets/cvrplib/splits/summary.json",
    )
    return parser.parse_args()


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_training_log(path: str | Path) -> Dict[str, object]:
    records = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_LOG_PATTERN.search(raw_line)
        if not match:
            continue
        records.append(
            {
                "epoch": int(match.group("epoch")),
                "loss": float(match.group("loss")),
                "cost_mean": float(match.group("cost_mean")),
                "cost_best": float(match.group("cost_best")),
                "entropy": float(match.group("entropy")),
                "N": int(match.group("N")),
                "B": int(match.group("B")),
                "K": int(match.group("K")),
            }
        )
    if not records:
        raise ValueError(f"No epoch records found in log: {path}")

    best_by_cost_best = min(records, key=lambda item: item["cost_best"])
    best_by_cost_mean = min(records, key=lambda item: item["cost_mean"])
    return {
        "epochs_logged": len(records),
        "first_epoch": records[0],
        "last_epoch": records[-1],
        "best_cost_best_epoch": best_by_cost_best,
        "best_cost_mean_epoch": best_by_cost_mean,
    }


def summarize_checkpoints(path: str | Path) -> Dict[str, object]:
    checkpoint_paths = sorted(Path(path).glob("*.ep*.pt"))
    return {
        "count": len(checkpoint_paths),
        "files": [repo_relative(p) for p in checkpoint_paths],
    }


def summarize_formal_tables(cost_rows: List[Dict[str, str]], ops_rows: List[Dict[str, str]]) -> Dict[str, object]:
    full_main_cost = [row for row in cost_rows if row["scenario"] == "full_main"]
    full_backup_cost = [row for row in cost_rows if row["scenario"] == "full_backup"]
    full_main_cost_sorted = sorted(full_main_cost, key=lambda row: float(row["cost_mean_mean"]))
    full_backup_cost_sorted = sorted(full_backup_cost, key=lambda row: float(row["cost_mean_mean"]))

    full_main_ops = [row for row in ops_rows if row["scenario"] == "full_main"]
    full_main_ops_sorted = sorted(full_main_ops, key=lambda row: float(row["on_time_rate_mean"]), reverse=True)
    full_main_accept_sorted = sorted(full_main_ops, key=lambda row: float(row["accept_rate_mean"]), reverse=True)

    return {
        "full_main_rank_by_cost_mean": full_main_cost_sorted,
        "full_backup_rank_by_cost_mean": full_backup_cost_sorted,
        "full_main_rank_by_on_time_rate": full_main_ops_sorted,
        "full_main_rank_by_accept_rate": full_main_accept_sorted,
    }


def build_markdown(summary: Dict[str, object]) -> str:
    frozen = summary["frozen_models"]
    train = summary["open_retrain"]
    formal = summary["formal_protocol"]
    split_summary = summary["open_dataset_split"]

    full_main_cost_rank = formal["derived"]["full_main_rank_by_cost_mean"]
    full_main_on_time_rank = formal["derived"]["full_main_rank_by_on_time_rate"]
    full_main_accept_rank = formal["derived"]["full_main_rank_by_accept_rate"]

    lines = [
        "# HGAT-POMO Reference Summary",
        "",
        "## Frozen models",
        f"- Main model: `{frozen['main_model']['path']}`",
        f"- Backup model: `{frozen['backup_model']['path']}`",
        "",
        "## Open-data training",
        f"- Logged epochs: {train['log_summary']['epochs_logged']}",
        f"- Best `cost_best` epoch: ep{train['log_summary']['best_cost_best_epoch']['epoch']:04d} "
        f"({train['log_summary']['best_cost_best_epoch']['cost_best']:.3f})",
        f"- Best `cost_mean` epoch: ep{train['log_summary']['best_cost_mean_epoch']['epoch']:04d} "
        f"({train['log_summary']['best_cost_mean_epoch']['cost_mean']:.3f})",
        f"- Final epoch: ep{train['log_summary']['last_epoch']['epoch']:04d} "
        f"(cost_mean={train['log_summary']['last_epoch']['cost_mean']:.3f}, "
        f"cost_best={train['log_summary']['last_epoch']['cost_best']:.3f})",
        f"- Checkpoints: {train['checkpoints']['count']}",
        "",
        "## Open split coverage",
        f"- Eligible instances: {split_summary['eligible_instances']}",
        f"- Families: {', '.join(split_summary['families'])}",
        f"- Split sizes: train={split_summary['split_sizes']['train']}, "
        f"val={split_summary['split_sizes']['val']}, test={split_summary['split_sizes']['test']}",
        "",
        "## Formal protocol",
        f"- Total runs: {formal['summary']['runs']}",
        f"- Seeds: {', '.join(str(x) for x in formal['summary']['seeds'])}",
        f"- Scenarios: {', '.join(formal['summary']['scenarios'])}",
        "",
        "### Full Main cost ranking",
        "| Rank | Method | cost_mean_mean | cost_best_mean | cost_worst_mean |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(full_main_cost_rank, start=1):
        lines.append(
            f"| {idx} | {row['method_label']} | {float(row['cost_mean_mean']):.3f} | "
            f"{float(row['cost_best_mean']):.3f} | {float(row['cost_worst_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "### Full Main on-time ranking",
            "| Rank | Method | on_time_rate_mean | accept_rate_mean | total_energy_mean | total_revenue_mean |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(full_main_on_time_rank, start=1):
        lines.append(
            f"| {idx} | {row['method_label']} | {float(row['on_time_rate_mean']):.4f} | "
            f"{float(row['accept_rate_mean']):.4f} | {float(row['total_energy_mean']):.3f} | "
            f"{float(row['total_revenue_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "### Full Main accept-rate ranking",
            "| Rank | Method | accept_rate_mean | on_time_rate_mean | avg_lateness_mean |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(full_main_accept_rank, start=1):
        lines.append(
            f"| {idx} | {row['method_label']} | {float(row['accept_rate_mean']):.4f} | "
            f"{float(row['on_time_rate_mean']):.4f} | {float(row['avg_lateness_mean']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Source files",
            f"- Frozen manifest: `{summary['files']['frozen_manifest']}`",
            f"- Train log: `{summary['files']['train_log']}`",
            f"- Formal summary: `{summary['files']['formal_summary']}`",
            f"- Cost table: `{summary['files']['cost_table']}`",
            f"- Ops table: `{summary['files']['ops_table']}`",
            f"- Ablation table: `{summary['files']['ablation_table']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_manifest = load_json(args.frozen_manifest)
    split_summary = load_json(args.split_summary)
    formal_summary = load_json(args.formal_summary)
    cost_rows = read_csv_rows(args.cost_table)
    ops_rows = read_csv_rows(args.ops_table)
    ablation_rows = read_csv_rows(args.ablation_table)

    summary = {
        "files": {
            "frozen_manifest": repo_relative(args.frozen_manifest),
            "train_log": repo_relative(args.train_log),
            "formal_summary": repo_relative(args.formal_summary),
            "cost_table": repo_relative(args.cost_table),
            "ops_table": repo_relative(args.ops_table),
            "ablation_table": repo_relative(args.ablation_table),
            "split_summary": repo_relative(args.split_summary),
        },
        "frozen_models": frozen_manifest,
        "open_dataset_split": split_summary,
        "open_retrain": {
            "log_summary": summarize_training_log(args.train_log),
            "checkpoints": summarize_checkpoints(args.checkpoints_dir),
        },
        "formal_protocol": {
            "summary": formal_summary,
            "tables": {
                "cost_rows": cost_rows,
                "ops_rows": ops_rows,
                "ablation_rows": ablation_rows,
            },
            "derived": summarize_formal_tables(cost_rows, ops_rows),
        },
    }

    dump_json(summary, output_dir / "summary.json")
    dump_text(build_markdown(summary), output_dir / "summary.md")
    print(f"Wrote HGAT reference summary to: {output_dir}")


if __name__ == "__main__":
    main()
