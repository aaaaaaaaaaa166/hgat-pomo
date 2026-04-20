from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from common import dump_json, dump_text, load_json, repo_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a workspace-level comparison summary.")
    parser.add_argument(
        "--results-root",
        type=str,
        default="external_methods/results",
    )
    parser.add_argument(
        "--hgat-summary",
        type=str,
        default="external_methods/results/hgat_pomo_reference/summary.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="external_methods/results/comparison_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="external_methods/results/comparison_summary.md",
    )
    return parser.parse_args()


def gather_external_metrics(results_root: Path) -> List[Dict[str, object]]:
    metrics_records: List[Dict[str, object]] = []
    for path in sorted(results_root.glob("**/metrics.json")):
        if "hgat_pomo_reference" in path.parts:
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        payload["_path"] = repo_relative(path)
        metrics_records.append(payload)
    return metrics_records


def build_markdown(hgat_summary: Dict[str, object], external_metrics: List[Dict[str, object]]) -> str:
    full_main_cost_rank = hgat_summary["formal_protocol"]["derived"]["full_main_rank_by_cost_mean"]
    lines = [
        "# Workspace Comparison Summary",
        "",
        "## HGAT-POMO dynamic thesis protocol",
        "| Rank | Method | full_main cost_mean_mean | on_time_rate_mean | accept_rate_mean |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    ops_by_method = {
        row["method"]: row
        for row in hgat_summary["formal_protocol"]["tables"]["ops_rows"]
        if row["scenario"] == "full_main"
    }
    for idx, row in enumerate(full_main_cost_rank, start=1):
        ops = ops_by_method.get(row["method"], {})
        lines.append(
            f"| {idx} | {row['method_label']} | {float(row['cost_mean_mean']):.3f} | "
            f"{float(ops.get('on_time_rate_mean', 0.0)):.4f} | {float(ops.get('accept_rate_mean', 0.0)):.4f} |"
        )

    lines.extend(
        [
            "",
            "## External static CVRP runs",
            "These runs use the same open CVRPLIB split files and the same sampler, but on a frozen static CVRP proxy.",
            "",
            "| Method | Run Path | Key Test Metric | Extra |",
            "| --- | --- | --- | --- |",
        ]
    )
    if not external_metrics:
        lines.append("| - | - | - | No external metrics found |")
    else:
        for payload in external_metrics:
            method = payload.get("method", "<unknown>")
            run_path = payload.get("_path", "")
            if "test_metrics" in payload:
                metric = f"aug_avg_cost={payload['test_metrics'].get('aug_avg_cost', 'n/a')}"
                extra = f"no_aug_avg_cost={payload['test_metrics'].get('no_aug_avg_cost', 'n/a')}"
            else:
                metric = f"test_avg_cost={payload.get('test_avg_cost', 'n/a')}"
                extra = f"best_val_cost={payload.get('best_val_cost', 'n/a')}"
            lines.append(f"| {method} | `{run_path}` | {metric} | {extra} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    hgat_summary = load_json(args.hgat_summary)
    external_metrics = gather_external_metrics(results_root)

    summary = {
        "hgat_reference_summary": repo_relative(args.hgat_summary),
        "external_metric_files": [payload["_path"] for payload in external_metrics],
        "external_metrics": external_metrics,
    }
    dump_json(summary, args.output_json)
    dump_text(build_markdown(hgat_summary, external_metrics), args.output_md)
    print(f"Wrote comparison summary to: {args.output_md}")


if __name__ == "__main__":
    main()
