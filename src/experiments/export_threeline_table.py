from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, List


VARIANT_LABELS = {
    "full": "Full Model",
    "no_traffic": "w/o Traffic",
    "no_time_window": "w/o Time Window",
    "no_soc": "w/o SOC Constraint",
    "no_curriculum": "w/o Curriculum",
}


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def fmt_mean_std(mean_str: str, std_str: str, digits: int) -> str:
    m = float(mean_str)
    s = float(std_str)
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def build_latex(
    rows: List[Dict[str, str]],
    caption: str,
    label: str,
    digits_cost: int,
    digits_pct: int,
    sort_by: str,
) -> str:
    if sort_by == "variant":
        rows = sorted(rows, key=lambda x: x["variant"])
    elif sort_by == "model_best":
        rows = sorted(rows, key=lambda x: float(x["model_best_mean_mean"]))

    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{tabular}{lccc}")
    lines.append("    \\toprule")
    lines.append("    Variant & Model Best (mean$\\pm$std) & Random Best (mean$\\pm$std) & Improve\\% (mean$\\pm$std) \\\\")
    lines.append("    \\midrule")

    for r in rows:
        v_key = r["variant"]
        v = VARIANT_LABELS.get(v_key, v_key)
        model_best = fmt_mean_std(r["model_best_mean_mean"], r["model_best_mean_std"], digits_cost)
        rand_best = fmt_mean_std(r["random_best_mean_mean"], r["random_best_mean_std"], digits_cost)
        improve = fmt_mean_std(r["improve_best_pct_mean"], r["improve_best_pct_std"], digits_pct)
        lines.append(f"    {v} & {model_best} & {rand_best} & {improve} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis three-line (booktabs) table from summary.csv.")
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--caption", type=str, default="Ablation Study Results")
    parser.add_argument("--label", type=str, default="tab:ablation_results")
    parser.add_argument("--digits-cost", type=int, default=3)
    parser.add_argument("--digits-pct", type=int, default=2)
    parser.add_argument("--sort-by", type=str, default="variant", choices=["variant", "model_best"])
    parser.add_argument("--print", action="store_true", dest="do_print")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv).resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_csv}")

    rows = read_rows(summary_csv)
    if not rows:
        raise ValueError(f"summary csv has no rows: {summary_csv}")

    latex = build_latex(
        rows=rows,
        caption=args.caption,
        label=args.label,
        digits_cost=args.digits_cost,
        digits_pct=args.digits_pct,
        sort_by=args.sort_by,
    )

    out_path = Path(args.out).resolve() if args.out else summary_csv.with_name("paper_table.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    if args.do_print:
        print(latex, end="")
    print(f"Saved LaTeX three-line table to {out_path}")


if __name__ == "__main__":
    main()
