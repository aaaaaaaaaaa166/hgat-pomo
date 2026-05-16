from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METHOD_ORDER = {
    "raw_baseline": 0,
    "v2_repair_only": 1,
    "tail_risk_constrained_joint_beam": 2,
    "oracle_best_acceptance": 3,
    "oracle_best_on_time": 4,
}

CONFIG_ORDER = {
    "combined_D": 0,
    "combined_E": 1,
    "combined_F": 2,
    "combined_G": 3,
}

SUMMARY_FIELDS = [
    "experiment_name",
    "method_name",
    "eval_instances",
    "response_window",
    "delivery_window_extension",
    "resource_count",
    "order_density_ratio",
    "acceptance_rate",
    "acceptance_rate_gap_to_0_80",
    "on_time_rate",
    "on_time_rate_gap_to_0_50",
    "late_orders",
    "average_lateness",
    "max_lateness",
    "total_energy_consumption",
    "total_flight_distance",
    "hard_constraint_violations",
    "reached_80_acceptance",
    "reached_50_on_time",
    "reached_both_targets",
    "recommendation",
]

PAPER_FIELDS = [
    "Configuration",
    "Response Window",
    "Delivery Window Extension",
    "Resources",
    "Method",
    "Eval Instances",
    "Acceptance Rate",
    "On-time Rate",
    "Late Orders",
    "Avg. Lateness",
    "Max Lateness",
    "Energy",
    "Distance",
    "Hard Violations",
    "Reaches 80/50",
]

ORACLE_METHODS = {"oracle_best_acceptance", "oracle_best_on_time"}


def _num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        x = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_compact(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if abs(x - round(x)) <= 1e-12:
        return str(int(round(x)))
    return f"{x:g}"


def _bool_text(value: bool) -> str:
    return "True" if bool(value) else "False"


def _reaches(row: Dict[str, Any]) -> bool:
    return (
        _num(row, "acceptance_rate") >= 0.80
        and _num(row, "on_time_rate") >= 0.50
        and int(_num(row, "hard_constraint_violations")) == 0
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _infer_eval_instances(run_dir: Path, rows: Sequence[Dict[str, Any]]) -> int:
    cfg_path = run_dir / "configs" / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return int(cfg.get("eval_instances"))
        except Exception:
            pass
    for row in rows:
        value = row.get("eval_instances")
        if value not in ("", None):
            return int(float(value))
    match = re.search(r"(\d+)$", run_dir.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer eval_instances for {run_dir}")


def _normalize_row(row: Dict[str, Any], eval_instances: int) -> Dict[str, Any]:
    out = dict(row)
    acc = _num(out, "acceptance_rate")
    ot = _num(out, "on_time_rate")
    hard = int(_num(out, "hard_constraint_violations"))
    reached_acc = acc >= 0.80
    reached_ot = ot >= 0.50
    reached_both = reached_acc and reached_ot and hard == 0
    out["eval_instances"] = int(eval_instances)
    out["acceptance_rate_gap_to_0_80"] = 0.80 - acc
    out["on_time_rate_gap_to_0_50"] = 0.50 - ot
    out["hard_constraint_violations"] = hard
    out["late_orders"] = int(_num(out, "late_orders"))
    out["resource_count"] = int(_num(out, "resource_count"))
    out["order_density_ratio"] = _num(out, "order_density_ratio", 1.0)
    out["delivery_window_extension"] = _num(out, "delivery_window_extension")
    out["reached_80_acceptance"] = reached_acc
    out["reached_50_on_time"] = reached_ot
    out["reached_both_targets"] = reached_both
    if reached_both:
        out["recommendation"] = "reaches_80_50_with_zero_hard_violations"
    elif hard != 0:
        out["recommendation"] = "invalid_hard_constraint_violation"
    elif reached_acc:
        out["recommendation"] = "acceptance_target_only"
    elif reached_ot:
        out["recommendation"] = "on_time_target_only"
    else:
        out["recommendation"] = "below_target"
    return out


def load_stability_rows(output_dir: Path, eval_dirs: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel in eval_dirs:
        run_dir = output_dir / rel
        summary_path = run_dir / "reports" / "final_business_constraint_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
        raw_rows = _read_csv(summary_path)
        eval_instances = _infer_eval_instances(run_dir, raw_rows)
        rows.extend(_normalize_row(row, eval_instances) for row in raw_rows)
    return sorted(
        rows,
        key=lambda r: (
            int(r["eval_instances"]),
            CONFIG_ORDER.get(str(r["experiment_name"]), 99),
            METHOD_ORDER.get(str(r["method_name"]), 99),
        ),
    )


def _paper_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Configuration": row["experiment_name"],
        "Response Window": row["response_window"],
        "Delivery Window Extension": _fmt(row["delivery_window_extension"], 1),
        "Resources": int(row["resource_count"]),
        "Method": row["method_name"],
        "Eval Instances": int(row["eval_instances"]),
        "Acceptance Rate": _fmt(row["acceptance_rate"], 6),
        "On-time Rate": _fmt(row["on_time_rate"], 6),
        "Late Orders": int(row["late_orders"]),
        "Avg. Lateness": _fmt(row["average_lateness"], 6),
        "Max Lateness": _fmt(row["max_lateness"], 6),
        "Energy": _fmt(row["total_energy_consumption"], 6),
        "Distance": _fmt(row["total_flight_distance"], 6),
        "Hard Violations": int(row["hard_constraint_violations"]),
        "Reaches 80/50": _bool_text(_reaches(row)),
    }


def _write_markdown_table(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str], title: Optional[str] = None) -> None:
    lines: List[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def write_latex_table(path: Path, paper_rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Stability validation of business-constraint configurations for the 80/50 target.}",
        "\\label{tab:business_constraint_stability}",
        "\\begin{tabular}{llrrlrrrrrrrrrl}",
        "\\toprule",
        "Configuration & Resp. & Due Ext. & Res. & Method & Eval & Acc. & On-time & Late & Avg. Late & Max Late & Energy & Distance & Hard & 80/50 \\\\",
        "\\midrule",
    ]
    for row in paper_rows:
        fields = [
            _latex_escape(row["Configuration"]),
            _latex_escape(row["Response Window"]),
            _fmt_compact(row["Delivery Window Extension"]),
            str(row["Resources"]),
            _latex_escape(row["Method"]),
            str(row["Eval Instances"]),
            _fmt(row["Acceptance Rate"], 3),
            _fmt(row["On-time Rate"], 3),
            str(row["Late Orders"]),
            _fmt(row["Avg. Lateness"], 2),
            _fmt(row["Max Lateness"], 2),
            _fmt(row["Energy"], 2),
            _fmt(row["Distance"], 2),
            str(row["Hard Violations"]),
            _latex_escape(row["Reaches 80/50"]),
        ]
        lines.append(" & ".join(fields) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _group_by_pair(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["experiment_name"]), str(row["method_name"])), []).append(row)
    return grouped


def _stability_pairs(rows: Sequence[Dict[str, Any]], required_evals: Sequence[int]) -> List[Dict[str, Any]]:
    grouped = _group_by_pair(rows)
    stable: List[Dict[str, Any]] = []
    required = set(int(x) for x in required_evals)
    for (config, method), pair_rows in grouped.items():
        if method not in ORACLE_METHODS:
            continue
        by_eval = {int(r["eval_instances"]): r for r in pair_rows}
        if not required.issubset(by_eval):
            continue
        selected = [by_eval[e] for e in sorted(required)]
        if all(_reaches(r) for r in selected):
            stable.append(
                {
                    "config": config,
                    "method": method,
                    "rows": selected,
                    "resource_count": int(selected[-1]["resource_count"]),
                    "response_window": float(selected[-1]["response_window"]),
                    "delivery_window_extension": float(selected[-1]["delivery_window_extension"]),
                    "worst_avg_lateness": max(_num(r, "average_lateness") for r in selected),
                    "worst_max_lateness": max(_num(r, "max_lateness") for r in selected),
                    "worst_energy": max(_num(r, "total_energy_consumption") for r in selected),
                    "worst_distance": max(_num(r, "total_flight_distance") for r in selected),
                }
            )
    return stable


def _recommendation_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    method_penalty = 0 if item["method"] == "oracle_best_on_time" else 1
    return (
        int(item["resource_count"]),
        float(item["response_window"]),
        float(item["delivery_window_extension"]),
        float(item["worst_avg_lateness"]),
        float(item["worst_max_lateness"]),
        float(item["worst_energy"]),
        float(item["worst_distance"]),
        method_penalty,
    )


def select_recommendation(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    required_evals = sorted({int(r["eval_instances"]) for r in rows})
    stable = _stability_pairs(rows, required_evals)
    if stable:
        return min(stable, key=_recommendation_key), stable
    return None, stable


def _row_label(row: Dict[str, Any]) -> str:
    return (
        f"{row['experiment_name']} / {row['method_name']} eval={int(row['eval_instances'])}: "
        f"acc={_fmt(row['acceptance_rate'], 6)}, on_time={_fmt(row['on_time_rate'], 6)}, "
        f"late={int(row['late_orders'])}, avg_late={_fmt(row['average_lateness'], 6)}, "
        f"max_late={_fmt(row['max_lateness'], 6)}, energy={_fmt(row['total_energy_consumption'], 6)}, "
        f"distance={_fmt(row['total_flight_distance'], 6)}, hard={int(row['hard_constraint_violations'])}"
    )


def _load_prior_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [_normalize_row(row, int(float(row.get("eval_instances") or 30))) for row in _read_csv(path)]


def _best_prior_example(rows: Sequence[Dict[str, Any]], prefix: str, require_acc: bool = False, require_on_time: bool = False) -> Optional[Dict[str, Any]]:
    candidates = [r for r in rows if str(r.get("experiment_name", "")).startswith(prefix) and int(_num(r, "hard_constraint_violations")) == 0]
    if require_acc:
        candidates = [r for r in candidates if _num(r, "acceptance_rate") >= 0.80]
    if require_on_time:
        candidates = [r for r in candidates if _num(r, "on_time_rate") >= 0.50]
    if not candidates:
        return None
    if require_acc:
        return max(candidates, key=lambda r: _num(r, "on_time_rate"))
    if require_on_time:
        return max(candidates, key=lambda r: _num(r, "acceptance_rate"))
    return min(candidates, key=lambda r: max(0.0, 0.80 - _num(r, "acceptance_rate")) + max(0.0, 0.50 - _num(r, "on_time_rate")))


def _write_stability_validation(path: Path, rows: Sequence[Dict[str, Any]], recommendation: Optional[Dict[str, Any]], stable: Sequence[Dict[str, Any]]) -> None:
    grouped = _group_by_pair(rows)
    evals = sorted({int(r["eval_instances"]) for r in rows})
    lines = [
        "# Business Constraint Stability Validation",
        "",
        "## Scope",
        "",
        "- No model training, ServicePolicy training, imitation dataset generation, baseline weight replacement, or joint-teacher tuning was performed.",
        "- The validation is limited to combined_D, combined_E, combined_F, and combined_G.",
        "- Methods: raw_baseline, v2_repair_only, tail_risk_constrained_joint_beam, oracle_best_acceptance, oracle_best_on_time.",
        "- The tail_risk_constrained_joint_beam row is used as a safe baseline reference in this runner; it is anchor-locked to raw_baseline.",
        f"- Eval scales: {', '.join(str(x) for x in evals)}.",
        "",
        "## Stability Matrix",
        "",
        "| configuration | method | " + " | ".join(f"eval={e}" for e in evals) + " | stable |",
        "|" + "|".join("---" for _ in range(3 + len(evals))) + "|",
    ]
    for config in sorted(CONFIG_ORDER, key=lambda x: CONFIG_ORDER[x]):
        for method in sorted(METHOD_ORDER, key=lambda x: METHOD_ORDER[x]):
            pair_rows = grouped.get((config, method), [])
            by_eval = {int(r["eval_instances"]): r for r in pair_rows}
            statuses = []
            for e in evals:
                row = by_eval.get(e)
                statuses.append("pass" if row and _reaches(row) else "fail")
            stable_text = "yes" if statuses and all(s == "pass" for s in statuses) and method in ORACLE_METHODS else "no"
            lines.append(f"| {config} | {method} | " + " | ".join(statuses) + f" | {stable_text} |")
    lines.extend(["", "## Recommended Stable Pair", ""])
    if recommendation:
        lines.append(
            f"- Recommended: `{recommendation['config']} / {recommendation['method']}` "
            f"(response={recommendation['response_window']}, due+{recommendation['delivery_window_extension']}, "
            f"resources={recommendation['resource_count']})."
        )
    else:
        lines.append("- No oracle pair reached 80/50 with zero hard violations at every evaluated scale.")
    if stable:
        lines.extend(["", "## Stable Oracle Pairs", ""])
        for item in sorted(stable, key=_recommendation_key):
            lines.append(
                f"- `{item['config']} / {item['method']}`: response={item['response_window']}, "
                f"due+{item['delivery_window_extension']}, resources={item['resource_count']}, "
                f"worst_avg_late={item['worst_avg_lateness']:.6f}, worst_max_late={item['worst_max_lateness']:.6f}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommendation(path: Path, rows: Sequence[Dict[str, Any]], prior_rows: Sequence[Dict[str, Any]], recommendation: Optional[Dict[str, Any]], stable: Sequence[Dict[str, Any]]) -> None:
    response_example = _best_prior_example(prior_rows, "response_window_", require_acc=True)
    delivery_example = _best_prior_example(prior_rows, "delivery_window_", require_on_time=True)
    resource_example = _best_prior_example(prior_rows, "resource_count_", require_on_time=True)
    lower_combo_rows = [r for r in prior_rows if str(r.get("experiment_name")) in {"combined_A", "combined_B", "combined_C"} and str(r.get("method_name")) in ORACLE_METHODS]
    lower_best = min(
        lower_combo_rows,
        key=lambda r: max(0.0, 0.80 - _num(r, "acceptance_rate")) + max(0.0, 0.50 - _num(r, "on_time_rate")),
        default=None,
    )
    lines = [
        "# Recommendation For 80/50",
        "",
        "## Decision",
        "",
    ]
    if recommendation:
        lines.append(
            f"- Recommend `{recommendation['config']} / {recommendation['method']}` as the smallest stable observed feasible configuration."
        )
        lines.append(
            f"- Business settings: response_window={recommendation['response_window']}, "
            f"delivery_window_extension=+{recommendation['delivery_window_extension']}, resources={recommendation['resource_count']}."
        )
    else:
        lines.append("- No D/E/F/G oracle method was stable at every evaluated scale; do not claim a stable 80/50 configuration.")
    lines.extend(
        [
            "- Continue model training: no.",
            "- Continue teacher or ServicePolicy work: no.",
            "- Recommended lever: change business rules and resource configuration.",
            "",
            "## Boundary Explanation",
            "",
        ]
    )
    if response_example:
        lines.append(
            f"- Response window alone is not enough: the best response-only high-acceptance row was `{response_example['experiment_name']} / {response_example['method_name']}` "
            f"with acc={_fmt(response_example['acceptance_rate'], 3)} and on_time={_fmt(response_example['on_time_rate'], 3)}."
        )
    else:
        lines.append("- Response window alone is not enough: no response-only row simultaneously reached the on-time target in the prior full sensitivity run.")
    if delivery_example:
        lines.append(
            f"- Delivery window alone is not enough: the best delivery-only on-time row was `{delivery_example['experiment_name']} / {delivery_example['method_name']}` "
            f"with acc={_fmt(delivery_example['acceptance_rate'], 3)} and on_time={_fmt(delivery_example['on_time_rate'], 3)}."
        )
    else:
        lines.append("- Delivery window alone is not enough: no delivery-only row simultaneously reached the acceptance target in the prior full sensitivity run.")
    if resource_example:
        lines.append(
            f"- Resources alone are not enough: the best resource-only on-time row was `{resource_example['experiment_name']} / {resource_example['method_name']}` "
            f"with acc={_fmt(resource_example['acceptance_rate'], 3)} and on_time={_fmt(resource_example['on_time_rate'], 3)}."
        )
    else:
        lines.append("- Resources alone are not enough: no resource-only row simultaneously reached the acceptance target in the prior full sensitivity run.")
    if lower_best:
        lines.append(
            f"- Lower combined settings A/B/C did not clear both targets; the closest lower setting was `{lower_best['experiment_name']} / {lower_best['method_name']}` "
            f"with acc={_fmt(lower_best['acceptance_rate'], 3)} and on_time={_fmt(lower_best['on_time_rate'], 3)}."
        )
    lines.extend(
        [
            "- The target is primarily a business-configuration target, not a model-training target under the original constraints.",
            "- If resources must be reduced below the recommended pair, current evidence only shows that response=5.0 and due+3.0 with one resource is insufficient; a separate focused sweep is needed to quantify the extra window.",
            "- If time windows cannot be relaxed, current evidence only shows that up to five resources in the single-axis run does not reach 80/50; additional resources or acceptance-rule changes would need a separate feasibility run.",
            "",
            "## Key Oracle Rows",
            "",
        ]
    )
    for row in rows:
        if str(row["method_name"]) in ORACLE_METHODS:
            lines.append(f"- {_row_label(row)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_summary_md(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    paper_rows = [_paper_row(r) for r in rows]
    _write_markdown_table(path, paper_rows, PAPER_FIELDS, title="Final Business Constraint Summary")


def _write_thesis_paragraph(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    recommendation: Optional[Dict[str, Any]],
    prior_rows: Sequence[Dict[str, Any]],
) -> None:
    rec_text = "未观察到稳定可行组合"
    extra = ""
    if recommendation:
        rec_rows = sorted(recommendation["rows"], key=lambda r: int(r["eval_instances"]))
        prior_match = next(
            (
                r
                for r in prior_rows
                if str(r.get("experiment_name")) == recommendation["config"]
                and str(r.get("method_name")) == recommendation["method"]
            ),
            None,
        )
        prior_text = ""
        if prior_match:
            prior_text = (
                f"在早期 eval={int(prior_match['eval_instances'])} 验证中，该组合已达到 "
                f"acc={_fmt(prior_match['acceptance_rate'], 3)}、"
                f"on_time={_fmt(prior_match['on_time_rate'], 3)}、"
                f"hard={int(prior_match['hard_constraint_violations'])}。"
            )
        metric_text = "；".join(
            f"eval={int(r['eval_instances'])} 时 acc={_fmt(r['acceptance_rate'], 3)}、on_time={_fmt(r['on_time_rate'], 3)}、hard={int(r['hard_constraint_violations'])}"
            for r in rec_rows
        )
        rec_text = (
            f"最小稳定观察可行组合为 response_window={recommendation['response_window']}、"
            f"delivery_window_extension=+{recommendation['delivery_window_extension']}、"
            f"resources={recommendation['resource_count']}，对应方法为 {recommendation['method']}"
        )
        extra = f"{prior_text}本轮稳定性验证结果为：{metric_text}。"
    paragraph = (
        "在原始动态响应窗口和单资源串行调度条件下，80% 接单率与 50% 准时率的联合目标不可达。"
        "多轮模型侧优化，包括 reward 调整、V2 repair、joint teacher、safe-deviation teacher 以及 ServicePolicy 准备流程，"
        "均未能在原始业务约束下稳定突破 baseline 并同时满足硬约束。业务约束敏感性实验表明，"
        "目标达成依赖响应窗口、配送时窗和并行资源的组合放宽，而不是单一模型训练带来的改进。"
        f"{rec_text}。{extra}"
        "因此，本研究建议优先调整业务约束和资源配置，而不是继续投入旧模型的长时间训练或继续扩展 teacher/ServicePolicy 流程。"
    )
    path.write_text("# Thesis Conclusion Paragraph\n\n" + paragraph + "\n", encoding="utf-8")


def write_reports(output_dir: Path, rows: Sequence[Dict[str, Any]], prior_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    recommendation, stable = select_recommendation(rows)
    paper_rows = [_paper_row(r) for r in rows]

    _write_csv(reports / "final_business_constraint_summary.csv", rows, SUMMARY_FIELDS)
    _write_json(
        reports / "final_business_constraint_summary.json",
        {
            "rows": list(rows),
            "recommended_pair": None
            if recommendation is None
            else {
                "config": recommendation["config"],
                "method": recommendation["method"],
                "response_window": recommendation["response_window"],
                "delivery_window_extension": recommendation["delivery_window_extension"],
                "resources": recommendation["resource_count"],
            },
            "stable_oracle_pairs": [
                {
                    "config": item["config"],
                    "method": item["method"],
                    "response_window": item["response_window"],
                    "delivery_window_extension": item["delivery_window_extension"],
                    "resources": item["resource_count"],
                }
                for item in stable
            ],
        },
    )
    _write_final_summary_md(reports / "final_business_constraint_summary.md", rows)
    _write_csv(reports / "paper_business_constraint_table.csv", paper_rows, PAPER_FIELDS)
    _write_markdown_table(reports / "paper_business_constraint_table.md", paper_rows, PAPER_FIELDS, title="Paper Business Constraint Table")
    write_latex_table(reports / "paper_business_constraint_table.tex", paper_rows)
    _write_stability_validation(reports / "business_constraint_stability_validation.md", rows, recommendation, stable)
    _write_recommendation(reports / "recommendation_for_80_50.md", rows, prior_rows, recommendation, stable)
    _write_thesis_paragraph(reports / "thesis_conclusion_paragraph.md", rows, recommendation, prior_rows)
    return {
        "reports_dir": str(reports.resolve()),
        "rows": len(rows),
        "recommended_pair": None if recommendation is None else f"{recommendation['config']} / {recommendation['method']}",
        "stable_oracle_pairs": [f"{item['config']} / {item['method']}" for item in stable],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize D/E/F/G business-constraint stability validation.")
    parser.add_argument("--output-dir", type=str, default="experiments/business_constraint_sensitivity_80_50_stability")
    parser.add_argument("--eval-dirs", type=str, default="eval_50,eval_100")
    parser.add_argument(
        "--prior-summary",
        type=str,
        default="experiments/business_constraint_sensitivity_80_50/reports/final_business_constraint_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    eval_dirs = [x.strip() for x in str(args.eval_dirs).split(",") if x.strip()]
    rows = load_stability_rows(output_dir, eval_dirs)
    prior_summary = Path(args.prior_summary) if args.prior_summary else None
    prior_rows = _load_prior_rows(prior_summary)
    result = write_reports(output_dir, rows, prior_rows)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
