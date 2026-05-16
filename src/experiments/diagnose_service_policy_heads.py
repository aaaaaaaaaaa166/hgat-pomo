from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.models.service_policy import ServicePolicy


FIELDS = [
    "model_name",
    "samples",
    "accept_samples",
    "accept_positive_rate",
    "model_accept_rate",
    "accept_accuracy",
    "accept_precision",
    "accept_recall",
    "route_samples",
    "route_top1_accuracy",
    "route_top3_accuracy",
    "assignment_samples",
    "assignment_accuracy",
    "assignment_drone_positive_rate",
    "model_drone_assignment_rate",
    "assignment_drone_recall",
    "lateness_samples",
    "lateness_mae",
    "lateness_rmse",
    "lateness_risk_auc",
    "on_time_class_accuracy",
    "risky_order_false_accept_rate",
    "predicted_late_but_accepted_count",
    "accepted_late_count",
    "teacher_policy_match_rate",
]


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "samples" in payload:
        return list(payload["samples"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format: {path}")


def _load_model(path: str, device: torch.device) -> ServicePolicy:
    state = torch.load(path, map_location=device, weights_only=False)
    dims = state.get("dims", {})
    cfg = state.get("config", {})
    model = ServicePolicy(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        heads=int(cfg.get("heads", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
        k_nn_orders=int(cfg.get("k_nn_orders", 8)),
        num_encoder_layers=int(cfg.get("encoder_layers", 2)),
        order_feature_dim=int(dims.get("order_feature_dim", 22)),
        truck_feature_dim=int(dims.get("truck_feature_dim", 6)),
        drone_feature_dim=int(dims.get("drone_feature_dim", 6)),
    ).to(device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _target_lateness(sample: Dict[str, Any]) -> float:
    for key in ("actual_lateness_label", "predicted_lateness_label", "lateness_risk_label"):
        value = sample.get(key, None)
        if value not in ("", None):
            return max(0.0, _num(value))
    return 0.0


def _safe_div(num: float, den: float) -> float:
    return 0.0 if abs(float(den)) <= 1e-12 else float(num) / float(den)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0.0
    for ps in pos:
        for ns in neg:
            total += 1.0
            if ps > ns:
                wins += 1.0
            elif abs(ps - ns) <= 1e-12:
                wins += 0.5
    return _safe_div(wins, total)


def evaluate_heads(
    model: ServicePolicy,
    samples: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    lateness_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    accept_total = 0
    accept_correct = 0
    accept_tp = 0
    accept_fp = 0
    accept_fn = 0
    accept_positive = 0
    model_accept = 0
    risky_false_accept = 0
    risky_teacher_reject = 0
    predicted_late_but_accepted = 0
    accepted_late = 0

    route_total = 0
    route_top1 = 0
    route_top3 = 0
    assignment_total = 0
    assignment_correct = 0
    assignment_drone_positive = 0
    model_drone_assignment = 0
    assignment_drone_tp = 0
    lateness_total = 0
    abs_err = 0.0
    sq_err = 0.0
    on_time_correct = 0
    risk_scores: List[float] = []
    late_labels: List[int] = []
    teacher_match = 0
    teacher_match_total = 0

    with torch.no_grad():
        for sample in samples:
            out = model.forward_data(sample["data"].to(device))
            current = int(sample.get("current_order_id", -1))
            accept_label = int(sample.get("accept_label", -100))
            best_next = int(sample.get("best_next_order", -100))
            target_node = current if current > 0 else best_next
            target_late = _target_lateness(sample)

            if accept_label != -100 and current > 0 and current < int(out["accept_logits"].size(0)):
                pred = int(torch.argmax(out["accept_logits"][current]).item())
                accept_total += 1
                accept_positive += int(accept_label == 1)
                model_accept += int(pred == 1)
                accept_correct += int(pred == accept_label)
                accept_tp += int(pred == 1 and accept_label == 1)
                accept_fp += int(pred == 1 and accept_label == 0)
                accept_fn += int(pred == 0 and accept_label == 1)
                if target_late > float(lateness_threshold) and accept_label == 0:
                    risky_teacher_reject += 1
                    risky_false_accept += int(pred == 1)
                if pred == 1 and target_late > float(lateness_threshold):
                    accepted_late += 1
                teacher_match += int(pred == accept_label)
                teacher_match_total += 1

            if best_next != -100 and 0 <= best_next < int(out["route_priority_logits"].numel()):
                scores = out["route_priority_logits"].detach().clone()
                extra = sample.get("extra", {}) or {}
                mask = extra.get("truck_mask")
                if mask is not None:
                    mask = mask.to(scores.device).bool() if torch.is_tensor(mask) else torch.as_tensor(mask, device=scores.device).bool()
                    scores = scores.masked_fill(~mask, -1e9)
                k = min(3, int(scores.numel()))
                topk = torch.topk(scores, k=k).indices.detach().cpu().tolist()
                route_total += 1
                route_top1 += int(topk[0] == best_next)
                route_top3 += int(best_next in topk)
                teacher_match += int(topk[0] == best_next)
                teacher_match_total += 1
                assignment_label = int(sample.get("assignment_label", -100))
                if assignment_label != -100:
                    assignment_cls = 0 if assignment_label < 0 else assignment_label + 1
                    assignment_logits = torch.cat([out["no_drone_logit"], out["drone_assignment_logits"]], dim=0)
                    if 0 <= assignment_cls < int(assignment_logits.numel()):
                        pred_assignment = int(torch.argmax(assignment_logits).item())
                        assignment_total += 1
                        assignment_correct += int(pred_assignment == assignment_cls)
                        assignment_drone_positive += int(assignment_cls > 0)
                        model_drone_assignment += int(pred_assignment > 0)
                        assignment_drone_tp += int(pred_assignment > 0 and assignment_cls > 0)

            if target_node > 0 and target_node < int(out["lateness_risk"].numel()):
                pred_late = float(torch.clamp(out["lateness_risk"][target_node], min=0.0).detach().cpu())
                err = pred_late - target_late
                lateness_total += 1
                abs_err += abs(err)
                sq_err += err * err
                late_label = int(target_late > float(lateness_threshold))
                pred_late_label = int(pred_late > float(lateness_threshold))
                on_time_correct += int(pred_late_label == late_label)
                risk_scores.append(pred_late)
                late_labels.append(late_label)
                if current > 0 and accept_label != -100:
                    pred_accept = int(torch.argmax(out["accept_logits"][current]).item())
                    if pred_accept == 1 and pred_late > float(lateness_threshold):
                        predicted_late_but_accepted += 1

    auc = _auc(risk_scores, late_labels)
    return {
        "model_name": model_name,
        "samples": int(len(samples)),
        "accept_samples": int(accept_total),
        "accept_positive_rate": _safe_div(accept_positive, accept_total),
        "model_accept_rate": _safe_div(model_accept, accept_total),
        "accept_accuracy": _safe_div(accept_correct, accept_total),
        "accept_precision": _safe_div(accept_tp, accept_tp + accept_fp),
        "accept_recall": _safe_div(accept_tp, accept_tp + accept_fn),
        "route_samples": int(route_total),
        "route_top1_accuracy": _safe_div(route_top1, route_total),
        "route_top3_accuracy": _safe_div(route_top3, route_total),
        "assignment_samples": int(assignment_total),
        "assignment_accuracy": _safe_div(assignment_correct, assignment_total),
        "assignment_drone_positive_rate": _safe_div(assignment_drone_positive, assignment_total),
        "model_drone_assignment_rate": _safe_div(model_drone_assignment, assignment_total),
        "assignment_drone_recall": _safe_div(assignment_drone_tp, assignment_drone_positive),
        "lateness_samples": int(lateness_total),
        "lateness_mae": _safe_div(abs_err, lateness_total),
        "lateness_rmse": math.sqrt(_safe_div(sq_err, lateness_total)),
        "lateness_risk_auc": "" if auc is None else float(auc),
        "on_time_class_accuracy": _safe_div(on_time_correct, lateness_total),
        "risky_order_false_accept_rate": _safe_div(risky_false_accept, risky_teacher_reject),
        "predicted_late_but_accepted_count": int(predicted_late_but_accepted),
        "accepted_late_count": int(accepted_late),
        "teacher_policy_match_rate": _safe_div(teacher_match, teacher_match_total),
    }


def dataset_distribution(samples: Sequence[Dict[str, Any]], lateness_threshold: float) -> Dict[str, Any]:
    accept_labels = [int(s.get("accept_label", -100)) for s in samples if int(s.get("accept_label", -100)) != -100]
    route_labels = [int(s.get("best_next_order", -100)) for s in samples if int(s.get("best_next_order", -100)) != -100]
    lateness = [_target_lateness(s) for s in samples if (int(s.get("current_order_id", -1)) > 0 or int(s.get("best_next_order", -100)) > 0)]
    late_count = sum(1 for x in lateness if x > float(lateness_threshold))
    return {
        "samples": int(len(samples)),
        "accept_samples": int(len(accept_labels)),
        "accept_positive": int(sum(1 for x in accept_labels if x == 1)),
        "accept_negative": int(sum(1 for x in accept_labels if x == 0)),
        "accept_positive_rate": _safe_div(sum(1 for x in accept_labels if x == 1), len(accept_labels)),
        "route_samples": int(len(route_labels)),
        "lateness_samples": int(len(lateness)),
        "on_time_samples": int(len(lateness) - late_count),
        "late_samples": int(late_count),
        "late_sample_rate": _safe_div(late_count, len(lateness)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report_path: Path, head_report_path: Path, rows: Sequence[Dict[str, Any]], dist: Dict[str, Any], args: argparse.Namespace) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_lines = [
        "# Service Policy Head Metrics",
        "",
        f"- dataset: `{args.dataset_path}`",
        f"- lateness_threshold: `{args.lateness_threshold}`",
        "",
        "| model | accept_acc | accept_prec | accept_rec | model_accept_rate | route_top1 | route_top3 | assign_acc | model_drone_rate | late_mae | late_rmse | late_auc | on_time_cls_acc | risky_false_accept | teacher_match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        auc = row["lateness_risk_auc"] if row["lateness_risk_auc"] != "" else "n/a"
        metric_lines.append(
            f"| {row['model_name']} | {float(row['accept_accuracy']):.6f} | {float(row['accept_precision']):.6f} | "
            f"{float(row['accept_recall']):.6f} | {float(row['model_accept_rate']):.6f} | "
            f"{float(row['route_top1_accuracy']):.6f} | {float(row['route_top3_accuracy']):.6f} | "
            f"{float(row['assignment_accuracy']):.6f} | {float(row['model_drone_assignment_rate']):.6f} | "
            f"{float(row['lateness_mae']):.6f} | {float(row['lateness_rmse']):.6f} | {auc} | "
            f"{float(row['on_time_class_accuracy']):.6f} | {float(row['risky_order_false_accept_rate']):.6f} | "
            f"{float(row['teacher_policy_match_rate']):.6f} |"
        )
    head_report_path.write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    best = rows[0] if rows else {}
    diagnosis = [
        "# ServicePolicy On-Time Failure Diagnosis",
        "",
        "## Dataset balance",
        "",
        f"- samples: `{dist['samples']}`",
        f"- accept positives / negatives: `{dist['accept_positive']}` / `{dist['accept_negative']}`",
        f"- accept positive rate: `{float(dist['accept_positive_rate']):.6f}`",
        f"- route samples: `{dist['route_samples']}`",
        f"- on-time / late samples: `{dist['on_time_samples']}` / `{dist['late_samples']}`",
        f"- late sample rate: `{float(dist['late_sample_rate']):.6f}`",
        "",
        "## Findings",
        "",
    ]
    if best:
        accept_has_no_negatives = int(dist["accept_negative"]) == 0
        lateness_has_no_late_samples = int(dist["late_samples"]) == 0
        best_auc = best["lateness_risk_auc"] if best["lateness_risk_auc"] != "" else "n/a"
        diagnosis.extend(
            [
                f"- Accept head is strong/overactive when model_accept_rate is high. Current first model rate: `{float(best['model_accept_rate']):.6f}` against teacher positive rate `{float(best['accept_positive_rate']):.6f}`.",
                f"- Route top-1 accuracy is `{float(best['route_top1_accuracy']):.6f}` and top-3 accuracy is `{float(best['route_top3_accuracy']):.6f}`; poor top-1 indicates the model is not reproducing teacher service order.",
                f"- Assignment accuracy is `{float(best['assignment_accuracy']):.6f}` with model drone assignment rate `{float(best['model_drone_assignment_rate']):.6f}` against teacher drone-positive rate `{float(best['assignment_drone_positive_rate']):.6f}`.",
                f"- Lateness risk MAE/RMSE are `{float(best['lateness_mae']):.6f}` / `{float(best['lateness_rmse']):.6f}`; high error means the risk head is not calibrated enough for decoding.",
                f"- On-time class accuracy is `{float(best['on_time_class_accuracy']):.6f}` and risk AUC is `{best_auc}`.",
                f"- Predicted-late-but-accepted count is `{int(best['predicted_late_but_accepted_count'])}` and accepted-late count is `{int(best['accepted_late_count'])}`.",
            ]
        )
        if accept_has_no_negatives:
            diagnosis.append(
                "- Accept labels contain no reject examples, so accept accuracy/precision/recall are inflated and the accept head has no supervised signal for refusing risky orders."
            )
        if lateness_has_no_late_samples:
            diagnosis.append(
                "- Lateness labels contain no late examples at the chosen threshold, so zero lateness error or perfect on-time classification is not evidence of calibrated real-world lateness risk."
            )
        diagnosis.extend(
            [
                "",
                "## Diagnosis",
                "",
                "- The main failure is the combination of incomplete negative/risk supervision and imperfect route sequencing, not total inability to imitate.",
                "- The current dataset teaches the model to accept nearly everything, while giving little direct signal about which accepted orders become late under model-driven routing.",
                "- The model tends to preserve high acceptance while missing the oracle's on-time ordering logic.",
                "- Decoding can use lateness risk as a guard, but the guard is weak when the risk head is trained only on non-late labels.",
                "- RL fine-tune remains blocked until imitation-only reaches the small gate.",
            ]
        )
    report_path.write_text("\n".join(diagnosis) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute ServicePolicy head diagnostics on an imitation dataset.")
    p.add_argument("--dataset-path", type=str, default="experiments/default_business_env_training/imitation/oracle_best_on_time_dataset.pt")
    p.add_argument("--model-paths", type=str, required=True)
    p.add_argument("--model-names", type=str, default="")
    p.add_argument("--metrics-csv", type=str, default="experiments/default_business_env_training/metrics/service_policy_head_metrics.csv")
    p.add_argument("--head-report-path", type=str, default="experiments/default_business_env_training/reports/service_policy_head_metrics.md")
    p.add_argument("--diagnosis-report-path", type=str, default="experiments/default_business_env_training/reports/service_policy_on_time_failure_diagnosis.md")
    p.add_argument("--lateness-threshold", type=float, default=1.0)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = _load_dataset(args.dataset_path)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = [x.strip() for x in str(args.model_paths).split(",") if x.strip()]
    names = [x.strip() for x in str(args.model_names).split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        name = names[idx] if idx < len(names) else Path(path).parent.name
        model = _load_model(path, device)
        rows.append(evaluate_heads(model, samples, model_name=name, lateness_threshold=float(args.lateness_threshold), device=device))
    dist = dataset_distribution(samples, float(args.lateness_threshold))
    write_csv(Path(args.metrics_csv), rows)
    write_reports(Path(args.diagnosis_report_path), Path(args.head_report_path), rows, dist, args)
    print(json.dumps({"rows": rows, "dataset_distribution": dist}, indent=2))


if __name__ == "__main__":
    main()
