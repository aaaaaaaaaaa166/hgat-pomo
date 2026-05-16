from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.models.service_policy import ServicePolicy


def _load_dataset(path: str) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "samples" in payload:
        return list(payload["samples"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported imitation dataset format: {path}")


def _infer_dims(sample: Dict[str, Any]) -> Dict[str, int]:
    data = sample["data"]
    return {
        "order_feature_dim": int(data["order"].x.size(-1)),
        "truck_feature_dim": int(data["truck"].x.size(-1)),
        "drone_feature_dim": int(data["drone"].x.size(-1)),
    }


def _target_lateness(sample: Dict[str, Any]) -> float:
    for key in ("actual_lateness_label", "predicted_lateness_label", "lateness_risk_label"):
        value = sample.get(key, None)
        if value not in ("", None):
            try:
                return max(0.0, float(value))
            except Exception:
                pass
    return 0.0


def _route_pairwise_loss(
    logits: torch.Tensor,
    sample: Dict[str, Any],
    best_next: int,
    *,
    margin: float,
    max_negatives: int,
) -> Optional[torch.Tensor]:
    extra = sample.get("extra", {}) or {}
    truck_mask = extra.get("truck_mask")
    if truck_mask is None:
        return None
    mask = truck_mask.to(logits.device).bool() if torch.is_tensor(truck_mask) else torch.as_tensor(truck_mask, device=logits.device).bool()
    if best_next < 0 or best_next >= int(logits.numel()) or best_next >= int(mask.numel()) or not bool(mask[best_next]):
        return None
    neg = torch.where(mask)[0]
    neg = neg[(neg != int(best_next)) & (neg > 0)]
    if neg.numel() == 0:
        return None
    with torch.no_grad():
        neg_scores = logits[neg].detach()
        keep = torch.topk(neg_scores, k=min(int(max_negatives), int(neg.numel()))).indices
        neg = neg[keep]
    best_score = logits[int(best_next)]
    losses = F.relu(float(margin) - best_score + logits[neg])
    return losses.mean()


def _assignment_class(sample: Dict[str, Any], out: Dict[str, torch.Tensor]) -> Optional[int]:
    label = int(sample.get("assignment_label", -100))
    if label == -100:
        return None
    cls = 0 if label < 0 else label + 1
    num_classes = int(out["no_drone_logit"].numel() + out["drone_assignment_logits"].numel())
    if cls < 0 or cls >= num_classes:
        return None
    return int(cls)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    samples = _load_dataset(args.dataset_path)
    if not samples:
        raise ValueError("Empty imitation dataset.")
    dims = _infer_dims(samples[0])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ServicePolicy(
        hidden_dim=int(args.hidden_dim),
        heads=int(args.heads),
        dropout=float(args.dropout),
        k_nn_orders=int(args.k_nn_orders),
        num_encoder_layers=int(args.encoder_layers),
        **dims,
    ).to(device)
    if args.resume_path:
        state = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    rows: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_path = output_dir / "service_policy_imitation_best.pt"
    last_path = output_dir / "service_policy_imitation_last.pt"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(len(samples))
        total_loss = 0.0
        total_accept = 0.0
        total_route = 0.0
        total_late = 0.0
        total_score = 0.0
        total_assignment = 0.0
        total_on_time = 0.0
        total_risky_accept = 0.0
        total_pairwise = 0.0
        used = 0
        for idx in perm.tolist():
            sample = samples[idx]
            out = model.forward_data(sample["data"])
            losses: List[torch.Tensor] = []
            accept_label = int(sample.get("accept_label", -100))
            current = int(sample.get("current_order_id", -1))
            if accept_label != -100 and current > 0:
                accept_loss = F.cross_entropy(out["accept_logits"][current].view(1, -1), torch.tensor([accept_label], device=device))
                losses.append(float(args.accept_loss_weight) * accept_loss)
                total_accept += float(accept_loss.detach().cpu())
            best_next = int(sample.get("best_next_order", -100))
            if best_next != -100 and 0 <= best_next < int(out["route_priority_logits"].numel()):
                route_loss = F.cross_entropy(out["route_priority_logits"].view(1, -1), torch.tensor([best_next], device=device))
                losses.append(float(args.route_loss_weight) * route_loss)
                total_route += float(route_loss.detach().cpu())
                assignment_cls = _assignment_class(sample, out)
                if assignment_cls is not None and float(args.assignment_loss_weight) > 0.0:
                    assignment_logits = torch.cat([out["no_drone_logit"], out["drone_assignment_logits"]], dim=0).view(1, -1)
                    assignment_loss = F.cross_entropy(assignment_logits, torch.tensor([assignment_cls], device=device))
                    losses.append(float(args.assignment_loss_weight) * assignment_loss)
                    total_assignment += float(assignment_loss.detach().cpu())
                if float(args.pairwise_route_loss_weight) > 0.0:
                    pair_loss = _route_pairwise_loss(
                        out["route_priority_logits"],
                        sample,
                        best_next,
                        margin=float(args.pairwise_margin),
                        max_negatives=int(args.pairwise_max_negatives),
                    )
                    if pair_loss is not None:
                        losses.append(float(args.pairwise_route_loss_weight) * pair_loss)
                        total_pairwise += float(pair_loss.detach().cpu())
            target_node = current if current > 0 else best_next
            if target_node > 0 and target_node < int(out["lateness_risk"].numel()):
                late_target_f = _target_lateness(sample)
                late_target = torch.tensor(late_target_f, device=device)
                late_loss = F.smooth_l1_loss(out["lateness_risk"][target_node], late_target)
                score_target = torch.tensor(float(sample.get("insertion_score_label", 0.0)), device=device)
                score_loss = F.smooth_l1_loss(out["insertion_score"][target_node], score_target)
                losses.append(float(args.lateness_loss_weight) * late_loss)
                losses.append(float(args.score_loss_weight) * score_loss)
                total_late += float(late_loss.detach().cpu())
                total_score += float(score_loss.detach().cpu())
                if float(args.on_time_loss_weight) > 0.0:
                    on_time_target = torch.tensor(1.0 if late_target_f <= float(args.on_time_lateness_threshold) else 0.0, device=device)
                    on_time_loss = F.binary_cross_entropy_with_logits(-out["lateness_risk"][target_node], on_time_target)
                    losses.append(float(args.on_time_loss_weight) * on_time_loss)
                    total_on_time += float(on_time_loss.detach().cpu())
                if (
                    float(args.risky_accept_penalty) > 0.0
                    and accept_label != -100
                    and current > 0
                    and late_target_f > float(args.risky_lateness_threshold)
                ):
                    accept_prob = F.softmax(out["accept_logits"][current], dim=-1)[1]
                    severity = min(5.0, late_target_f / max(1e-6, float(args.risky_lateness_threshold)))
                    risky_loss = accept_prob * float(severity)
                    losses.append(float(args.risky_accept_penalty) * risky_loss)
                    total_risky_accept += float(risky_loss.detach().cpu())
            if not losses:
                continue
            loss = sum(losses)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} sample={idx}: {float(loss.detach().cpu())}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            used += 1
        denom = max(1, used)
        row = {
            "epoch": epoch,
            "samples_used": used,
            "loss": total_loss / denom,
            "accept_loss": total_accept / denom,
            "route_loss": total_route / denom,
            "lateness_loss": total_late / denom,
            "score_loss": total_score / denom,
            "assignment_loss": total_assignment / denom,
            "on_time_loss": total_on_time / denom,
            "risky_accept_loss": total_risky_accept / denom,
            "pairwise_route_loss": total_pairwise / denom,
        }
        rows.append(row)
        if row["loss"] < best_loss:
            best_loss = float(row["loss"])
            torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "dims": dims, "epoch": epoch, "loss": best_loss}, best_path)
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "dims": dims, "epoch": epoch, "loss": row["loss"]}, last_path)
        print(f"[service_policy] epoch={epoch} loss={row['loss']:.6f}")

    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"best_path": str(best_path), "last_path": str(last_path), "history_path": str(history_path), "best_loss": best_loss}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised warm start for ServicePolicy. Does not touch baseline weights.")
    p.add_argument("--dataset-path", type=str, default="experiments/service_v2/imitation/imitation_dataset.pt")
    p.add_argument("--output-dir", type=str, default="experiments/service_v2/models")
    p.add_argument("--resume-path", type=str, default="")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--k-nn-orders", type=int, default=8)
    p.add_argument("--accept-loss-weight", type=float, default=1.0)
    p.add_argument("--route-loss-weight", type=float, default=1.0)
    p.add_argument("--lateness-loss-weight", type=float, default=0.2)
    p.add_argument("--score-loss-weight", type=float, default=0.05)
    p.add_argument("--assignment-loss-weight", type=float, default=0.0)
    p.add_argument("--on-time-loss-weight", type=float, default=0.0)
    p.add_argument("--on-time-lateness-threshold", type=float, default=1e-6)
    p.add_argument("--risky-accept-penalty", type=float, default=0.0)
    p.add_argument("--risky-lateness-threshold", type=float, default=1.0)
    p.add_argument("--pairwise-route-loss-weight", type=float, default=0.0)
    p.add_argument("--pairwise-margin", type=float, default=1.0)
    p.add_argument("--pairwise-max-negatives", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
