from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class TimeWindowInferenceConfig:
    enable_time_window_bias: bool = False
    lateness_bias_weight: float = 0.0
    severe_lateness_threshold: float = 10.0
    severe_lateness_bias_weight: float = 0.0
    allow_late_if_no_feasible_action: bool = True
    enable_local_repair: bool = False
    repair_max_iterations: int = 50
    repair_window_size: int = 4
    repair_objective: str = "lateness"


def _safe_due(env: Any, node: int) -> float:
    if node <= 0:
        return float("inf")
    try:
        return float(env.due[int(node)])
    except Exception:
        return float("inf")


def _lateness_risk(predicted_lateness: float, cfg: TimeWindowInferenceConfig) -> float:
    late = max(0.0, float(predicted_lateness))
    severe = max(0.0, late - float(cfg.severe_lateness_threshold))
    return float(float(cfg.lateness_bias_weight) * late + float(cfg.severe_lateness_bias_weight) * severe)


def predict_action_lateness(
    env: Any,
    obs: Dict[str, Any],
    j: int,
    k: Optional[int] = None,
    cfg: Optional[TimeWindowInferenceConfig] = None,
) -> Dict[str, Any]:
    """Estimate whether a candidate inference action is likely to create lateness.

    The estimate mirrors the environment's timing equations but avoids stochastic
    traffic so it can be used before executing the action.
    """
    cfg = cfg or TimeWindowInferenceConfig()
    i = int(obs.get("i", 0))
    t = float(obs.get("t", 0.0))
    j = int(j)
    k = env.K_NONE if k is None else int(k)
    current_request = int(obs.get("current_decision_request", -1))

    if current_request > 0:
        if j == 0:
            return {
                "estimated_arrival_time": t,
                "planned_delivery_time": float(env.decision_deadline[current_request]),
                "predicted_lateness": 0.0,
                "will_be_late": False,
                "lateness_risk_score": 0.0,
                "feasibility_reason": "reject_decision",
                "node_lateness": {},
            }
        due_t = _safe_due(env, current_request)
        try:
            travel = float(env._tau_truck(i, current_request, apply_traffic=False, bucket=str(obs.get("time_bucket", env.get_time_bucket()))))
        except Exception:
            travel = float(env.dist_mat[i, current_request]) / max(1e-9, float(env.cfg.vT))
        finish = t + travel + float(env.cfg.sT)
        late = 0.0 if not math.isfinite(due_t) else max(0.0, finish - due_t)
        return {
            "estimated_arrival_time": float(finish),
            "planned_delivery_time": due_t,
            "predicted_lateness": float(late),
            "will_be_late": bool(late > 1e-9),
            "lateness_risk_score": _lateness_risk(late, cfg),
            "feasibility_reason": "accept_decision_estimate",
            "node_lateness": {int(current_request): float(late)},
        }

    node_lateness: Dict[int, float] = {}
    finish_times: Dict[int, float] = {}
    bucket = str(obs.get("time_bucket", env.get_time_bucket()))

    try:
        truck_mask = env.get_masks()["truck_mask"]
    except Exception:
        truck_mask = np.zeros((env.N + 1,), dtype=np.int8)
    if not (0 <= j <= env.N) or int(truck_mask[j]) == 0:
        return {
            "estimated_arrival_time": t,
            "planned_delivery_time": _safe_due(env, j),
            "predicted_lateness": float("inf"),
            "will_be_late": True,
            "lateness_risk_score": float("inf"),
            "feasibility_reason": "infeasible_truck_action",
            "node_lateness": {},
        }

    try:
        truck_travel = float(env._tau_truck(i, j, apply_traffic=False, bucket=bucket))
    except Exception:
        truck_travel = float(env.dist_mat[i, j]) / max(1e-9, float(env.cfg.vT))

    truck_service = 0.0
    if j == 0:
        truck_service = float(env.cfg.depot_service_time) if i != 0 else 0.0
    elif env._request_feasible_for_truck(
        node=j,
        accepted=env.state["accepted"],
        served=env.state["served"],
        loaded=env.state["loaded"],
        truck_pickup_load=float(env.state["truck_pickup_load"]),
        t=t,
    ):
        truck_service = float(env.cfg.sT)
    finish_j = t + truck_travel + truck_service
    if j > 0 and truck_service > 0.0:
        due_j = _safe_due(env, j)
        late_j = 0.0 if not math.isfinite(due_j) else max(0.0, finish_j - due_j)
        node_lateness[j] = float(late_j)
        finish_times[j] = float(finish_j)

    if k != env.K_NONE:
        try:
            drone_mask = env.get_masks(j=j)["drone_mask"]
        except Exception:
            drone_mask = np.zeros((env.N + 1,), dtype=np.int8)
        if not (1 <= k <= env.N) or int(drone_mask[k]) == 0:
            return {
                "estimated_arrival_time": t,
                "planned_delivery_time": _safe_due(env, k),
                "predicted_lateness": float("inf"),
                "will_be_late": True,
                "lateness_risk_score": float("inf"),
                "feasibility_reason": "infeasible_drone_action",
                "node_lateness": dict(node_lateness),
            }
        finish_k = t + float(env._tau_drone(i, k, j)) + float(env.cfg.sD)
        due_k = _safe_due(env, k)
        late_k = 0.0 if not math.isfinite(due_k) else max(0.0, finish_k - due_k)
        node_lateness[k] = float(late_k)
        finish_times[k] = float(finish_k)

    max_late = max([0.0] + list(node_lateness.values()))
    planned = _safe_due(env, j if j > 0 else (k if k != env.K_NONE else 0))
    estimated = max([t] + list(finish_times.values()))
    if not node_lateness:
        reason = "no_service_or_wait"
    elif max_late > 1e-9:
        reason = "predicted_late"
    else:
        reason = "predicted_on_time"
    return {
        "estimated_arrival_time": float(estimated),
        "planned_delivery_time": planned,
        "predicted_lateness": float(max_late),
        "will_be_late": bool(max_late > 1e-9),
        "lateness_risk_score": _lateness_risk(max_late, cfg),
        "feasibility_reason": reason,
        "node_lateness": node_lateness,
    }


def _shape_logits(decoder: Any, logits: torch.Tensor, greedy: bool) -> torch.Tensor:
    if float(decoder.tanh_clipping) > 0:
        logits = float(decoder.tanh_clipping) * torch.tanh(logits)
    if not greedy and float(decoder.temperature) > 0 and abs(float(decoder.temperature) - 1.0) > 1e-8:
        logits = logits / float(decoder.temperature)
    return logits


def _sample_from_logits(logits: torch.Tensor, greedy: bool) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    p = probs.clamp_min(1e-12)
    entropy = -(p * torch.log(p)).sum()
    if greedy:
        idx = int(torch.argmax(probs, dim=-1).item())
        logp = torch.log(probs[idx].clamp_min(1e-12))
    else:
        dist = torch.distributions.Categorical(probs=probs)
        idx = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(idx, device=probs.device))
    return idx, logp, entropy, probs


def select_action_with_time_window_bias(
    policy: Any,
    env: Any,
    obs: Dict[str, Any],
    cfg: TimeWindowInferenceConfig,
    *,
    greedy: bool = False,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    """Select one inference action with optional time-window logit bias."""
    debug: Dict[str, Any] = {
        "time_window_bias_enabled": bool(cfg.enable_time_window_bias),
        "j_candidates": [],
        "k_candidates": [],
        "policy_action": None,
        "biased_action": None,
        "bias_changed_action": False,
    }
    current_request = int(obs.get("current_decision_request", -1))
    with torch.no_grad():
        z, truck_mask = policy._encode(env, obs)
        decoder = policy.decoder
        ctx = torch.cat([z["truck"].squeeze(0), z["drone"].squeeze(0)], dim=0)
        q = decoder.q_j(ctx).unsqueeze(0)
        keys = decoder.k_j(z["order"])
        logits_j = (q * keys).sum(dim=-1).squeeze(0) / math.sqrt(decoder.hidden_dim)
        logits_j = _shape_logits(decoder, logits_j, greedy=greedy)
        base_logits_j = decoder.masked_logits(logits_j, truck_mask.float())

        biased_logits_j = base_logits_j.clone()
        feasible_j = torch.where(truck_mask.float() > 0)[0].detach().cpu().numpy().astype(int).tolist()
        all_late = True
        risks: Dict[int, float] = {}
        preds_j: Dict[int, Dict[str, Any]] = {}
        service_js: List[int] = []
        for j in feasible_j:
            pred = predict_action_lateness(env, obs, j=j, k=env.K_NONE, cfg=cfg)
            risk = float(pred["lateness_risk_score"]) if cfg.enable_time_window_bias else 0.0
            risks[int(j)] = risk
            preds_j[int(j)] = pred
            if j != 0 and pred.get("node_lateness"):
                service_js.append(int(j))
            all_late = all_late and bool(pred["will_be_late"])
            debug["j_candidates"].append({"j": int(j), **pred, "risk": risk})
        normalize_j_risk = 0.0
        if (
            cfg.enable_time_window_bias
            and cfg.allow_late_if_no_feasible_action
            and current_request <= 0
            and service_js
            and all(bool(preds_j[j]["will_be_late"]) for j in service_js)
        ):
            finite_risks = [float(risks[j]) for j in service_js if math.isfinite(float(risks[j]))]
            normalize_j_risk = min(finite_risks) if finite_risks else 0.0
        if cfg.enable_time_window_bias:
            for j, risk in risks.items():
                if math.isfinite(risk):
                    adjusted_risk = max(0.0, float(risk) - float(normalize_j_risk)) if j in service_js else float(risk)
                    biased_logits_j[j] = biased_logits_j[j] - adjusted_risk

        j_base, _, _, _ = _sample_from_logits(base_logits_j, greedy=greedy)
        j, logp_j, ent_j, _ = _sample_from_logits(biased_logits_j, greedy=greedy)

        dm = env.get_masks(j=j)["drone_mask"]
        drone_mask = torch.as_tensor(dm, dtype=torch.float32, device=z["order"].device)
        drone_mask[0] = 0.0
        ctx_k = torch.cat([z["truck"].squeeze(0), z["drone"].squeeze(0), z["order"][j]], dim=0)
        qk = decoder.q_k(ctx_k)
        keys_k = decoder.k_k(z["order"])
        logits_k = (keys_k * qk.unsqueeze(0)).sum(dim=-1) / math.sqrt(decoder.hidden_dim)
        logits_k = _shape_logits(decoder, logits_k, greedy=greedy)
        masked_k = decoder.masked_logits(logits_k, drone_mask)
        logit_none = (qk * decoder.k_none).sum() / math.sqrt(decoder.hidden_dim)
        logit_none = _shape_logits(decoder, logit_none, greedy=greedy)
        base_logits_k = torch.cat([masked_k, logit_none.view(1)], dim=0)
        biased_logits_k = base_logits_k.clone()

        feasible_k = torch.where(drone_mask > 0)[0].detach().cpu().numpy().astype(int).tolist()
        k_risks: Dict[int, float] = {}
        k_preds: Dict[int, Dict[str, Any]] = {}
        for k in feasible_k:
            pred = predict_action_lateness(env, obs, j=j, k=k, cfg=cfg)
            risk = float(pred["lateness_risk_score"]) if cfg.enable_time_window_bias else 0.0
            k_risks[int(k)] = risk
            k_preds[int(k)] = pred
            debug["k_candidates"].append({"k": int(k), **pred, "risk": risk})
        normalize_k_risk = 0.0
        if (
            cfg.enable_time_window_bias
            and cfg.allow_late_if_no_feasible_action
            and feasible_k
            and all(bool(k_preds[k]["will_be_late"]) for k in feasible_k)
        ):
            finite_risks = [float(k_risks[k]) for k in feasible_k if math.isfinite(float(k_risks[k]))]
            normalize_k_risk = min(finite_risks) if finite_risks else 0.0
        for k in feasible_k:
            risk = float(k_risks[k])
            if cfg.enable_time_window_bias and math.isfinite(risk):
                biased_logits_k[k] = biased_logits_k[k] - max(0.0, risk - float(normalize_k_risk))
        debug["k_candidates"].append({"k": int(env.K_NONE), "predicted_lateness": 0.0, "will_be_late": False, "lateness_risk_score": 0.0, "feasibility_reason": "no_drone"})

        k_base_idx, _, _, _ = _sample_from_logits(base_logits_k, greedy=greedy)
        k_idx, logp_k, ent_k, _ = _sample_from_logits(biased_logits_k, greedy=greedy)
        k_base = env.K_NONE if k_base_idx == base_logits_k.shape[0] - 1 else int(k_base_idx)
        k = env.K_NONE if k_idx == biased_logits_k.shape[0] - 1 else int(k_idx)

    debug["policy_action"] = [int(k_base), int(j_base)]
    debug["biased_action"] = [int(k), int(j)]
    debug["bias_changed_action"] = bool(debug["policy_action"] != debug["biased_action"])
    debug["all_feasible_j_predicted_late"] = bool(all_late) if feasible_j else False
    debug["j_risk_normalization"] = float(normalize_j_risk)
    debug["k_risk_normalization"] = float(normalize_k_risk)
    return (int(k), int(j)), debug


def _action_score(env: Any, obs: Dict[str, Any], action: Tuple[int, int], cfg: TimeWindowInferenceConfig) -> Tuple[float, float, float, float, float]:
    k, j = int(action[0]), int(action[1])
    pred = predict_action_lateness(env, obs, j=j, k=k, cfg=cfg)
    node_late = list(pred.get("node_lateness", {}).values())
    late_count = float(sum(1 for x in node_late if float(x) > 1e-9))
    max_late = float(max([0.0] + [float(x) for x in node_late]))
    total_late = float(sum(max(0.0, float(x)) for x in node_late))
    i = int(obs.get("i", 0))
    dist = 0.0
    energy = 0.0
    if j >= 0:
        try:
            dist += float(env.get_dense_edge_attr()[i, j, 0])
            energy += float(env._truck_energy(i, j, float(obs.get("truck_load", 0.0))))
        except Exception:
            dist += float(env.dist_mat[i, j])
    if k != env.K_NONE:
        dist += float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
        try:
            energy += float(env._drone_energy(i, k, j))
        except Exception:
            pass
    return (late_count, max_late, total_late, dist, energy)


def _action_service_count(env: Any, obs: Dict[str, Any], action: Tuple[int, int], cfg: TimeWindowInferenceConfig) -> int:
    try:
        pred = predict_action_lateness(env, obs, j=int(action[1]), k=int(action[0]), cfg=cfg)
        return int(len(pred.get("node_lateness", {}) or {}))
    except Exception:
        return 0


def local_repair_action(
    env: Any,
    obs: Dict[str, Any],
    action: Tuple[int, int],
    cfg: TimeWindowInferenceConfig,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    """Conservative online local repair over feasible current actions.

    It only chooses actions already allowed by the environment masks, so hard
    constraints remain protected by the environment.
    """
    if not cfg.enable_local_repair:
        return action, {"repair_enabled": False, "repair_changed_action": False}

    original = (int(action[0]), int(action[1]))
    current_request = int(obs.get("current_decision_request", -1))
    if current_request > 0:
        return original, {
            "repair_enabled": True,
            "repair_changed_action": False,
            "repair_skipped_reason": "decision_step_preserve_accept_reject",
        }

    original_service_count = _action_service_count(env, obs, original, cfg)
    candidates: List[Tuple[int, int]] = []
    masks = env.get_masks()
    feasible_j = np.where(masks["truck_mask"] > 0)[0].astype(int).tolist()

    non_depot = [j for j in feasible_j if j != 0]
    due_sorted = sorted(non_depot, key=lambda node: (_safe_due(env, int(node)), float(env.dist_mat[int(obs.get("i", 0)), int(node)])))
    j_pool = [original[1]] + due_sorted[: max(1, int(cfg.repair_window_size))]
    if original_service_count == 0 and 0 in feasible_j:
        j_pool.append(0)
    seen_j = []
    for j in j_pool:
        if j in feasible_j and j not in seen_j:
            seen_j.append(int(j))
    for j in seen_j:
        candidates.append((env.K_NONE, int(j)))
        dm = env.get_masks(j=j)["drone_mask"]
        feasible_k = np.where(dm > 0)[0].astype(int).tolist()
        feasible_k = sorted(feasible_k, key=lambda node: (_safe_due(env, int(node)), float(env.dist_mat[int(obs.get("i", 0)), int(node)])))
        for k in feasible_k[: max(1, int(cfg.repair_window_size))]:
            candidates.append((int(k), int(j)))

    if original not in candidates:
        candidates.append(original)

    original_score = _action_score(env, obs, original, cfg)
    best = original
    best_score = original_score
    for cand in candidates[: max(1, int(cfg.repair_max_iterations))]:
        try:
            if _action_service_count(env, obs, cand, cfg) < original_service_count:
                continue
            score = _action_score(env, obs, cand, cfg)
        except Exception:
            continue
        if score < best_score:
            best = cand
            best_score = score

    changed = bool(best != original)
    return best, {
        "repair_enabled": True,
        "repair_changed_action": changed,
        "original_action": [int(original[0]), int(original[1])],
        "repaired_action": [int(best[0]), int(best[1])],
        "original_score": list(original_score),
        "repaired_score": list(best_score),
        "candidate_count": int(len(candidates)),
    }
