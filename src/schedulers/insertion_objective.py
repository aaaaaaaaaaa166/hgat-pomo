from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.evaluation.v2_sequence_scoring import action_distance_energy
from src.schedulers.feasibility import classify_order_feasibility


@dataclass
class InsertionObjectiveConfig:
    accept_reward: float = 6.0
    on_time_reward: float = 8.0
    reject_penalty: float = 5.0
    late_order_penalty: float = 8.0
    lateness_weight: float = 1.2
    max_lateness_weight: float = 2.0
    future_impact_weight: float = 1.0
    energy_weight: float = 0.04
    distance_weight: float = 0.02
    severe_lateness_threshold: float = 10.0
    severe_lateness_weight: float = 4.0
    hard_constraint_penalty: float = 1_000_000.0
    acceptance_backlog_weight: float = 8.0
    min_accept_slack_after_backlog: float = 0.0
    late_accept_guard_penalty: float = 200.0
    queue_service_time_factor: float = 0.35


def _mean_service_span(env: Any) -> float:
    if int(getattr(env, "N", 0)) <= 0:
        return float(getattr(env.cfg, "sT", 0.05))
    dist = np.asarray(getattr(env, "dist_mat", np.zeros((1, 1))), dtype=np.float32)
    if dist.shape[0] <= 1:
        return float(getattr(env.cfg, "sT", 0.05))
    upper = dist[1:, 1:]
    positive = upper[upper > 1e-9]
    mean_dist = float(np.median(positive)) if positive.size > 0 else float(np.mean(dist[0, 1:]))
    return mean_dist / max(1e-9, float(env.cfg.vT)) + float(env.cfg.sT)


def _accepted_unserved_count(obs: Dict[str, Any]) -> int:
    accepted = np.asarray(obs.get("accepted", []), dtype=np.float32)
    served = np.asarray(obs.get("served", []), dtype=np.float32)
    rejected = np.asarray(obs.get("rejected", np.zeros_like(accepted)), dtype=np.float32)
    if accepted.size == 0 or served.size == 0:
        return 0
    mask = (accepted > 0.5) & (served <= 0.5) & (rejected <= 0.5)
    if mask.size > 0:
        mask[0] = False
    return int(mask.sum())


def score_action(env: Any, obs: Dict[str, Any], action: Tuple[int, int], cfg: InsertionObjectiveConfig) -> Tuple[float, Dict[str, Any]]:
    k, j = int(action[0]), int(action[1])
    current = int(obs.get("current_decision_request", -1))
    if current > 0 and j == 0:
        feas = classify_order_feasibility(env, current)
        reject_penalty = 0.0 if feas.hard_infeasible else float(cfg.reject_penalty)
        return reject_penalty, {"decision": "reject", "feasibility": feas.to_dict(), "score": reject_penalty}

    nodes = [n for n in (j, k) if n > 0 and n != getattr(env, "K_NONE", -1)]
    details: List[Dict[str, Any]] = []
    late_count = 0
    total_late = 0.0
    max_late = 0.0
    future = 0.0
    hard = False
    for node in nodes:
        feas = classify_order_feasibility(env, node)
        details.append(feas.to_dict())
        hard = hard or bool(feas.hard_infeasible)
        late = max(0.0, float(feas.predicted_lateness))
        if late > 1e-9:
            late_count += 1
        total_late += late
        max_late = max(max_late, late)
        future += max(0.0, float(feas.future_impact_score))
    distance, energy = action_distance_energy(env, obs, action)
    if hard:
        return float(cfg.hard_constraint_penalty), {"hard_infeasible": True, "nodes": details}
    on_time = max(0, len(nodes) - late_count)
    accept = 1 if current > 0 and j > 0 else 0
    severe = max(0.0, max_late - float(cfg.severe_lateness_threshold))
    backlog_risk = 0.0
    guard_penalty = 0.0
    if accept and details:
        # Accepting a dynamic order is not the same as serving it immediately.
        # Estimate the waiting pressure from already accepted/unserved orders so
        # the scheduler can distinguish feasible accepts from likely late accepts.
        queue_delay = float(cfg.queue_service_time_factor) * float(_accepted_unserved_count(obs)) * _mean_service_span(env)
        slack_after_queue = float(details[0].get("slack_after_arrival", 0.0)) - queue_delay
        backlog_risk = max(0.0, -slack_after_queue)
        if slack_after_queue < float(cfg.min_accept_slack_after_backlog):
            guard_penalty = float(cfg.late_accept_guard_penalty) * (
                float(cfg.min_accept_slack_after_backlog) - slack_after_queue
            )
    score = 0.0
    score -= float(cfg.accept_reward) * float(accept)
    score -= float(cfg.on_time_reward) * float(on_time)
    score += float(cfg.late_order_penalty) * float(late_count)
    score += float(cfg.lateness_weight) * float(total_late)
    score += float(cfg.max_lateness_weight) * float(max_late)
    score += float(cfg.severe_lateness_weight) * float(severe)
    score += float(cfg.future_impact_weight) * float(future)
    score += float(cfg.acceptance_backlog_weight) * float(backlog_risk)
    score += float(guard_penalty)
    score += float(cfg.energy_weight) * float(energy)
    score += float(cfg.distance_weight) * float(distance)
    return float(score), {
        "nodes": details,
        "on_time": int(on_time),
        "late_count": int(late_count),
        "total_lateness": float(total_late),
        "max_lateness": float(max_late),
        "future_impact": float(future),
        "backlog_risk": float(backlog_risk),
        "guard_penalty": float(guard_penalty),
        "energy": float(energy),
        "distance": float(distance),
        "score": float(score),
    }


def simulate_action_score(env: Any, obs: Dict[str, Any], action: Tuple[int, int], cfg: InsertionObjectiveConfig) -> Tuple[float, Dict[str, Any]]:
    pre_score, pre_detail = score_action(env, obs, action, cfg)
    if pre_score >= float(cfg.hard_constraint_penalty):
        return pre_score, pre_detail
    try:
        e2 = env.copy()
        _, reward, _, info = e2.step(action)
    except Exception as exc:
        return float(cfg.hard_constraint_penalty), {"error": str(exc), "action": [int(action[0]), int(action[1])]}
    service_late = [max(0.0, float(x)) for x in (info.get("service_lateness", {}) or {}).values()]
    late_count = sum(1 for x in service_late if x > 1e-9)
    max_late = max([0.0] + service_late)
    score = pre_score
    score += 0.5 * float(info.get("dt", 0.0))
    score += float(cfg.late_order_penalty) * late_count
    score += float(cfg.max_lateness_weight) * max_late
    detail = dict(pre_detail)
    detail.update({"sim_reward": float(reward), "sim_info": info, "sim_score": float(score)})
    return float(score), detail
