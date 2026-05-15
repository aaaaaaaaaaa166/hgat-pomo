from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.schedulers.feasibility import classify_order_feasibility
from src.schedulers.joint_beam_objective import (
    JointBeamMetrics,
    JointBeamObjectiveConfig,
    add_metrics,
    average_lateness,
    dominance_prune,
    merge_transition_metrics,
    pending_order_risk_score,
    score_metrics,
    transition_metrics,
)


@dataclass
class JointAcceptRouteBeamConfig:
    beam_size: int = 16
    lookahead_depth: int = 3
    candidate_top_k: int = 10
    max_expanded_states: int = 5000
    time_limit_seconds: float = 2.0
    enable_dominance_pruning: bool = True
    enable_acceptance_guard: bool = False
    guard_max_lateness_factor: float = 1.05
    guard_distance_factor: float = 1.10
    guard_energy_factor: float = 1.10
    guard_abs_lateness_slack: float = 1e-6
    guard_abs_distance_slack: float = 1e-6
    guard_abs_energy_slack: float = 1e-6
    guard_sim_max_steps: int = 96
    objective: JointBeamObjectiveConfig = field(default_factory=JointBeamObjectiveConfig)


@dataclass
class TailRiskBudget:
    baseline_acceptance_rate: float = 0.0
    baseline_on_time_rate: float = 0.0
    baseline_late_orders: int = 0
    baseline_average_lateness: float = 0.0
    baseline_max_lateness: float = 0.0
    baseline_total_energy: float = 0.0
    baseline_total_distance: float = 0.0
    baseline_severe_late_count: int = 0
    max_lateness_ratio: float = 1.02
    avg_lateness_ratio: float = 1.02
    energy_ratio: float = 1.03
    distance_ratio: float = 1.03
    severe_lateness_threshold: float = 30.0
    severe_lateness_hard_cap: float = 0.0
    enable_baseline_anchored_budget: bool = True

    def caps(self) -> Dict[str, float]:
        max_late_cap = float(self.baseline_max_lateness) * float(self.max_lateness_ratio)
        severe_cap = float(self.severe_lateness_hard_cap) if float(self.severe_lateness_hard_cap) > 0.0 else max_late_cap
        return {
            "max_lateness": max_late_cap,
            "average_lateness": float(self.baseline_average_lateness) * float(self.avg_lateness_ratio),
            "total_energy": float(self.baseline_total_energy) * float(self.energy_ratio),
            "total_distance": float(self.baseline_total_distance) * float(self.distance_ratio),
            "severe_lateness": severe_cap,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_acceptance_rate": float(self.baseline_acceptance_rate),
            "baseline_on_time_rate": float(self.baseline_on_time_rate),
            "baseline_late_orders": int(self.baseline_late_orders),
            "baseline_average_lateness": float(self.baseline_average_lateness),
            "baseline_max_lateness": float(self.baseline_max_lateness),
            "baseline_total_energy": float(self.baseline_total_energy),
            "baseline_total_distance": float(self.baseline_total_distance),
            "baseline_severe_late_count": int(self.baseline_severe_late_count),
            "max_lateness_ratio": float(self.max_lateness_ratio),
            "avg_lateness_ratio": float(self.avg_lateness_ratio),
            "energy_ratio": float(self.energy_ratio),
            "distance_ratio": float(self.distance_ratio),
            "severe_lateness_threshold": float(self.severe_lateness_threshold),
            "severe_lateness_hard_cap": float(self.severe_lateness_hard_cap),
            "enable_baseline_anchored_budget": bool(self.enable_baseline_anchored_budget),
            "caps": self.caps(),
        }


@dataclass
class JointBeamState:
    env: Any
    obs: Dict[str, Any]
    first_action: Optional[Tuple[int, int]] = None
    actions: List[Tuple[int, int]] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    metrics: JointBeamMetrics = field(default_factory=JointBeamMetrics)
    score: float = 0.0
    risk_components: Dict[str, float] = field(default_factory=dict)
    done: bool = False

    def debug_snapshot(self) -> Dict[str, Any]:
        accepted = np.asarray(self.obs.get("accepted", []), dtype=np.int8)
        rejected = np.asarray(self.obs.get("rejected", []), dtype=np.int8)
        expired = np.asarray(self.obs.get("expired", np.zeros_like(rejected)), dtype=np.int8)
        served = np.asarray(self.obs.get("served", []), dtype=np.int8)
        pending = [int(x) for x in self.obs.get("pending_queue", [])]
        accepted_unserved: List[int] = []
        if accepted.size and served.size:
            mask = (accepted > 0) & (served <= 0) & (rejected <= 0)
            accepted_unserved = [int(x) for x in np.where(mask)[0] if int(x) > 0]
        return {
            "current_time": float(self.obs.get("t", 0.0)),
            "current_position": int(self.obs.get("i", 0)),
            "pending_orders": pending,
            "accepted_unserved_orders": accepted_unserved,
            "rejected_orders": [int(x) for x in np.where(rejected > 0)[0] if int(x) > 0],
            "expired_orders": [int(x) for x in np.where(expired > 0)[0] if int(x) > 0],
            "served_orders": [int(x) for x in np.where(served > 0)[0] if int(x) > 0],
            "route_sequence": [[int(a[0]), int(a[1])] for a in self.actions],
            "response_deadlines": _deadline_map(self.env, "decision_deadline"),
            "delivery_deadlines": _deadline_map(self.env, "due"),
            "score": float(self.score),
            "score_components": self.metrics.to_dict(),
            "risk_components": dict(self.risk_components),
        }


def _deadline_map(env: Any, attr: str) -> Dict[str, float]:
    arr = np.asarray(getattr(env, attr, []), dtype=np.float32)
    out: Dict[str, float] = {}
    for node in range(1, min(len(arr), int(getattr(env, "N", 0)) + 1)):
        val = float(arr[node])
        if np.isfinite(val):
            out[str(int(node))] = val
    return out


def _is_action_feasible(env: Any, action: Tuple[int, int]) -> bool:
    k, j = int(action[0]), int(action[1])
    try:
        masks = env.get_masks()
        if not (0 <= j <= int(env.N)) or int(masks["truck_mask"][j]) == 0:
            return False
        if k == env.K_NONE:
            return True
        dm = env.get_masks(j=j)["drone_mask"]
        return bool(1 <= k <= int(env.N) and int(dm[k]) > 0)
    except Exception:
        return False


def _accepted_unserved(obs: Dict[str, Any]) -> List[int]:
    accepted = np.asarray(obs.get("accepted", []), dtype=np.int8)
    served = np.asarray(obs.get("served", []), dtype=np.int8)
    rejected = np.asarray(obs.get("rejected", np.zeros_like(accepted)), dtype=np.int8)
    if accepted.size == 0 or served.size == 0:
        return []
    mask = (accepted > 0) & (served <= 0) & (rejected <= 0)
    return [int(x) for x in np.where(mask)[0] if int(x) > 0]


def _rank_service_nodes(env: Any, obs: Dict[str, Any], nodes: Sequence[int]) -> List[int]:
    i = int(obs.get("i", 0))

    def key(node: int) -> Tuple[float, float, int]:
        due = float(env.due[node]) if np.isfinite(float(env.due[node])) else 1e9
        dist = float(env.dist_mat[i, node])
        return (due, dist, int(node))

    return sorted([int(x) for x in nodes], key=key)


def _candidate_route_actions(env: Any, obs: Dict[str, Any], cfg: JointAcceptRouteBeamConfig) -> List[Dict[str, Any]]:
    masks = env.get_masks()
    feasible_j = np.where(np.asarray(masks["truck_mask"]) > 0)[0].astype(int).tolist()
    accepted_unserved = set(_accepted_unserved(obs))
    ranked_j = _rank_service_nodes(env, obs, [j for j in feasible_j if j in accepted_unserved])
    actions: List[Dict[str, Any]] = []
    for j in ranked_j[: max(1, int(cfg.candidate_top_k))]:
        actions.append({"action": (env.K_NONE, int(j)), "action_type": "serve_accepted", "order_id": int(j)})
        try:
            dm = np.asarray(env.get_masks(j=j)["drone_mask"])
            feasible_k = np.where(dm > 0)[0].astype(int).tolist()
        except Exception:
            feasible_k = []
        ranked_k = _rank_service_nodes(env, obs, [k for k in feasible_k if k in accepted_unserved and k != j])
        for k in ranked_k[:3]:
            actions.append({"action": (int(k), int(j)), "action_type": "serve_accepted", "order_id": int(j), "drone_order_id": int(k)})
    if 0 in feasible_j:
        actions.append({"action": (env.K_NONE, 0), "action_type": "wait" if not ranked_j else "return_depot", "order_id": 0})
    return _dedupe_actions(actions)[: max(1, int(cfg.candidate_top_k))]


def _dedupe_actions(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        key = tuple(item["action"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def _candidate_actions(env: Any, obs: Dict[str, Any], cfg: JointAcceptRouteBeamConfig) -> List[Dict[str, Any]]:
    current = int(obs.get("current_decision_request", -1))
    if current > 0:
        items: List[Dict[str, Any]] = []
        guard_accept = True
        guard_debug: Dict[str, Any] = {}
        if bool(cfg.enable_acceptance_guard):
            guard_accept, guard_debug = _acceptance_guard(env, current, cfg)
        if _is_action_feasible(env, (env.K_NONE, 0)):
            feas = classify_order_feasibility(env, current)
            reason = feas.reject_reason or ("hard_infeasible" if feas.hard_infeasible else "beam_reject")
            items.append(
                {
                    "action": (env.K_NONE, 0),
                    "action_type": "reject",
                    "order_id": int(current),
                    "reject_reason": reason,
                    "feasibility": feas.to_dict(),
                }
            )
        if guard_accept and _is_action_feasible(env, (env.K_NONE, current)):
            feas = classify_order_feasibility(env, current)
            items.append(
                {
                    "action": (env.K_NONE, int(current)),
                    "action_type": "accept_only",
                    "order_id": int(current),
                    "feasibility": feas.to_dict(),
                    "acceptance_guard": guard_debug,
                }
            )
            items.append(
                {
                    "action": (env.K_NONE, int(current)),
                    "action_type": "accept_and_serve",
                    "order_id": int(current),
                    "feasibility": feas.to_dict(),
                    "composite_serve_order": int(current),
                    "acceptance_guard": guard_debug,
                }
            )
        elif bool(cfg.enable_acceptance_guard) and items:
            items[0]["acceptance_guard"] = guard_debug
        return _dedupe_actions(items)
    return _candidate_route_actions(env, obs, cfg)


def _metrics_dict(metrics: JointBeamMetrics) -> Dict[str, Any]:
    return metrics.to_dict()


def _guard_route_score(env: Any, obs: Dict[str, Any], action: Tuple[int, int], cfg: JointAcceptRouteBeamConfig) -> Tuple[float, Dict[str, Any]]:
    sim = _apply_one(env, action)
    if sim is None:
        return float("inf"), {"infeasible": True}
    env2, _, _, _, info = sim
    metrics = transition_metrics(env2, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold))
    i = int(obs.get("i", 0))
    j = int(action[1])
    k = int(action[0])
    nodes = [x for x in (j, k) if x not in {env.K_NONE, 0}]
    min_due = min([float(env.due[n]) if np.isfinite(float(env.due[n])) else 1e9 for n in nodes] + [1e9])
    dist_proxy = float(env.dist_mat[i, j]) if 0 <= j <= int(env.N) else 0.0
    if k != env.K_NONE and 1 <= k <= int(env.N):
        dist_proxy += float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
    score = (
        1_000_000.0 * float(metrics.hard_constraint_violations)
        + 10_000.0 * float(metrics.late_orders)
        + 500.0 * float(metrics.max_lateness)
        + 100.0 * float(metrics.total_lateness)
        + 0.10 * float(metrics.total_distance)
        + 0.02 * dist_proxy
        + 0.01 * min_due
        - 25.0 * float(metrics.on_time_orders)
    )
    return float(score), {"step_metrics": metrics.to_dict(), "min_due": float(min_due), "dist_proxy": float(dist_proxy)}


def _select_guard_route_action(env: Any, obs: Dict[str, Any], cfg: JointAcceptRouteBeamConfig) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    candidates = _candidate_route_actions(env, obs, cfg)
    if not candidates:
        return (env.K_NONE, 0), {"candidate_count": 0, "fallback": True}
    ranked: List[Dict[str, Any]] = []
    for item in candidates:
        action = (int(item["action"][0]), int(item["action"][1]))
        score, detail = _guard_route_score(env, obs, action, cfg)
        ranked.append({"action": action, "score": float(score), "action_type": item.get("action_type", ""), **detail})
    ranked.sort(key=lambda x: x["score"])
    best = ranked[0]
    return (int(best["action"][0]), int(best["action"][1])), {
        "candidate_count": len(ranked),
        "selected_score": float(best["score"]),
        "selected_action_type": str(best.get("action_type", "")),
        "candidates": [
            {
                "action": [int(x["action"][0]), int(x["action"][1])],
                "score": float(x["score"]),
                "action_type": str(x.get("action_type", "")),
            }
            for x in ranked[:10]
        ],
    }


def _simulate_backlog_after_decision(
    env: Any,
    action: Tuple[int, int],
    cfg: JointAcceptRouteBeamConfig,
) -> Dict[str, Any]:
    e = env.copy()
    metrics = JointBeamMetrics()
    steps: List[Dict[str, Any]] = []
    done = False
    first = _apply_one(e, action)
    if first is None:
        metrics.hard_constraint_violations += 1
        return {
            "done": False,
            "decision_action_feasible": False,
            "metrics": metrics,
            "steps": steps,
            "accepted_unserved": _accepted_unserved(e.get_obs()),
        }
    e, obs, _, done, info = first
    metrics = add_metrics(metrics, transition_metrics(e, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold)))
    steps.append({"action": [int(action[0]), int(action[1])], "phase": str(info.get("phase", "")), "info": "decision"})

    for _ in range(max(1, int(cfg.guard_sim_max_steps))):
        if done:
            break
        obs = e.get_obs()
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            reject_action = (e.K_NONE, 0)
            one = _apply_one(e, reject_action)
            if one is None:
                metrics.hard_constraint_violations += 1
                break
            e, obs, _, done, info = one
            metrics = add_metrics(metrics, transition_metrics(e, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold)))
            steps.append(
                {
                    "action": [int(reject_action[0]), int(reject_action[1])],
                    "phase": str(info.get("phase", "")),
                    "info": "guard_reject_future_request",
                }
            )
            continue
        backlog = _accepted_unserved(obs)
        if not backlog:
            break
        route_action, route_debug = _select_guard_route_action(e, obs, cfg)
        one = _apply_one(e, route_action)
        if one is None:
            metrics.hard_constraint_violations += 1
            break
        e, obs, _, done, info = one
        metrics = add_metrics(metrics, transition_metrics(e, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold)))
        steps.append(
            {
                "action": [int(route_action[0]), int(route_action[1])],
                "phase": str(info.get("phase", "")),
                "served_nodes": [int(x) for x in info.get("served_nodes", []) or []],
                "route_debug": route_debug,
            }
        )
    else:
        metrics.hard_constraint_violations += 1
        steps.append({"action": ["TIMEOUT"], "phase": "guard_timeout"})

    final_obs = e.get_obs()
    remaining = _accepted_unserved(final_obs)
    if remaining:
        metrics.hard_constraint_violations += 1
    return {
        "done": bool(done),
        "decision_action_feasible": True,
        "metrics": metrics,
        "steps": steps[:20],
        "accepted_unserved": remaining,
    }


def _within_factor(candidate: float, baseline: float, factor: float, abs_slack: float) -> bool:
    return float(candidate) <= float(baseline) * float(factor) + float(abs_slack) + 1e-9


def _acceptance_guard(env: Any, order_id: int, cfg: JointAcceptRouteBeamConfig) -> Tuple[bool, Dict[str, Any]]:
    feas = classify_order_feasibility(env, int(order_id))
    reject_sim = _simulate_backlog_after_decision(env, (env.K_NONE, 0), cfg)
    accept_sim = _simulate_backlog_after_decision(env, (env.K_NONE, int(order_id)), cfg)
    reject_m: JointBeamMetrics = reject_sim["metrics"]
    accept_m: JointBeamMetrics = accept_sim["metrics"]
    checks = {
        "hard_feasible_now": not bool(feas.hard_infeasible),
        "predicted_on_time_now": float(feas.predicted_lateness) <= 1e-9,
        "guard_hard_zero": int(accept_m.hard_constraint_violations) == 0,
        "late_orders_no_increase": int(accept_m.late_orders) <= int(reject_m.late_orders),
        "max_lateness_within_factor": _within_factor(
            accept_m.max_lateness,
            reject_m.max_lateness,
            cfg.guard_max_lateness_factor,
            cfg.guard_abs_lateness_slack,
        ),
        "distance_within_factor": _within_factor(
            accept_m.total_distance,
            reject_m.total_distance,
            cfg.guard_distance_factor,
            cfg.guard_abs_distance_slack,
        ),
        "energy_within_factor": _within_factor(
            accept_m.total_energy,
            reject_m.total_energy,
            cfg.guard_energy_factor,
            cfg.guard_abs_energy_slack,
        ),
    }
    passed = bool(all(checks.values()))
    return passed, {
        "order_id": int(order_id),
        "passed": passed,
        "checks": checks,
        "feasibility": feas.to_dict(),
        "thresholds": {
            "max_lateness_factor": float(cfg.guard_max_lateness_factor),
            "distance_factor": float(cfg.guard_distance_factor),
            "energy_factor": float(cfg.guard_energy_factor),
            "sim_max_steps": int(cfg.guard_sim_max_steps),
        },
        "reject_metrics": _metrics_dict(reject_m),
        "accept_metrics": _metrics_dict(accept_m),
        "reject_remaining": [int(x) for x in reject_sim.get("accepted_unserved", [])],
        "accept_remaining": [int(x) for x in accept_sim.get("accepted_unserved", [])],
        "accept_trace": accept_sim.get("steps", [])[:8],
    }


def _tail_risk_violations(metrics: JointBeamMetrics, budget: TailRiskBudget) -> Dict[str, float]:
    caps = budget.caps()
    avg_late = average_lateness(metrics)
    violations: Dict[str, float] = {}
    if int(metrics.hard_constraint_violations) > 0:
        violations["hard_constraint_violations"] = float(metrics.hard_constraint_violations)
    if not bool(budget.enable_baseline_anchored_budget):
        return violations
    if float(metrics.max_lateness) > caps["max_lateness"] + 1e-9:
        violations["max_lateness"] = float(metrics.max_lateness) - caps["max_lateness"]
    if avg_late > caps["average_lateness"] + 1e-9:
        violations["average_lateness"] = avg_late - caps["average_lateness"]
    if float(metrics.total_energy) > caps["total_energy"] + 1e-9:
        violations["total_energy"] = float(metrics.total_energy) - caps["total_energy"]
    if float(metrics.total_distance) > caps["total_distance"] + 1e-9:
        violations["total_distance"] = float(metrics.total_distance) - caps["total_distance"]
    if float(metrics.max_lateness) > caps["severe_lateness"] + 1e-9:
        violations["severe_lateness_hard_cap"] = float(metrics.max_lateness) - caps["severe_lateness"]
    if int(metrics.severe_late_count) > int(budget.baseline_severe_late_count):
        violations["new_severe_late_count"] = float(int(metrics.severe_late_count) - int(budget.baseline_severe_late_count))
    return violations


def _tail_risk_violation_score(violations: Dict[str, float]) -> float:
    weights = {
        "hard_constraint_violations": 1_000_000_000.0,
        "severe_lateness_hard_cap": 100_000_000.0,
        "new_severe_late_count": 50_000_000.0,
        "max_lateness": 1_000_000.0,
        "average_lateness": 500_000.0,
        "total_energy": 50_000.0,
        "total_distance": 25_000.0,
    }
    return float(sum(float(weights.get(k, 10_000.0)) * max(0.0, float(v)) for k, v in violations.items()))


def _tail_risk_rank_key(metrics: JointBeamMetrics, budget: TailRiskBudget) -> Tuple[float, ...]:
    violations = _tail_risk_violations(metrics, budget)
    return (
        float(int(bool(violations))),
        _tail_risk_violation_score(violations),
        float(metrics.hard_constraint_violations),
        float(metrics.severe_late_count),
        float(metrics.max_lateness),
        average_lateness(metrics),
        float(metrics.late_orders),
        -float(metrics.on_time_orders),
        -float(metrics.accepted_orders),
        float(metrics.total_energy),
        float(metrics.total_distance),
    )


def _tail_risk_accept_allowed(
    *,
    reject_total: JointBeamMetrics,
    accept_total: JointBeamMetrics,
    budget: TailRiskBudget,
) -> Tuple[bool, Dict[str, Any]]:
    reject_avg = average_lateness(reject_total)
    accept_avg = average_lateness(accept_total)
    violations = _tail_risk_violations(accept_total, budget)
    tail_ok = not violations
    improves_service = (
        int(accept_total.late_orders) < int(reject_total.late_orders)
        or int(accept_total.on_time_orders) > int(reject_total.on_time_orders)
    )
    no_new_severe_late = int(accept_total.severe_late_count) <= int(reject_total.severe_late_count)
    no_tail_worse = (
        int(accept_total.late_orders) <= int(reject_total.late_orders)
        and int(accept_total.severe_late_count) <= int(reject_total.severe_late_count)
        and float(accept_total.max_lateness) <= float(reject_total.max_lateness) + 1e-9
        and accept_avg <= reject_avg + 1e-9
    )
    harmful_for_acceptance_only = (
        int(accept_total.accepted_orders) > int(reject_total.accepted_orders)
        and not improves_service
        and (
            int(accept_total.late_orders) > int(reject_total.late_orders)
            or int(accept_total.severe_late_count) > int(reject_total.severe_late_count)
            or float(accept_total.max_lateness) > float(reject_total.max_lateness) + 1e-9
            or accept_avg > reject_avg + 1e-9
        )
    )
    allowed = bool(tail_ok and no_new_severe_late and (improves_service or no_tail_worse) and not harmful_for_acceptance_only)
    return allowed, {
        "allowed": allowed,
        "tail_ok": tail_ok,
        "improves_service": bool(improves_service),
        "no_new_severe_late": bool(no_new_severe_late),
        "no_tail_worse": bool(no_tail_worse),
        "harmful_for_acceptance_only": bool(harmful_for_acceptance_only),
        "violations": violations,
        "reject_metrics": reject_total.to_dict(),
        "accept_metrics": accept_total.to_dict(),
        "rank_key": list(_tail_risk_rank_key(accept_total, budget)),
    }


def _select_tail_risk_route_action(
    env: Any,
    obs: Dict[str, Any],
    cfg: JointAcceptRouteBeamConfig,
    budget: TailRiskBudget,
    cumulative: JointBeamMetrics,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    candidates = _candidate_route_actions(env, obs, cfg)
    if not candidates:
        return (env.K_NONE, 0), {"mode": "tail_risk_route", "fallback": True, "candidate_count": 0}
    ranked: List[Dict[str, Any]] = []
    for item in candidates:
        action = (int(item["action"][0]), int(item["action"][1]))
        sim = _simulate_backlog_after_route_action(env, action, cfg)
        total = add_metrics(cumulative, sim["metrics"])
        ranked.append(
            {
                "action": action,
                "action_type": str(item.get("action_type", "")),
                "score_key": _tail_risk_rank_key(total, budget),
                "future_metrics": sim["metrics"].to_dict(),
                "total_metrics": total.to_dict(),
                "violations": _tail_risk_violations(total, budget),
                "trace": sim.get("steps", [])[:6],
            }
        )
    ranked.sort(key=lambda x: x["score_key"])
    best = ranked[0]
    action = best["action"]
    return (int(action[0]), int(action[1])), {
        "mode": "tail_risk_route",
        "candidate_count": len(ranked),
        "selected_action": [int(action[0]), int(action[1])],
        "selected_action_type": str(best.get("action_type", "")),
        "selected_score_key": [float(x) for x in best["score_key"]],
        "selected_total_metrics": best["total_metrics"],
        "selected_violations": best["violations"],
        "selected_trace": best.get("trace", []),
        "candidates": [
            {
                "action": [int(x["action"][0]), int(x["action"][1])],
                "action_type": str(x.get("action_type", "")),
                "score_key": [float(v) for v in x["score_key"]],
                "violations": x["violations"],
            }
            for x in ranked[:10]
        ],
    }


def select_tail_risk_constrained_joint_action(
    env: Any,
    obs: Dict[str, Any],
    cfg: JointAcceptRouteBeamConfig,
    budget: TailRiskBudget,
    cumulative: JointBeamMetrics,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    current = int(obs.get("current_decision_request", -1))
    if current <= 0:
        return _select_tail_risk_route_action(env, obs, cfg, budget, cumulative)

    candidates: List[Dict[str, Any]] = []
    reject_action = (env.K_NONE, 0)
    reject_sim = _simulate_backlog_after_decision(env, reject_action, cfg)
    reject_total = add_metrics(cumulative, reject_sim["metrics"])
    if _is_action_feasible(env, reject_action):
        candidates.append(
            {
                "action": reject_action,
                "action_type": "reject",
                "score_key": _tail_risk_rank_key(reject_total, budget),
                "future_metrics": reject_sim["metrics"].to_dict(),
                "total_metrics": reject_total.to_dict(),
                "violations": _tail_risk_violations(reject_total, budget),
                "acceptance_rule": {"allowed": True, "reason": "reject_reference"},
            }
        )

    accept_action = (env.K_NONE, int(current))
    if _is_action_feasible(env, accept_action):
        accept_sim = _simulate_backlog_after_decision(env, accept_action, cfg)
        accept_total = add_metrics(cumulative, accept_sim["metrics"])
        allowed, rule = _tail_risk_accept_allowed(reject_total=reject_total, accept_total=accept_total, budget=budget)
        if allowed:
            candidates.append(
                {
                    "action": accept_action,
                    "action_type": "accept",
                    "score_key": _tail_risk_rank_key(accept_total, budget),
                    "future_metrics": accept_sim["metrics"].to_dict(),
                    "total_metrics": accept_total.to_dict(),
                    "violations": _tail_risk_violations(accept_total, budget),
                    "acceptance_rule": rule,
                }
            )
        else:
            candidates[0]["blocked_accept"] = rule

    if not candidates:
        return reject_action, {"mode": "tail_risk_acceptance", "fallback": True, "candidate_count": 0}
    candidates.sort(key=lambda x: x["score_key"])
    best = candidates[0]
    action = best["action"]
    return (int(action[0]), int(action[1])), {
        "mode": "tail_risk_acceptance",
        "order_id": int(current),
        "candidate_count": len(candidates),
        "selected_action": [int(action[0]), int(action[1])],
        "selected_action_type": str(best.get("action_type", "")),
        "selected_score_key": [float(x) for x in best["score_key"]],
        "selected_total_metrics": best["total_metrics"],
        "selected_violations": best["violations"],
        "acceptance_rule": best.get("acceptance_rule", {}),
        "blocked_accept": candidates[0].get("blocked_accept", {}),
        "budget": budget.to_dict(),
        "candidates": [
            {
                "action": [int(x["action"][0]), int(x["action"][1])],
                "action_type": str(x.get("action_type", "")),
                "score_key": [float(v) for v in x["score_key"]],
                "violations": x["violations"],
            }
            for x in candidates[:10]
        ],
    }


def _simulate_backlog_after_route_action(
    env: Any,
    action: Tuple[int, int],
    cfg: JointAcceptRouteBeamConfig,
) -> Dict[str, Any]:
    e = env.copy()
    metrics = JointBeamMetrics()
    steps: List[Dict[str, Any]] = []
    done = False
    first = _apply_one(e, action)
    if first is None:
        metrics.hard_constraint_violations += 1
        return {"metrics": metrics, "steps": steps, "accepted_unserved": _accepted_unserved(e.get_obs())}
    e, obs, _, done, info = first
    metrics = add_metrics(metrics, transition_metrics(e, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold)))
    steps.append(
        {
            "action": [int(action[0]), int(action[1])],
            "phase": str(info.get("phase", "")),
            "served_nodes": [int(x) for x in info.get("served_nodes", []) or []],
        }
    )

    for _ in range(max(1, int(cfg.guard_sim_max_steps))):
        if done:
            break
        obs = e.get_obs()
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            reject_action = (e.K_NONE, 0)
            one = _apply_one(e, reject_action)
            if one is None:
                metrics.hard_constraint_violations += 1
                break
            e, obs, _, done, info = one
            metrics = add_metrics(metrics, transition_metrics(e, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold)))
            steps.append(
                {
                    "action": [int(reject_action[0]), int(reject_action[1])],
                    "phase": str(info.get("phase", "")),
                    "info": "guard_reject_future_request",
                }
            )
            continue
        backlog = _accepted_unserved(obs)
        if not backlog:
            break
        next_action, route_debug = _select_guard_route_action(e, obs, cfg)
        one = _apply_one(e, next_action)
        if one is None:
            metrics.hard_constraint_violations += 1
            break
        e, obs, _, done, info = one
        metrics = add_metrics(metrics, transition_metrics(e, info, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold)))
        steps.append(
            {
                "action": [int(next_action[0]), int(next_action[1])],
                "phase": str(info.get("phase", "")),
                "served_nodes": [int(x) for x in info.get("served_nodes", []) or []],
                "route_debug": route_debug,
            }
        )
    else:
        metrics.hard_constraint_violations += 1
        steps.append({"action": ["TIMEOUT"], "phase": "guard_timeout"})

    remaining = _accepted_unserved(e.get_obs())
    if remaining:
        metrics.hard_constraint_violations += 1
    return {"metrics": metrics, "steps": steps[:20], "accepted_unserved": remaining}


def _guard_full_route_score(metrics: JointBeamMetrics) -> float:
    return float(
        1_000_000.0 * float(metrics.hard_constraint_violations)
        + 10_000.0 * float(metrics.late_orders)
        + 750.0 * float(metrics.max_lateness)
        + 150.0 * float(metrics.total_lateness)
        + 0.50 * float(metrics.total_distance)
        + 0.25 * float(metrics.total_energy)
        - 25.0 * float(metrics.on_time_orders)
    )


def _select_guarded_route_action(
    env: Any,
    obs: Dict[str, Any],
    cfg: JointAcceptRouteBeamConfig,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    candidates = _candidate_route_actions(env, obs, cfg)
    if not candidates:
        return (env.K_NONE, 0), {"mode": "joint_accept_route_beam_guarded_route", "fallback": True, "candidate_count": 0}
    ranked: List[Dict[str, Any]] = []
    for item in candidates:
        action = (int(item["action"][0]), int(item["action"][1]))
        sim = _simulate_backlog_after_route_action(env, action, cfg)
        metrics: JointBeamMetrics = sim["metrics"]
        ranked.append(
            {
                "action": action,
                "action_type": str(item.get("action_type", "")),
                "order_id": int(item.get("order_id", 0)),
                "score": _guard_full_route_score(metrics),
                "metrics": metrics.to_dict(),
                "remaining": [int(x) for x in sim.get("accepted_unserved", [])],
                "trace": sim.get("steps", [])[:6],
            }
        )
    ranked.sort(
        key=lambda x: (
            float(x["score"]),
            int(x["metrics"].get("late_orders", 0)),
            float(x["metrics"].get("max_lateness", 0.0)),
            float(x["metrics"].get("total_distance", 0.0)),
        )
    )
    best = ranked[0]
    action = best["action"]
    return (int(action[0]), int(action[1])), {
        "mode": "joint_accept_route_beam_guarded_route",
        "candidate_count": len(ranked),
        "selected_action": [int(action[0]), int(action[1])],
        "selected_action_type": str(best.get("action_type", "")),
        "selected_score": float(best["score"]),
        "selected_metrics": best["metrics"],
        "selected_trace": best.get("trace", []),
        "candidates": [
            {
                "action": [int(x["action"][0]), int(x["action"][1])],
                "action_type": str(x.get("action_type", "")),
                "score": float(x["score"]),
                "metrics": x["metrics"],
            }
            for x in ranked[:10]
        ],
    }


def _apply_one(env: Any, action: Tuple[int, int]) -> Optional[Tuple[Any, Dict[str, Any], float, bool, Dict[str, Any]]]:
    if not _is_action_feasible(env, action):
        return None
    e2 = env.copy()
    obs2, reward, done, info = e2.step(action)
    return e2, obs2, float(reward), bool(done), info


def _one_step_metrics(env: Any, action: Tuple[int, int], severe_lateness_threshold: float) -> JointBeamMetrics:
    sim = _apply_one(env, action)
    if sim is None:
        bad = JointBeamMetrics()
        bad.hard_constraint_violations = 1
        return bad
    env2, _, _, _, info = sim
    return transition_metrics(env2, info, severe_lateness_threshold=float(severe_lateness_threshold))


def _parse_anchor_action(action: Any, env: Any) -> Optional[Tuple[int, int]]:
    try:
        if isinstance(action, (list, tuple)) and len(action) == 2:
            return int(action[0]), int(action[1])
    except Exception:
        return None
    return None


def _anchor_deviation_allowed(
    *,
    cumulative: JointBeamMetrics,
    candidate_delta: JointBeamMetrics,
    anchor_delta: JointBeamMetrics,
    budget: TailRiskBudget,
) -> Tuple[bool, Dict[str, Any]]:
    candidate_total = add_metrics(cumulative, candidate_delta)
    anchor_total = add_metrics(cumulative, anchor_delta)
    candidate_violations = _tail_risk_violations(candidate_total, budget)
    candidate_avg = average_lateness(candidate_total)
    anchor_avg = average_lateness(anchor_total)
    checks = {
        "candidate_budget_ok": not bool(candidate_violations),
        "accepted_not_below_anchor": int(candidate_total.accepted_orders) >= int(anchor_total.accepted_orders),
        "on_time_not_below_anchor": int(candidate_total.on_time_orders) >= int(anchor_total.on_time_orders),
        "late_not_above_anchor": int(candidate_total.late_orders) <= int(anchor_total.late_orders),
        "severe_not_above_anchor": int(candidate_total.severe_late_count) <= int(anchor_total.severe_late_count),
        "max_lateness_not_above_anchor": float(candidate_total.max_lateness) <= float(anchor_total.max_lateness) + 1e-9,
        "average_lateness_not_above_anchor": candidate_avg <= anchor_avg + 1e-9,
        "energy_not_above_anchor": float(candidate_total.total_energy) <= float(anchor_total.total_energy) + 1e-9,
        "distance_not_above_anchor": float(candidate_total.total_distance) <= float(anchor_total.total_distance) + 1e-9,
    }
    allowed = bool(all(checks.values()))
    return allowed, {
        "allowed": allowed,
        "checks": checks,
        "candidate_violations": candidate_violations,
        "candidate_total": candidate_total.to_dict(),
        "anchor_total": anchor_total.to_dict(),
    }


def _try_composite_accept_and_serve(
    env: Any,
    first_obs: Dict[str, Any],
    action: Tuple[int, int],
    order_id: int,
) -> Optional[Tuple[Any, Dict[str, Any], List[Tuple[int, int]], List[Dict[str, Any]], bool]]:
    first = _apply_one(env, action)
    if first is None:
        return None
    e2, obs2, _, done, info1 = first
    if done:
        return e2, obs2, [action], [info1], True
    # Only collapse accept+serve when no new response decision blocks routing.
    if int(obs2.get("current_decision_request", -1)) > 0:
        return e2, obs2, [action], [info1], False
    service_action = (e2.K_NONE, int(order_id))
    if not _is_action_feasible(e2, service_action):
        return e2, obs2, [action], [info1], False
    second = _apply_one(e2, service_action)
    if second is None:
        return e2, obs2, [action], [info1], False
    e3, obs3, _, done2, info2 = second
    return e3, obs3, [action, service_action], [info1, info2], bool(done2)


def _simulate_child(
    state: JointBeamState,
    item: Dict[str, Any],
    cfg: JointAcceptRouteBeamConfig,
) -> Optional[JointBeamState]:
    action = (int(item["action"][0]), int(item["action"][1]))
    action_type = str(item.get("action_type", "unknown"))
    try:
        if action_type == "accept_and_serve":
            sim = _try_composite_accept_and_serve(state.env, state.obs, action, int(item.get("order_id", action[1])))
            if sim is None:
                return None
            env2, obs2, actions, infos, done = sim
        else:
            one = _apply_one(state.env, action)
            if one is None:
                return None
            env2, obs2, _, done, info = one
            actions = [action]
            infos = [info]
    except Exception as exc:
        bad = state.metrics.copy()
        bad.hard_constraint_violations += 1
        return JointBeamState(
            env=state.env.copy(),
            obs=state.obs,
            first_action=state.first_action or action,
            actions=list(state.actions) + [action],
            action_types=list(state.action_types) + [action_type],
            infos=list(state.infos) + [{"error": str(exc), "action_type": action_type}],
            metrics=bad,
            score=score_metrics(bad, cfg.objective),
            done=False,
        )
    delta = merge_transition_metrics(env2, infos, severe_lateness_threshold=float(cfg.objective.severe_lateness_threshold))
    metrics = add_metrics(state.metrics, delta)
    risk_score, risk_components = pending_order_risk_score(env2, cfg.objective)
    score = score_metrics(metrics, cfg.objective) + float(risk_score)
    first_action = state.first_action or actions[0]
    return JointBeamState(
        env=env2,
        obs=obs2,
        first_action=first_action,
        actions=list(state.actions) + actions,
        action_types=list(state.action_types) + [action_type],
        infos=list(state.infos) + infos,
        metrics=metrics,
        score=float(score),
        risk_components=risk_components,
        done=bool(done),
    )


def select_joint_accept_route_action(
    env: Any,
    obs: Dict[str, Any],
    cfg: JointAcceptRouteBeamConfig,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    if bool(cfg.enable_acceptance_guard) and int(obs.get("current_decision_request", -1)) <= 0:
        return _select_guarded_route_action(env, obs, cfg)

    start = time.time()
    root_candidates = _candidate_actions(env, obs, cfg)
    if not root_candidates:
        return (env.K_NONE, 0), {"mode": "joint_accept_route_beam", "fallback": True, "candidate_count": 0}

    beams: List[JointBeamState] = [JointBeamState(env=env.copy(), obs=obs)]
    expanded = 0
    timed_out = False
    depth_limit = max(1, int(cfg.lookahead_depth))
    for _depth in range(depth_limit):
        children: List[JointBeamState] = []
        for beam in beams:
            if beam.done:
                children.append(beam)
                continue
            if time.time() - start > float(cfg.time_limit_seconds):
                timed_out = True
                break
            candidate_items = root_candidates if _depth == 0 and not beam.actions else _candidate_actions(beam.env, beam.obs, cfg)
            for item in candidate_items:
                if expanded >= int(cfg.max_expanded_states):
                    timed_out = True
                    break
                child = _simulate_child(beam, item, cfg)
                expanded += 1
                if child is not None:
                    children.append(child)
            if timed_out:
                break
        if not children:
            break
        if bool(cfg.enable_dominance_pruning):
            children = dominance_prune(children)
        children.sort(key=lambda x: (x.score, x.metrics.late_orders, x.metrics.max_lateness, x.metrics.total_distance))
        beams = children[: max(1, int(cfg.beam_size))]
        if timed_out:
            break

    beams.sort(key=lambda x: (x.score, x.metrics.late_orders, x.metrics.max_lateness, x.metrics.total_distance))
    best = beams[0]
    action = best.first_action or root_candidates[0]["action"]
    return (int(action[0]), int(action[1])), {
        "mode": "joint_accept_route_beam",
        "candidate_count": len(root_candidates),
        "expanded_states": int(expanded),
        "timed_out": bool(timed_out),
        "lookahead_depth": int(cfg.lookahead_depth),
        "beam_size": int(cfg.beam_size),
        "selected_action": [int(action[0]), int(action[1])],
        "selected_sequence": [[int(a[0]), int(a[1])] for a in best.actions],
        "selected_action_types": list(best.action_types),
        "selected_score": float(best.score),
        "selected_state": best.debug_snapshot(),
        "objective": cfg.objective.to_dict(),
        "root_candidates": [
            {
                "action": [int(x["action"][0]), int(x["action"][1])],
                "action_type": str(x.get("action_type", "")),
                "order_id": int(x.get("order_id", 0)),
                "reject_reason": str(x.get("reject_reason", "")),
            }
            for x in root_candidates[:20]
        ],
    }


def rollout_joint_accept_route_beam(
    env: Any,
    cfg: JointAcceptRouteBeamConfig,
    *,
    max_steps: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    traj: List[Dict[str, Any]] = []
    debug_steps: List[Dict[str, Any]] = []
    total_cost = 0.0
    done = False
    expanded_total = 0
    for step in range(int(max_steps)):
        action, debug = select_joint_accept_route_action(e, obs, cfg)
        obs2, reward, done, info = e.step(action)
        total_cost += -float(reward)
        expanded_total += int(debug.get("expanded_states", 0))
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "joint_beam_debug": debug})
        if step < 30:
            debug_steps.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += float(cfg.objective.hard_violation_weight)
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -float(cfg.objective.hard_violation_weight), "info": {"timeout": True}})
    return float(total_cost), traj, {
        "done": bool(done),
        "steps": len(traj),
        "expanded_states": int(expanded_total),
        "debug_steps": debug_steps,
    }


def rollout_tail_risk_constrained_joint_beam(
    env: Any,
    cfg: JointAcceptRouteBeamConfig,
    budget: TailRiskBudget,
    *,
    max_steps: int,
    anchor_actions: Optional[Sequence[Any]] = None,
    allow_anchor_deviation: bool = False,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    traj: List[Dict[str, Any]] = []
    debug_steps: List[Dict[str, Any]] = []
    cumulative = JointBeamMetrics()
    total_cost = 0.0
    done = False
    anchor_overrides = 0
    for step in range(int(max_steps)):
        action, debug = select_tail_risk_constrained_joint_action(e, obs, cfg, budget, cumulative)
        anchor_action = None
        if anchor_actions is not None and step < len(anchor_actions):
            anchor_action = _parse_anchor_action(anchor_actions[step], e)
        if anchor_action is not None and _is_action_feasible(e, anchor_action) and tuple(action) != tuple(anchor_action):
            if bool(allow_anchor_deviation):
                candidate_delta = _one_step_metrics(e, action, float(budget.severe_lateness_threshold))
                anchor_delta = _one_step_metrics(e, anchor_action, float(budget.severe_lateness_threshold))
                allow_deviation, anchor_rule = _anchor_deviation_allowed(
                    cumulative=cumulative,
                    candidate_delta=candidate_delta,
                    anchor_delta=anchor_delta,
                    budget=budget,
                )
            else:
                allow_deviation = False
                anchor_rule = {
                    "allowed": False,
                    "reason": "baseline_anchor_locked_until_no_regret_deviation_is_enabled",
                }
            debug["anchor_reference"] = {
                "anchor_action": [int(anchor_action[0]), int(anchor_action[1])],
                "candidate_action": [int(action[0]), int(action[1])],
                "deviation_rule": anchor_rule,
            }
            if not allow_deviation:
                action = anchor_action
                anchor_overrides += 1
                debug["anchor_reference"]["override_to_anchor"] = True
            else:
                debug["anchor_reference"]["override_to_anchor"] = False
        obs2, reward, done, info = e.step(action)
        total_cost += -float(reward)
        delta = transition_metrics(e, info, severe_lateness_threshold=float(budget.severe_lateness_threshold))
        cumulative = add_metrics(cumulative, delta)
        debug = {
            **debug,
            "cumulative_metrics_after_step": cumulative.to_dict(),
            "step_delta_metrics": delta.to_dict(),
        }
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "tail_risk_debug": debug})
        if step < 30:
            debug_steps.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += float(cfg.objective.hard_violation_weight)
        cumulative.hard_constraint_violations += 1
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -float(cfg.objective.hard_violation_weight), "info": {"timeout": True}})
    return float(total_cost), traj, {
        "done": bool(done),
        "steps": len(traj),
        "budget": budget.to_dict(),
        "anchor_overrides": int(anchor_overrides),
        "final_cumulative_metrics": cumulative.to_dict(),
        "final_budget_violations": _tail_risk_violations(cumulative, budget),
        "debug_steps": debug_steps,
    }
