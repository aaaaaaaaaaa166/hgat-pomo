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
    dominance_prune,
    merge_transition_metrics,
    pending_order_risk_score,
    score_metrics,
)


@dataclass
class JointAcceptRouteBeamConfig:
    beam_size: int = 16
    lookahead_depth: int = 3
    candidate_top_k: int = 10
    max_expanded_states: int = 5000
    time_limit_seconds: float = 2.0
    enable_dominance_pruning: bool = True
    objective: JointBeamObjectiveConfig = field(default_factory=JointBeamObjectiveConfig)


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
        if _is_action_feasible(env, (env.K_NONE, current)):
            feas = classify_order_feasibility(env, current)
            items.append(
                {
                    "action": (env.K_NONE, int(current)),
                    "action_type": "accept_only",
                    "order_id": int(current),
                    "feasibility": feas.to_dict(),
                }
            )
            items.append(
                {
                    "action": (env.K_NONE, int(current)),
                    "action_type": "accept_and_serve",
                    "order_id": int(current),
                    "feasibility": feas.to_dict(),
                    "composite_serve_order": int(current),
                }
            )
        return _dedupe_actions(items)
    return _candidate_route_actions(env, obs, cfg)


def _apply_one(env: Any, action: Tuple[int, int]) -> Optional[Tuple[Any, Dict[str, Any], float, bool, Dict[str, Any]]]:
    if not _is_action_feasible(env, action):
        return None
    e2 = env.copy()
    obs2, reward, done, info = e2.step(action)
    return e2, obs2, float(reward), bool(done), info


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
    delta = merge_transition_metrics(env2, infos)
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
            for item in _candidate_actions(beam.env, beam.obs, cfg):
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
