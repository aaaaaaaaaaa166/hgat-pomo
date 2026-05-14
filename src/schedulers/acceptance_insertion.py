from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from src.schedulers.feasibility import classify_order_feasibility
from src.schedulers.insertion_objective import InsertionObjectiveConfig, simulate_action_score


@dataclass
class AcceptanceInsertionConfig:
    method: str = "hybrid_score_insertion"
    candidate_top_k: int = 12
    objective: InsertionObjectiveConfig = field(default_factory=InsertionObjectiveConfig)


def _candidate_actions(env: Any, obs: Dict[str, Any], cfg: AcceptanceInsertionConfig) -> List[Tuple[int, int]]:
    masks = env.get_masks()
    feasible_j = np.where(np.asarray(masks["truck_mask"]) > 0)[0].astype(int).tolist()
    current = int(obs.get("current_decision_request", -1))
    if current > 0:
        out = []
        if current in feasible_j:
            out.append((env.K_NONE, current))
        if 0 in feasible_j:
            out.append((env.K_NONE, 0))
        return out

    i = int(obs.get("i", 0))

    def key_j(j: int) -> Tuple[float, float, int]:
        due = float(env.due[j]) if j > 0 and np.isfinite(float(env.due[j])) else 1e9
        release = float(env.release[j]) if j > 0 else 0.0
        dist = float(env.dist_mat[i, j]) if j >= 0 else 0.0
        return (due, release + 0.05 * dist, int(j))

    raw: List[Tuple[int, int]] = []
    for j in sorted(feasible_j, key=key_j)[: max(1, int(cfg.candidate_top_k) * 2)]:
        raw.append((env.K_NONE, int(j)))
        try:
            dm = np.asarray(env.get_masks(j=j)["drone_mask"])
        except Exception:
            dm = np.zeros((env.N + 1,), dtype=np.int8)
        feasible_k = np.where(dm > 0)[0].astype(int).tolist()

        def key_k(k: int) -> Tuple[float, float, int]:
            due = float(env.due[k]) if np.isfinite(float(env.due[k])) else 1e9
            dist = float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
            return (due, dist, int(k))

        for k in sorted(feasible_k, key=key_k)[:4]:
            raw.append((int(k), int(j)))
    seen = set()
    out = []
    for action in raw:
        if action not in seen:
            seen.add(action)
            out.append(action)
    return out[: max(1, int(cfg.candidate_top_k))]


def _method_adjusted_score(method: str, score: float, detail: Dict[str, Any]) -> float:
    method = str(method)
    nodes = detail.get("nodes", []) or []
    min_due = min([float(n.get("estimated_arrival_time", 1e9)) + float(n.get("slack_after_arrival", 1e9)) for n in nodes] + [1e9])
    max_future = max([float(n.get("future_impact_score", 0.0)) for n in nodes] + [0.0])
    max_late = float(detail.get("max_lateness", 0.0))
    if method == "edd_insertion":
        return float(score) + 0.05 * min_due
    if method == "regret_insertion":
        return float(score) + 2.0 * max_future - 0.5 * len(nodes)
    if method == "min_lateness_insertion":
        return float(score) + 4.0 * max_late
    return float(score)


def select_acceptance_insertion_action(env: Any, obs: Dict[str, Any], cfg: AcceptanceInsertionConfig) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    current = int(obs.get("current_decision_request", -1))
    if current > 0:
        accept_action = (env.K_NONE, current)
        reject_action = (env.K_NONE, 0)
        accept_score, accept_detail = simulate_action_score(env, obs, accept_action, cfg.objective)
        reject_score, reject_detail = simulate_action_score(env, obs, reject_action, cfg.objective)
        if accept_score <= reject_score:
            return accept_action, {
                "phase": "acceptance",
                "selected": "accept",
                "accept_score": float(accept_score),
                "reject_score": float(reject_score),
                "feasibility": classify_order_feasibility(env, current).to_dict(),
                "accept_detail": accept_detail,
                "reject_detail": reject_detail,
            }
        return reject_action, {
            "phase": "acceptance",
            "selected": "reject",
            "accept_score": float(accept_score),
            "reject_score": float(reject_score),
            "feasibility": classify_order_feasibility(env, current).to_dict(),
            "accept_detail": accept_detail,
            "reject_detail": reject_detail,
        }

    candidates = _candidate_actions(env, obs, cfg)
    ranked = []
    for action in candidates:
        score, detail = simulate_action_score(env, obs, action, cfg.objective)
        adjusted = _method_adjusted_score(cfg.method, score, detail)
        ranked.append({"action": action, "score": float(adjusted), "raw_score": float(score), "detail": detail})
    ranked.sort(key=lambda x: x["score"])
    if not ranked:
        return (env.K_NONE, 0), {"phase": "route", "selected": "fallback_reject", "candidate_count": 0}
    best = ranked[0]
    return best["action"], {
        "phase": "route",
        "method": str(cfg.method),
        "selected_action": [int(best["action"][0]), int(best["action"][1])],
        "selected_score": float(best["score"]),
        "candidate_count": len(ranked),
        "candidates": [
            {
                "action": [int(x["action"][0]), int(x["action"][1])],
                "score": float(x["score"]),
                "raw_score": float(x["raw_score"]),
            }
            for x in ranked[:20]
        ],
    }


def rollout_acceptance_insertion(env: Any, cfg: AcceptanceInsertionConfig, *, max_steps: int) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    traj: List[Dict[str, Any]] = []
    total_cost = 0.0
    debug_steps: List[Dict[str, Any]] = []
    done = False
    for step in range(int(max_steps)):
        action, debug = select_acceptance_insertion_action(e, obs, cfg)
        obs2, reward, done, info = e.step(action)
        total_cost += -float(reward)
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "insertion_debug": debug})
        if step < 30:
            debug_steps.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += float(cfg.objective.hard_constraint_penalty)
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -float(cfg.objective.hard_constraint_penalty), "info": {"timeout": True}})
    return float(total_cost), traj, {"done": bool(done), "steps": len(traj), "debug_steps": debug_steps}
