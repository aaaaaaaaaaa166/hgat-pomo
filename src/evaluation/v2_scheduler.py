from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.v2_sequence_scoring import (
    V2SequenceScoreConfig,
    estimate_action_priority,
    score_components,
    sequence_score,
    transition_components,
)


@dataclass
class V2SchedulerConfig:
    lookahead_depth: int = 1
    beam_size: int = 1
    candidate_top_k: int = 8
    max_runtime_seconds: float = 0.20
    deterministic_seed: int = 0
    use_beam_search: bool = True
    score: V2SequenceScoreConfig = field(default_factory=V2SequenceScoreConfig)


def _is_action_feasible(env: Any, action: Tuple[int, int]) -> bool:
    k, j = int(action[0]), int(action[1])
    try:
        masks = env.get_masks()
        if not (0 <= j <= env.N) or int(masks["truck_mask"][j]) == 0:
            return False
        if k == env.K_NONE:
            return True
        dm = env.get_masks(j=j)["drone_mask"]
        return bool(1 <= k <= env.N and int(dm[k]) > 0)
    except Exception:
        return False


def generate_candidate_actions(
    env: Any,
    obs: Dict[str, Any],
    cfg: V2SchedulerConfig,
) -> List[Dict[str, Any]]:
    masks = env.get_masks()
    feasible_j = np.where(masks["truck_mask"] > 0)[0].astype(int).tolist()
    current_request = int(obs.get("current_decision_request", -1))
    raw_actions: List[Tuple[int, int]] = []

    if current_request > 0:
        if current_request in feasible_j:
            raw_actions.append((env.K_NONE, int(current_request)))
        if 0 in feasible_j:
            raw_actions.append((env.K_NONE, 0))
    else:
        j_ranked: List[Tuple[float, int]] = []
        for j in feasible_j:
            due = float(env.due[j]) if j > 0 else float("inf")
            if not np.isfinite(due):
                due = 1e6
            dist = float(env.dist_mat[int(obs.get("i", 0)), int(j)]) if j >= 0 else 0.0
            depot_bias = 3.0 if j == 0 and any(x > 0 for x in feasible_j if x != 0) else 0.0
            j_ranked.append((due + 0.1 * dist + depot_bias, int(j)))
        j_ranked.sort(key=lambda x: x[0])
        j_limit = max(1, min(len(j_ranked), int(cfg.candidate_top_k) * 2))
        for _, j in j_ranked[:j_limit]:
            raw_actions.append((env.K_NONE, int(j)))
            try:
                dm = env.get_masks(j=j)["drone_mask"]
            except Exception:
                dm = np.zeros((env.N + 1,), dtype=np.int8)
            feasible_k = np.where(dm > 0)[0].astype(int).tolist()
            k_ranked: List[Tuple[float, int]] = []
            for k in feasible_k:
                due = float(env.due[k]) if np.isfinite(float(env.due[k])) else 1e6
                dist = float(env.dist_mat[int(obs.get("i", 0)), int(k)]) + float(env.dist_mat[int(k), int(j)])
                k_ranked.append((due + 0.05 * dist, int(k)))
            k_ranked.sort(key=lambda x: x[0])
            for _, k in k_ranked[: max(1, min(4, int(cfg.candidate_top_k)))]:
                raw_actions.append((int(k), int(j)))

    seen = set()
    ranked: List[Dict[str, Any]] = []
    for action in raw_actions:
        action = (int(action[0]), int(action[1]))
        if action in seen or not _is_action_feasible(env, action):
            continue
        seen.add(action)
        priority, detail = estimate_action_priority(env, obs, action, cfg.score)
        ranked.append({"action": action, "priority": float(priority), "detail": detail})
    ranked.sort(key=lambda x: x["priority"])
    if not ranked and feasible_j:
        fallback = (env.K_NONE, int(feasible_j[0]))
        ranked.append({"action": fallback, "priority": 0.0, "detail": {"fallback": True}})
    return ranked[: max(1, int(cfg.candidate_top_k))]


def _simulate_child(
    env: Any,
    obs: Dict[str, Any],
    action: Tuple[int, int],
    cfg: V2SchedulerConfig,
) -> Optional[Dict[str, Any]]:
    if not _is_action_feasible(env, action):
        return None
    e = env.copy()
    try:
        obs2, reward, done, info = e.step(action)
    except Exception as exc:
        return {
            "env": env.copy(),
            "obs": obs,
            "done": False,
            "action": action,
            "reward": -float(cfg.score.hard_constraint_penalty),
            "info": {"error": str(exc), "hard_violation": True},
            "components": {"hard_violations": 1.0},
            "score_delta": float(cfg.score.hard_constraint_penalty),
        }
    comps = transition_components(e, obs, action, info, cfg.score)
    return {
        "env": e,
        "obs": obs2,
        "done": bool(done),
        "action": action,
        "reward": float(reward),
        "info": info,
        "components": comps,
        "score_delta": score_components(comps, cfg.score),
    }


def select_v2_action(
    env: Any,
    obs: Dict[str, Any],
    cfg: V2SchedulerConfig,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    start = time.time()
    candidates = generate_candidate_actions(env, obs, cfg)
    if not cfg.use_beam_search or int(cfg.lookahead_depth) <= 1 or int(cfg.beam_size) <= 1:
        best = candidates[0]
        return best["action"], {
            "mode": "greedy",
            "candidate_count": len(candidates),
            "selected_action": [int(best["action"][0]), int(best["action"][1])],
            "selected_priority": float(best["priority"]),
            "candidates": _debug_candidates(candidates),
        }

    beams: List[Dict[str, Any]] = [
        {
            "env": env.copy(),
            "obs": obs,
            "first_action": None,
            "actions": [],
            "components": [],
            "score": 0.0,
            "done": False,
        }
    ]
    expanded = 0
    timed_out = False
    depth_limit = max(1, int(cfg.lookahead_depth))
    for depth in range(depth_limit):
        children: List[Dict[str, Any]] = []
        for beam in beams:
            if beam["done"]:
                children.append(beam)
                continue
            if time.time() - start > float(cfg.max_runtime_seconds):
                timed_out = True
                break
            child_candidates = generate_candidate_actions(beam["env"], beam["obs"], cfg)
            for item in child_candidates:
                sim = _simulate_child(beam["env"], beam["obs"], item["action"], cfg)
                if sim is None:
                    continue
                expanded += 1
                first = item["action"] if beam["first_action"] is None else beam["first_action"]
                comps = list(beam["components"]) + [sim["components"]]
                total_score, merged = sequence_score(comps, cfg.score)
                children.append(
                    {
                        "env": sim["env"],
                        "obs": sim["obs"],
                        "first_action": first,
                        "actions": list(beam["actions"]) + [item["action"]],
                        "components": comps,
                        "merged_components": merged,
                        "score": float(total_score),
                        "done": bool(sim["done"]),
                    }
                )
        if timed_out:
            break
        if not children:
            break
        children.sort(key=lambda x: x["score"])
        beams = children[: max(1, int(cfg.beam_size))]

    beams.sort(key=lambda x: x["score"])
    best_beam = beams[0]
    first_action = best_beam["first_action"] or candidates[0]["action"]
    return (int(first_action[0]), int(first_action[1])), {
        "mode": "beam",
        "candidate_count": len(candidates),
        "expanded_nodes": int(expanded),
        "timed_out": bool(timed_out),
        "lookahead_depth": int(cfg.lookahead_depth),
        "beam_size": int(cfg.beam_size),
        "selected_action": [int(first_action[0]), int(first_action[1])],
        "selected_sequence": [[int(a[0]), int(a[1])] for a in best_beam.get("actions", [])],
        "selected_score": float(best_beam.get("score", 0.0)),
        "selected_components": best_beam.get("merged_components", {}),
        "root_candidates": _debug_candidates(candidates),
    }


def _debug_candidates(candidates: Sequence[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    out = []
    for item in list(candidates)[:limit]:
        action = item["action"]
        detail = item.get("detail", {})
        prediction = detail.get("prediction", {}) if isinstance(detail, dict) else {}
        out.append(
            {
                "action": [int(action[0]), int(action[1])],
                "priority": float(item.get("priority", 0.0)),
                "predicted_lateness": float(prediction.get("predicted_lateness", 0.0) or 0.0),
                "will_be_late": bool(prediction.get("will_be_late", False)),
                "feasibility_reason": prediction.get("feasibility_reason", ""),
            }
        )
    return out


def rollout_v2_scheduler(
    env: Any,
    cfg: V2SchedulerConfig,
    *,
    max_steps: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    np.random.seed(int(cfg.deterministic_seed))
    e = env.copy()
    obs = e.reset()
    traj: List[Dict[str, Any]] = []
    changed_steps: List[Dict[str, Any]] = []
    total_cost = 0.0
    done = False
    for step in range(max_steps):
        action, debug = select_v2_action(e, obs, cfg)
        obs2, reward, done, info = e.step(action)
        reward_f = float(reward)
        total_cost += -reward_f
        traj.append(
            {
                "obs": obs,
                "obs2": obs2,
                "action": action,
                "reward": reward_f,
                "info": info,
                "v2_debug": debug,
            }
        )
        if step < 20 or bool(debug.get("timed_out")):
            changed_steps.append(
                {
                    "step": int(step),
                    "t": float(obs.get("t", 0.0)),
                    "i": int(obs.get("i", 0)),
                    "action": [int(action[0]), int(action[1])],
                    "debug": debug,
                }
            )
        obs = obs2
        if done:
            break
    if not done:
        total_cost += float(cfg.score.hard_constraint_penalty)
        traj.append(
            {
                "obs": obs,
                "action": ("TIMEOUT",),
                "reward": -float(cfg.score.hard_constraint_penalty),
                "info": {"timeout": True, "max_steps": max_steps},
            }
        )
    return float(total_cost), traj, {"changed_steps": changed_steps, "steps": len(traj), "done": bool(done)}
