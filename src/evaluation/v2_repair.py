from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.evaluation.v2_scheduler import V2SchedulerConfig, _is_action_feasible, select_v2_action
from src.evaluation.v2_sequence_scoring import V2SequenceScoreConfig, trajectory_score


@dataclass
class V2RepairConfig:
    enable_v2_repair: bool = False
    repair_max_iterations: int = 50
    repair_window_size: int = 4
    repair_cross_drone_swap: bool = False
    repair_allow_reject_late_orders: bool = False
    repair_max_acceptance_drop: float = 0.005


def _is_route_item(item: Dict[str, Any]) -> bool:
    action = item.get("action")
    info = item.get("info", {}) or {}
    return isinstance(action, tuple) and len(action) == 2 and str(info.get("phase", "")) != "decision"


def _extract_route_actions(traj: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    return [(int(item["action"][0]), int(item["action"][1])) for item in traj if _is_route_item(item)]


def _extract_decisions(traj: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for item in traj:
        info = item.get("info", {}) or {}
        if str(info.get("phase", "")) == "decision":
            node = int(info.get("decision_node", -1))
            if node > 0:
                out[node] = 0 if info.get("decision") == "reject" else node
    return out


def _action_due(env: Any, action: Tuple[int, int]) -> float:
    k, j = int(action[0]), int(action[1])
    vals = []
    for node in (j, k):
        if node > 0:
            try:
                vals.append(float(env.due[node]))
            except Exception:
                pass
    vals = [v for v in vals if np.isfinite(v)]
    return min(vals) if vals else 1e9


def _severe_route_indices(traj: Sequence[Dict[str, Any]], threshold: float) -> List[int]:
    route_idx = -1
    severe: List[int] = []
    for item in traj:
        if not _is_route_item(item):
            continue
        route_idx += 1
        late = [float(v) for v in (item.get("info", {}).get("service_lateness", {}) or {}).values()]
        if late and max(late) >= float(threshold):
            severe.append(route_idx)
    return severe


def _candidate_sequences(
    env: Any,
    route_actions: Sequence[Tuple[int, int]],
    traj: Sequence[Dict[str, Any]],
    repair_cfg: V2RepairConfig,
    score_cfg: V2SequenceScoreConfig,
) -> Iterable[Tuple[str, List[Tuple[int, int]]]]:
    base = list(route_actions)
    yielded = 0
    max_iter = max(1, int(repair_cfg.repair_max_iterations))
    window = max(2, int(repair_cfg.repair_window_size))

    for idx in range(max(0, len(base) - 1)):
        cand = list(base)
        cand[idx], cand[idx + 1] = cand[idx + 1], cand[idx]
        yielded += 1
        yield f"adjacent_swap_{idx}_{idx + 1}", cand
        if yielded >= max_iter:
            return

    for start in range(0, max(0, len(base) - window + 1)):
        cand = list(base)
        sub = cand[start : start + window]
        sub.sort(key=lambda action: _action_due(env, action))
        cand[start : start + window] = sub
        if cand != base:
            yielded += 1
            yield f"window_due_sort_{start}_{start + window - 1}", cand
            if yielded >= max_iter:
                return

    for idx in _severe_route_indices(traj, threshold=score_cfg.severe_lateness_threshold):
        for shift in range(1, window + 1):
            new_idx = max(0, idx - shift)
            if new_idx == idx:
                continue
            cand = list(base)
            action = cand.pop(idx)
            cand.insert(new_idx, action)
            yielded += 1
            yield f"move_severe_{idx}_to_{new_idx}", cand
            if yielded >= max_iter:
                return


def _decision_action(env: Any, obs: Dict[str, Any], decisions: Dict[int, int]) -> Tuple[int, int]:
    current = int(obs.get("current_decision_request", -1))
    masks = env.get_masks()
    requested_j = int(decisions.get(current, current))
    if requested_j >= 0 and requested_j <= env.N and int(masks["truck_mask"][requested_j]) > 0:
        return (env.K_NONE, requested_j)
    if current > 0 and int(masks["truck_mask"][current]) > 0:
        return (env.K_NONE, current)
    return (env.K_NONE, 0)


def replay_route_sequence(
    initial_env: Any,
    route_actions: Sequence[Tuple[int, int]],
    decisions: Dict[int, int],
    scheduler_cfg: V2SchedulerConfig,
    *,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    env = initial_env.copy()
    obs = env.reset()
    queue = list(route_actions)
    traj: List[Dict[str, Any]] = []
    skipped_actions: List[List[int]] = []
    fallback_count = 0
    done = False
    for step in range(max_steps):
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            action = _decision_action(env, obs, decisions)
            debug = {"source": "preserved_decision"}
        else:
            action = None
            for idx, candidate in enumerate(list(queue)):
                candidate = (int(candidate[0]), int(candidate[1]))
                if _is_action_feasible(env, candidate):
                    action = candidate
                    queue.pop(idx)
                    break
                skipped_actions.append([int(candidate[0]), int(candidate[1])])
            if action is None:
                action, debug = select_v2_action(env, obs, scheduler_cfg)
                fallback_count += 1
            else:
                debug = {"source": "repaired_route_sequence"}
        obs2, reward, done, info = env.step(action)
        traj.append(
            {
                "obs": obs,
                "obs2": obs2,
                "action": action,
                "reward": float(reward),
                "info": info,
                "v2_repair_debug": debug,
            }
        )
        obs = obs2
        if done:
            break
    if not done:
        traj.append(
            {
                "obs": obs,
                "action": ("TIMEOUT",),
                "reward": -float(scheduler_cfg.score.hard_constraint_penalty),
                "info": {"timeout": True, "max_steps": max_steps},
            }
        )
    return traj, {
        "remaining_route_actions": int(len(queue)),
        "skipped_action_count": int(len(skipped_actions)),
        "skipped_actions_sample": skipped_actions[:20],
        "fallback_count": int(fallback_count),
        "done": bool(done),
    }


def repair_trajectory(
    initial_env: Any,
    original_traj: Sequence[Dict[str, Any]],
    scheduler_cfg: V2SchedulerConfig,
    repair_cfg: V2RepairConfig,
    *,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not repair_cfg.enable_v2_repair:
        return list(original_traj), {"repair_enabled": False, "accepted_moves": []}

    route_actions = _extract_route_actions(original_traj)
    decisions = _extract_decisions(original_traj)
    best_traj = list(original_traj)
    best_score, best_components = trajectory_score(best_traj, scheduler_cfg.score)
    log: Dict[str, Any] = {
        "repair_enabled": True,
        "cross_drone_swap_requested": bool(repair_cfg.repair_cross_drone_swap),
        "cross_drone_swap_note": "single_drone_environment_noop",
        "reject_late_orders_requested": bool(repair_cfg.repair_allow_reject_late_orders),
        "reject_late_orders_note": "preserved_decisions_for_feasible_replay",
        "initial_score": float(best_score),
        "initial_components": best_components,
        "tried_moves": [],
        "accepted_moves": [],
    }
    for name, candidate_route in _candidate_sequences(initial_env, route_actions, original_traj, repair_cfg, scheduler_cfg.score):
        cand_traj, replay_log = replay_route_sequence(
            initial_env,
            candidate_route,
            decisions,
            scheduler_cfg,
            max_steps=max_steps,
        )
        cand_score, cand_components = trajectory_score(cand_traj, scheduler_cfg.score)
        hard_ok = cand_components.get("hard_violations", 0.0) <= best_components.get("hard_violations", 0.0) + 1e-9
        late_better = cand_components.get("late_orders", 0.0) < best_components.get("late_orders", 0.0)
        score_better = cand_score < best_score - 1e-9
        accepted = bool(hard_ok and (late_better or score_better))
        move = {
            "move": name,
            "score": float(cand_score),
            "score_delta": float(cand_score - best_score),
            "late_orders": float(cand_components.get("late_orders", 0.0)),
            "max_lateness": float(cand_components.get("max_lateness", 0.0)),
            "accepted": accepted,
            "replay": replay_log,
        }
        log["tried_moves"].append(move)
        if accepted:
            best_traj = cand_traj
            best_score = cand_score
            best_components = cand_components
            log["accepted_moves"].append(move)
    log["final_score"] = float(best_score)
    log["final_components"] = best_components
    return best_traj, log
