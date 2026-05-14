from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.evaluation.time_window_inference import TimeWindowInferenceConfig, predict_action_lateness


@dataclass
class V2SequenceScoreConfig:
    accept_reward: float = 1.0
    on_time_reward: float = 2.0
    reject_penalty: float = 7.0
    time_window_weight: float = 1.0
    lateness_weight: float = 1.2
    late_count_penalty: float = 4.0
    max_lateness_weight: float = 1.2
    severe_lateness_threshold: float = 10.0
    severe_lateness_weight: float = 4.0
    energy_weight: float = 0.04
    distance_weight: float = 0.02
    utilization_balance_weight: float = 0.0
    hard_constraint_penalty: float = 1_000_000.0
    future_tight_window_horizon: float = 8.0
    future_tight_window_weight: float = 0.25


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _served_lateness(info: Dict[str, Any]) -> List[float]:
    vals = []
    for value in (info.get("service_lateness", {}) or {}).values():
        vals.append(max(0.0, safe_float(value)))
    return vals


def action_distance_energy(env: Any, obs: Dict[str, Any], action: Tuple[int, int]) -> Tuple[float, float]:
    k, j = int(action[0]), int(action[1])
    i = int(obs.get("i", 0))
    dist = 0.0
    energy = 0.0
    try:
        dense = env.get_dense_edge_attr()
        dist += float(dense[i, j, 0]) if j >= 0 else 0.0
    except Exception:
        dist += float(env.dist_mat[i, j]) if j >= 0 else 0.0
    try:
        energy += float(env._truck_energy(i, j, float(obs.get("truck_load", 0.0)))) if j >= 0 else 0.0
    except Exception:
        pass
    if k != env.K_NONE:
        dist += float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
        try:
            energy += float(env._drone_energy(i, k, j))
        except Exception:
            pass
    return float(dist), float(energy)


def hard_violation_count(env: Any, info: Dict[str, Any]) -> int:
    violations = 0
    k = int(info.get("k", env.K_NONE))
    if k != env.K_NONE:
        energy_use = safe_float(info.get("energy_use"))
        soc_prev = safe_float(info.get("soc_prev"))
        reserve = safe_float(getattr(env.cfg, "soc_min_reserve", 0.0))
        if energy_use > max(0.0, soc_prev - reserve) + 1e-9:
            violations += 1
        if safe_float(info.get("drone_time")) > safe_float(getattr(env.cfg, "B", 0.0)) + 1e-9:
            violations += 1
    if safe_float(info.get("truck_load_next")) > safe_float(getattr(env.cfg, "truck_capacity", 0.0)) + 1e-9:
        violations += 1
    return int(violations)


def future_tight_window_pressure(env: Any, cfg: V2SequenceScoreConfig) -> float:
    """Estimate sequence-level risk left by the current state.

    This is intentionally a heuristic. The beam search still validates actions
    through the environment; this term only helps avoid moves that strand many
    tight-window orders for later.
    """
    state = getattr(env, "state", {})
    if not state:
        return 0.0
    t = safe_float(state.get("t"))
    i = int(state.get("i", 0))
    accepted = np.asarray(state.get("accepted", []))
    rejected = np.asarray(state.get("rejected", []))
    served = np.asarray(state.get("served", []))
    loaded = np.asarray(state.get("loaded", []))
    truck_pickup_load = safe_float(state.get("truck_pickup_load"))
    pressure = 0.0
    horizon = max(0.0, float(cfg.future_tight_window_horizon))
    for node in range(1, int(getattr(env, "N", 0)) + 1):
        if node >= accepted.size or accepted[node] <= 0 or served[node] > 0 or rejected[node] > 0:
            continue
        due = safe_float(env.due[node], math.inf)
        if not math.isfinite(due):
            continue
        release = safe_float(env.release[node])
        if release > t + horizon:
            continue
        try:
            travel = float(env._tau_truck(i, node, apply_traffic=False, bucket=env.get_time_bucket(t_elapsed=t)))
        except Exception:
            travel = float(env.dist_mat[i, node]) / max(1e-9, float(env.cfg.vT))
        service = float(getattr(env.cfg, "sT", 0.0))
        feasible_now = False
        try:
            feasible_now = bool(
                env._request_feasible_for_truck(
                    node=node,
                    accepted=accepted,
                    served=served,
                    loaded=loaded,
                    truck_pickup_load=truck_pickup_load,
                    t=t,
                )
            )
        except Exception:
            feasible_now = False
        earliest = max(t, release) + travel + service
        slack = due - earliest
        if slack < 0:
            pressure += -slack + 2.0
        elif slack <= horizon:
            pressure += (horizon - slack) / max(1e-9, horizon)
        if not feasible_now and due - t <= horizon:
            pressure += 0.5
    return float(pressure)


def transition_components(
    env_after: Any,
    obs_before: Dict[str, Any],
    action: Tuple[int, int],
    info: Dict[str, Any],
    cfg: V2SequenceScoreConfig,
) -> Dict[str, float]:
    late_values = _served_lateness(info)
    served_count = len(late_values)
    late_count = sum(1 for x in late_values if x > 1e-9)
    on_time_count = served_count - late_count
    distance = safe_float(info.get("road_distance"))
    k = int(info.get("k", action[0] if isinstance(action, tuple) else env_after.K_NONE))
    i = int(info.get("i", obs_before.get("i", 0)))
    j = int(info.get("j", action[1] if isinstance(action, tuple) else 0))
    if k != env_after.K_NONE:
        distance += float(env_after.dist_mat[i, k]) + float(env_after.dist_mat[k, j])
    energy = safe_float(info.get("truck_energy_use")) + safe_float(info.get("energy_use"))
    reject_count = 0
    accept_count = 0
    if str(info.get("phase", "")) == "decision":
        if info.get("decision") == "reject":
            reject_count += 1
        elif info.get("decision") == "accept":
            accept_count += 1
    reject_count += len(info.get("expired_nodes", []) or [])
    max_late = max([0.0] + late_values)
    total_late = sum(late_values)
    return {
        "accepted_decisions": float(accept_count),
        "rejected_orders": float(reject_count),
        "served_orders": float(served_count),
        "on_time_orders": float(on_time_count),
        "late_orders": float(late_count),
        "total_lateness": float(total_late),
        "max_lateness": float(max_late),
        "severe_lateness": float(max(0.0, max_late - float(cfg.severe_lateness_threshold))),
        "energy": float(energy),
        "distance": float(distance),
        "hard_violations": float(hard_violation_count(env_after, info)),
        "future_tight_window_pressure": float(future_tight_window_pressure(env_after, cfg)),
    }


def merge_components(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            if key == "max_lateness":
                out[key] = max(out.get(key, 0.0), safe_float(value))
            else:
                out[key] = out.get(key, 0.0) + safe_float(value)
    return out


def score_components(components: Dict[str, float], cfg: V2SequenceScoreConfig) -> float:
    score = 0.0
    score -= float(cfg.accept_reward) * safe_float(components.get("accepted_decisions"))
    score -= float(cfg.on_time_reward) * safe_float(components.get("on_time_orders"))
    score += float(cfg.reject_penalty) * safe_float(components.get("rejected_orders"))
    score += float(cfg.late_count_penalty) * safe_float(components.get("late_orders"))
    score += float(cfg.lateness_weight) * safe_float(components.get("total_lateness"))
    score += float(cfg.max_lateness_weight) * safe_float(components.get("max_lateness"))
    score += float(cfg.severe_lateness_weight) * safe_float(components.get("severe_lateness"))
    score += float(cfg.energy_weight) * safe_float(components.get("energy"))
    score += float(cfg.distance_weight) * safe_float(components.get("distance"))
    score += float(cfg.future_tight_window_weight) * float(cfg.time_window_weight) * safe_float(
        components.get("future_tight_window_pressure")
    )
    score += float(cfg.hard_constraint_penalty) * safe_float(components.get("hard_violations"))
    return float(score)


def sequence_score(components: Sequence[Dict[str, float]], cfg: V2SequenceScoreConfig) -> Tuple[float, Dict[str, float]]:
    merged = merge_components(components)
    return score_components(merged, cfg), merged


def estimate_action_priority(
    env: Any,
    obs: Dict[str, Any],
    action: Tuple[int, int],
    cfg: V2SequenceScoreConfig,
) -> Tuple[float, Dict[str, Any]]:
    tw_cfg = TimeWindowInferenceConfig(
        lateness_bias_weight=float(cfg.lateness_weight),
        severe_lateness_threshold=float(cfg.severe_lateness_threshold),
        severe_lateness_bias_weight=float(cfg.severe_lateness_weight),
    )
    k, j = int(action[0]), int(action[1])
    pred = predict_action_lateness(env, obs, j=j, k=k, cfg=tw_cfg)
    node_lateness = [max(0.0, safe_float(v)) for v in (pred.get("node_lateness", {}) or {}).values()]
    late_count = sum(1 for x in node_lateness if x > 1e-9)
    max_late = max([0.0] + node_lateness)
    total_late = sum(node_lateness)
    distance, energy = action_distance_energy(env, obs, action)
    service_count = len(node_lateness)
    current_request = int(obs.get("current_decision_request", -1))
    reject = 1 if current_request > 0 and j == 0 else 0
    accept = 1 if current_request > 0 and j != 0 else 0
    components = {
        "accepted_decisions": float(accept),
        "rejected_orders": float(reject),
        "served_orders": float(service_count),
        "on_time_orders": float(service_count - late_count),
        "late_orders": float(late_count),
        "total_lateness": float(total_late),
        "max_lateness": float(max_late),
        "severe_lateness": float(max(0.0, max_late - float(cfg.severe_lateness_threshold))),
        "energy": float(energy),
        "distance": float(distance),
        "hard_violations": 0.0,
        "future_tight_window_pressure": 0.0,
    }
    score = score_components(components, cfg)
    if service_count == 0 and current_request <= 0:
        score += 0.25
    return float(score), {"prediction": pred, "components": components}


def trajectory_score(traj: Sequence[Dict[str, Any]], cfg: V2SequenceScoreConfig) -> Tuple[float, Dict[str, float]]:
    components: List[Dict[str, float]] = []
    for item in traj:
        info = item.get("info", {}) or {}
        if item.get("action") == ("TIMEOUT",):
            components.append({"hard_violations": 1.0})
            continue
        late_values = _served_lateness(info)
        served_count = len(late_values)
        late_count = sum(1 for x in late_values if x > 1e-9)
        reject_count = (1 if info.get("decision") == "reject" else 0) + len(info.get("expired_nodes", []) or [])
        accept_count = 1 if info.get("decision") == "accept" else 0
        max_late = max([0.0] + late_values)
        distance = safe_float(info.get("road_distance"))
        k = int(info.get("k", -1))
        # Drone distance is not always recoverable without the environment, but
        # energy is logged. Repair acceptance uses this as a conservative proxy
        # and final reports are recomputed from the replayed environment.
        components.append(
            {
                "accepted_decisions": float(accept_count),
                "rejected_orders": float(reject_count),
                "served_orders": float(served_count),
                "on_time_orders": float(served_count - late_count),
                "late_orders": float(late_count),
                "total_lateness": float(sum(late_values)),
                "max_lateness": float(max_late),
                "severe_lateness": float(max(0.0, max_late - float(cfg.severe_lateness_threshold))),
                "energy": safe_float(info.get("truck_energy_use")) + safe_float(info.get("energy_use")),
                "distance": float(distance),
                "hard_violations": 0.0,
                "future_tight_window_pressure": 0.0,
                "drone_action_count": 1.0 if k != -1 else 0.0,
            }
        )
    return sequence_score(components, cfg)
