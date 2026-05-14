from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def _time_ref(env: Any) -> float:
    finite = np.asarray(env.due[np.isfinite(env.due)], dtype=np.float32)
    release_max = float(np.max(env.release)) if getattr(env, "release", np.asarray([])).size > 0 else 0.0
    due_max = float(np.max(finite)) if finite.size > 0 else release_max + 1.0
    return max(1e-6, due_max, release_max + 1.0)


def sequence_tw_pressure(
    env: Any,
    *,
    t: float,
    i: int,
    accepted: np.ndarray,
    served: np.ndarray,
    rejected: np.ndarray,
    loaded: np.ndarray,
    truck_pickup_load: float,
) -> Dict[str, float]:
    """Heuristic lookahead pressure used for reward shaping.

    It estimates whether currently accepted, unserved, released orders are
    running out of slack from the truck's current position. The environment
    still enforces hard constraints; this is only a dense learning signal.
    """
    t_den = _time_ref(env)
    tight_threshold = max(1.0, 0.25 * float(getattr(env.cfg, "B", 6.0)) + float(env.cfg.sT))
    slacks = []
    predicted_late_count = 0
    tight_count = 0
    remaining_count = 0
    pressure = 0.0
    max_pred_late = 0.0
    for node in range(1, env.N + 1):
        if accepted[node] <= 0 or served[node] > 0 or rejected[node] > 0:
            continue
        if float(env.release[node]) > float(t) + 1e-9:
            continue
        remaining_count += 1
        due = float(env.due[node])
        if not math.isfinite(due):
            continue
        try:
            travel = float(env._tau_truck(int(i), int(node), apply_traffic=False, bucket=env.get_time_bucket(t_elapsed=t)))
        except Exception:
            travel = float(env.dist_mat[int(i), int(node)]) / max(1e-9, float(env.cfg.vT))
        finish = float(t) + travel + float(env.cfg.sT)
        slack = due - finish
        slacks.append(float(slack))
        if slack <= tight_threshold:
            tight_count += 1
        if slack < 0:
            late = -float(slack)
            predicted_late_count += 1
            max_pred_late = max(max_pred_late, late)
            pressure += 1.0 + min(10.0, late / max(1e-6, t_den))
        else:
            pressure += max(0.0, (tight_threshold - slack) / max(1e-6, tight_threshold))
    if slacks:
        min_slack = min(slacks)
        avg_slack = float(sum(slacks) / len(slacks))
    else:
        min_slack = 2.0 * t_den
        avg_slack = 2.0 * t_den
    return {
        "pressure": float(pressure),
        "tight_count": float(tight_count),
        "predicted_late_count": float(predicted_late_count),
        "remaining_count": float(remaining_count),
        "min_slack": float(min_slack),
        "avg_slack": float(avg_slack),
        "max_predicted_lateness": float(max_pred_late),
    }


def hard_violation_proxy(env: Any, info: Dict[str, Any]) -> float:
    violations = 0.0
    k = int(info.get("k", env.K_NONE))
    if k != env.K_NONE:
        energy_use = float(info.get("energy_use", 0.0))
        soc_prev = float(info.get("soc_prev", 0.0))
        if energy_use > max(0.0, soc_prev - float(env.cfg.soc_min_reserve)) + 1e-9:
            violations += 1.0
        if float(info.get("drone_time", 0.0)) > float(env.cfg.B) + 1e-9:
            violations += 1.0
    if float(info.get("truck_load_next", 0.0)) > float(env.cfg.truck_capacity) + 1e-9:
        violations += 1.0
    return float(violations)


def sequence_tw_reward_components(
    env: Any,
    *,
    pre_pressure: Dict[str, float],
    post_pressure: Dict[str, float],
    dt: float,
    late_served_count: float,
    total_lateness: float,
    max_step_lateness: float,
    truck_distance: float,
    drone_distance: float,
    energy_use: float,
    info_for_hard: Dict[str, Any],
) -> Dict[str, float]:
    future_risk_delta = max(0.0, float(post_pressure["pressure"]) - float(pre_pressure["pressure"]))
    slack_loss = max(0.0, float(pre_pressure["min_slack"]) - float(post_pressure["min_slack"]))
    slack_gain = max(0.0, float(post_pressure["min_slack"]) - float(pre_pressure["min_slack"]))
    tight_delay = max(0.0, float(pre_pressure["tight_count"])) * max(0.0, float(dt))
    severe = max(0.0, float(max_step_lateness) - float(env.cfg.severe_lateness_threshold))
    hard = hard_violation_proxy(env, info_for_hard)
    cap = max(1e-6, float(getattr(env.cfg, "truck_capacity", 1.0)))
    truck_load_ratio = max(0.0, float(info_for_hard.get("truck_load_next", 0.0))) / cap
    remaining_pressure = float(post_pressure.get("remaining_count", 0.0)) / max(1.0, float(getattr(env, "N", 1)))
    workload_imbalance = abs(float(truck_load_ratio) - float(remaining_pressure))

    comps = {
        "late_order_cost": float(env.cfg.late_order_penalty) * float(late_served_count),
        "lateness_duration_cost": float(env.cfg.lateness_duration_penalty) * float(total_lateness),
        "severe_lateness_cost": float(env.cfg.severe_lateness_penalty) * severe,
        "max_lateness_cost": float(env.cfg.max_lateness_penalty) * float(max_step_lateness),
        "future_lateness_risk_cost": float(env.cfg.future_lateness_risk_penalty) * future_risk_delta,
        "tight_order_delay_cost": float(env.cfg.tight_order_delay_penalty) * tight_delay,
        "slack_preservation_reward": float(env.cfg.slack_preservation_reward) * slack_gain,
        "energy_cost": float(env.cfg.energy_cost_weight) * float(energy_use),
        "distance_cost": float(env.cfg.distance_cost_weight) * (float(truck_distance) + float(drone_distance)),
        "workload_imbalance_cost": float(env.cfg.workload_balance_weight) * workload_imbalance,
        "hard_constraint_cost": float(env.cfg.hard_constraint_violation_penalty) * hard,
        "future_risk_delta": float(future_risk_delta),
        "slack_loss": float(slack_loss),
        "slack_gain": float(slack_gain),
        "tight_order_delay": float(tight_delay),
        "workload_imbalance": float(workload_imbalance),
        "hard_constraint_violations": float(hard),
    }
    comps["sequence_lateness_cost"] = (
        comps["late_order_cost"]
        + comps["lateness_duration_cost"]
        + comps["max_lateness_cost"]
        + comps["future_lateness_risk_cost"]
        + comps["tight_order_delay_cost"]
    )
    comps["severe_future_lateness_cost"] = float(env.cfg.severe_lateness_penalty) * max(
        0.0,
        float(post_pressure.get("max_predicted_lateness", 0.0)) - float(env.cfg.severe_lateness_threshold),
    )
    comps["max_lateness_proxy_cost"] = float(env.cfg.max_lateness_penalty) * float(
        post_pressure.get("max_predicted_lateness", 0.0)
    )
    comps["sequence_tw_total_cost"] = (
        comps["late_order_cost"]
        + comps["lateness_duration_cost"]
        + comps["severe_lateness_cost"]
        + comps["max_lateness_cost"]
        + comps["future_lateness_risk_cost"]
        + comps["tight_order_delay_cost"]
        + comps["severe_future_lateness_cost"]
        + comps["max_lateness_proxy_cost"]
        + comps["energy_cost"]
        + comps["distance_cost"]
        + comps["workload_imbalance_cost"]
        + comps["hard_constraint_cost"]
        - comps["slack_preservation_reward"]
    )
    return comps
