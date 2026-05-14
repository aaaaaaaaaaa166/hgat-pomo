from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch


ORDER_EXTRA_DIM = 8
TRUCK_EXTRA_DIM = 7
DRONE_EXTRA_DIM = 2


def _time_ref(env: Any) -> float:
    finite = np.asarray(env.due[np.isfinite(env.due)], dtype=np.float32)
    release_max = float(np.max(env.release)) if getattr(env, "release", np.asarray([])).size > 0 else 0.0
    due_max = float(np.max(finite)) if finite.size > 0 else release_max + 1.0
    return max(1e-6, due_max, release_max + 1.0)


def _truck_finish_estimate(env: Any, obs: Dict[str, Any], node: int) -> float:
    i = int(obs.get("i", 0))
    t = float(obs.get("t", 0.0))
    if node <= 0:
        return t
    try:
        travel = float(env._tau_truck(i, int(node), apply_traffic=False, bucket=str(obs.get("time_bucket", env.get_time_bucket()))))
    except Exception:
        travel = float(env.dist_mat[i, int(node)]) / max(1e-9, float(env.cfg.vT))
    return float(t + travel + float(env.cfg.sT))


def compute_order_sequence_tw_features(env: Any, obs: Dict[str, Any]) -> torch.Tensor:
    """Order-level optional sequence time-window features.

    Columns:
    estimated_arrival_norm, slack_norm, predicted_lateness_norm,
    is_tight_window, urgency, service_time_norm, remaining_deadline_norm,
    will_be_late.
    """
    n = int(env.N)
    m = n + 1
    t = float(obs.get("t", 0.0))
    known = np.asarray(obs.get("known", np.ones(m)), dtype=np.float32)
    accepted = np.asarray(obs.get("accepted", np.zeros(m)), dtype=np.float32)
    served = np.asarray(obs.get("served", np.zeros(m)), dtype=np.float32)
    rejected = np.asarray(obs.get("rejected", np.zeros(m)), dtype=np.float32)
    t_den = _time_ref(env)
    out = np.zeros((m, ORDER_EXTRA_DIM), dtype=np.float32)
    tight_threshold = max(1.0, 0.25 * float(getattr(env.cfg, "B", 6.0)) + float(env.cfg.sT))
    for node in range(1, m):
        if known[node] <= 0.5:
            continue
        due = float(env.due[node])
        finite_due = math.isfinite(due)
        finish = _truck_finish_estimate(env, obs, node)
        slack = (due - finish) if finite_due else 2.0 * t_den
        late = max(0.0, finish - due) if finite_due else 0.0
        remaining = (due - t) if finite_due else 2.0 * t_den
        active = accepted[node] > 0.5 and served[node] <= 0.5 and rejected[node] <= 0.5
        tight = bool(active and finite_due and slack <= tight_threshold)
        urgency = 0.0
        if active and finite_due:
            urgency = 1.0 / (1.0 + max(0.0, slack))
            if late > 0:
                urgency += min(2.0, late / max(1e-6, t_den))
        out[node] = np.asarray(
            [
                np.clip(finish / t_den, 0.0, 3.0),
                np.clip(slack / t_den, -2.0, 2.0),
                np.clip(late / t_den, 0.0, 3.0),
                1.0 if tight else 0.0,
                np.clip(urgency, 0.0, 3.0),
                np.clip(float(env.cfg.sT) / t_den, 0.0, 1.0),
                np.clip(remaining / t_den, -2.0, 2.0),
                1.0 if late > 1e-9 else 0.0,
            ],
            dtype=np.float32,
        )
    return torch.from_numpy(out)


def compute_global_sequence_tw_stats(env: Any, obs: Dict[str, Any]) -> Dict[str, float]:
    t_den = _time_ref(env)
    known = np.asarray(obs.get("known", np.ones(env.N + 1)), dtype=np.float32)
    accepted = np.asarray(obs.get("accepted", np.zeros(env.N + 1)), dtype=np.float32)
    served = np.asarray(obs.get("served", np.zeros(env.N + 1)), dtype=np.float32)
    rejected = np.asarray(obs.get("rejected", np.zeros(env.N + 1)), dtype=np.float32)
    slacks = []
    late_count = 0
    tight_count = 0
    risk = 0.0
    workload_imbalance = 0.0
    tight_threshold = max(1.0, 0.25 * float(getattr(env.cfg, "B", 6.0)) + float(env.cfg.sT))
    remaining = 0
    for node in range(1, env.N + 1):
        if known[node] <= 0.5 or accepted[node] <= 0.5 or served[node] > 0.5 or rejected[node] > 0.5:
            continue
        remaining += 1
        due = float(env.due[node])
        if not math.isfinite(due):
            continue
        finish = _truck_finish_estimate(env, obs, node)
        slack = due - finish
        slacks.append(slack)
        if slack <= tight_threshold:
            tight_count += 1
        if slack < 0:
            late_count += 1
            risk += min(5.0, -slack / t_den + 1.0)
        else:
            risk += max(0.0, (tight_threshold - slack) / max(1e-6, tight_threshold))
    truck_load_ratio = float(obs.get("truck_load_ratio", 0.0))
    workload_imbalance = abs(truck_load_ratio - 0.5)
    if slacks:
        min_slack = min(slacks)
        avg_slack = float(sum(slacks) / len(slacks))
    else:
        min_slack = 2.0 * t_den
        avg_slack = 2.0 * t_den
    return {
        "remaining_orders_count": float(remaining),
        "remaining_tight_orders_count": float(tight_count),
        "minimum_slack_among_remaining_orders": float(min_slack),
        "average_slack_among_remaining_orders": float(avg_slack),
        "number_of_orders_predicted_late_if_delayed": float(late_count),
        "current_global_lateness_risk": float(risk),
        "workload_balance_score": float(max(0.0, 1.0 - workload_imbalance)),
        "time_ref": float(t_den),
    }


def compute_truck_sequence_tw_features(env: Any, obs: Dict[str, Any]) -> torch.Tensor:
    stats = compute_global_sequence_tw_stats(env, obs)
    t_den = max(1e-6, float(stats["time_ref"]))
    n = max(1.0, float(env.N))
    out = torch.tensor(
        [
            stats["remaining_orders_count"] / n,
            stats["remaining_tight_orders_count"] / n,
            np.clip(stats["minimum_slack_among_remaining_orders"] / t_den, -2.0, 2.0),
            np.clip(stats["average_slack_among_remaining_orders"] / t_den, -2.0, 2.0),
            np.clip(stats["number_of_orders_predicted_late_if_delayed"] / n, 0.0, 3.0),
            np.clip(stats["current_global_lateness_risk"] / n, 0.0, 3.0),
            np.clip(stats["workload_balance_score"], 0.0, 1.0),
        ],
        dtype=torch.float32,
    )
    return out.view(1, -1)


def compute_drone_sequence_tw_features(env: Any, obs: Dict[str, Any]) -> torch.Tensor:
    t_den = _time_ref(env)
    t = float(obs.get("t", 0.0))
    soc = float(obs.get("soc", 0.0))
    return torch.tensor([[np.clip(t / t_den, 0.0, 3.0), np.clip(soc, 0.0, 1.0)]], dtype=torch.float32)
