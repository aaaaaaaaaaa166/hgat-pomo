from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

from src.evaluation.time_window_inference import TimeWindowInferenceConfig, predict_action_lateness
from src.training.sequence_time_window_reward import sequence_tw_pressure


@dataclass
class OrderFeasibility:
    order_id: int
    hard_infeasible: bool
    feasible_on_time: bool
    feasible_but_late: bool
    risky_due_to_future_orders: bool
    reject_reason: str
    estimated_arrival_time: float
    predicted_lateness: float
    slack_after_arrival: float
    lateness_risk_score: float
    future_impact_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _time_ref(env: Any) -> float:
    finite = np.asarray(env.due[np.isfinite(env.due)], dtype=np.float32)
    release_max = float(np.max(env.release)) if getattr(env, "release", np.asarray([])).size > 0 else 0.0
    due_max = float(np.max(finite)) if finite.size > 0 else release_max + 1.0
    return max(1e-6, due_max, release_max + 1.0)


def _future_pressure(env: Any, obs: Dict[str, Any]) -> float:
    try:
        p = sequence_tw_pressure(
            env,
            t=float(obs.get("t", 0.0)),
            i=int(obs.get("i", 0)),
            accepted=env.state["accepted"],
            served=env.state["served"],
            rejected=env.state["rejected"],
            loaded=env.state["loaded"],
            truck_pickup_load=float(env.state["truck_pickup_load"]),
        )
        return float(p.get("pressure", 0.0))
    except Exception:
        return 0.0


def classify_order_feasibility(env: Any, order_id: int) -> OrderFeasibility:
    """Classify an order for accept/reject and routing decisions.

    Hard-infeasible reasons are intended for masks. Time-window lateness is a
    risk signal and remains serviceable unless the order is already unavailable.
    """
    obs = env.get_obs()
    node = int(order_id)
    t = float(obs.get("t", 0.0))
    reasons = []
    if not (1 <= node <= int(env.N)):
        reasons.append("invalid_order_id")
    else:
        served = np.asarray(obs.get("served", []))
        accepted = np.asarray(obs.get("accepted", []))
        rejected = np.asarray(obs.get("rejected", []))
        expired = np.asarray(obs.get("expired", np.zeros_like(served)))
        if served[node] > 0:
            reasons.append("already_served")
        if rejected[node] > 0:
            reasons.append("already_rejected")
        if expired[node] > 0:
            reasons.append("already_expired")
        if float(env.release[node]) > t + 1e-9:
            reasons.append("not_released")
        if int(env.is_dynamic[node]) > 0 and accepted[node] <= 0 and t > float(env.decision_deadline[node]) + 1e-9:
            reasons.append("response_window_expired")

    action = (env.K_NONE, node)
    if node > 0 and not reasons:
        masks = env.get_masks()
        if node >= len(masks["truck_mask"]) or int(masks["truck_mask"][node]) == 0:
            # If the order is a pending dynamic request, accepting it is still
            # represented as (K_NONE, node). Otherwise this is a service mask.
            current = int(obs.get("current_decision_request", -1))
            if current != node:
                reasons.append("service_action_masked")
    pred = predict_action_lateness(
        env,
        obs,
        j=node if node > 0 else 0,
        k=env.K_NONE,
        cfg=TimeWindowInferenceConfig(lateness_bias_weight=1.0, severe_lateness_bias_weight=2.0),
    )
    eta = float(pred.get("estimated_arrival_time", t))
    due = float(env.due[node]) if node > 0 else float("inf")
    slack = due - eta if math.isfinite(due) else 2.0 * _time_ref(env)
    late = max(0.0, float(pred.get("predicted_lateness", 0.0) or 0.0))
    future_impact = 0.0
    if node > 0 and not reasons:
        try:
            pre = _future_pressure(env, obs)
            e2 = env.copy()
            obs2, _, _, _ = e2.step(action)
            post = _future_pressure(e2, obs2)
            future_impact = max(0.0, post - pre)
        except Exception:
            future_impact = 0.0
    hard = bool(reasons)
    return OrderFeasibility(
        order_id=node,
        hard_infeasible=hard,
        feasible_on_time=bool(not hard and late <= 1e-9),
        feasible_but_late=bool(not hard and late > 1e-9),
        risky_due_to_future_orders=bool(not hard and future_impact > 1e-9),
        reject_reason=";".join(reasons),
        estimated_arrival_time=float(eta),
        predicted_lateness=float(late),
        slack_after_arrival=float(slack),
        lateness_risk_score=float(pred.get("lateness_risk_score", late) or 0.0),
        future_impact_score=float(future_impact),
    )
