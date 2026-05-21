from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.evaluation.time_window_inference import TimeWindowInferenceConfig, predict_action_lateness
from src.training.sequence_time_window_reward import sequence_tw_pressure


def _hard_reason(env: Any, obs: Dict[str, Any], action: Tuple[int, int]) -> str:
    k, j = int(action[0]), int(action[1])
    try:
        masks = env.get_masks()
        if not (0 <= j <= env.N) or int(masks["truck_mask"][j]) == 0:
            return "truck_action_masked"
        if k != env.K_NONE:
            dm = env.get_masks(j=j)["drone_mask"]
            if not (1 <= k <= env.N) or int(dm[k]) == 0:
                return "drone_action_masked"
    except Exception as exc:
        return f"mask_error:{exc}"

    if k != env.K_NONE:
        try:
            energy = float(env._drone_energy(int(obs.get("i", 0)), k, j))
            soc = float(obs.get("soc", 0.0))
            if energy > max(0.0, soc - float(env.cfg.soc_min_reserve)) + 1e-9:
                return "battery_insufficient"
            drone_time = float(env._tau_drone(int(obs.get("i", 0)), k, j)) + float(env.cfg.sD)
            if drone_time > float(env.cfg.B) + 1e-9:
                return "range_exceeded"
            if float(env.demand[k]) > float(env.cfg.QD) + 1e-9:
                return "drone_capacity_exceeded"
        except Exception as exc:
            return f"drone_check_error:{exc}"
    return ""


def classify_action_feasibility(
    env: Any,
    obs: Dict[str, Any],
    action: Tuple[int, int],
    *,
    tight_slack_threshold: float | None = None,
) -> Dict[str, Any]:
    """Classify a candidate action without turning soft time windows into masks."""
    k, j = int(action[0]), int(action[1])
    hard_reason = _hard_reason(env, obs, (k, j))
    tw_cfg = TimeWindowInferenceConfig(
        lateness_bias_weight=1.0,
        severe_lateness_threshold=10.0,
        severe_lateness_bias_weight=2.0,
    )
    pred = predict_action_lateness(env, obs, j=j, k=k, cfg=tw_cfg)
    late_score = float(pred.get("lateness_risk_score", 0.0) or 0.0)
    t = float(obs.get("t", 0.0))
    tight_threshold = (
        float(tight_slack_threshold)
        if tight_slack_threshold is not None
        else max(1.0, 0.25 * float(getattr(env.cfg, "B", 6.0)) + float(env.cfg.sT))
    )

    urgent = False
    for node in (j, k):
        if node <= 0 or node == env.K_NONE:
            continue
        due = float(env.due[node])
        if np.isfinite(due) and due - t <= tight_threshold:
            urgent = True

    future_impact = 0.0
    try:
        pre = sequence_tw_pressure(
            env,
            t=t,
            i=int(obs.get("i", 0)),
            accepted=env.state["accepted"],
            served=env.state["served"],
            rejected=env.state["rejected"],
            loaded=env.state["loaded"],
            truck_pickup_load=float(env.state["truck_pickup_load"]),
        )
        e2 = env.copy()
        obs2, _, _, _ = e2.step((k, j))
        post = sequence_tw_pressure(
            e2,
            t=float(obs2.get("t", t)),
            i=int(obs2.get("i", j)),
            accepted=e2.state["accepted"],
            served=e2.state["served"],
            rejected=e2.state["rejected"],
            loaded=e2.state["loaded"],
            truck_pickup_load=float(e2.state["truck_pickup_load"]),
        )
        future_impact = max(0.0, float(post.get("pressure", 0.0)) - float(pre.get("pressure", 0.0)))
    except Exception:
        future_impact = 0.0

    if hard_reason:
        cls = "hard_infeasible"
    elif urgent and late_score <= 1e-9:
        cls = "feasible_urgent"
    elif late_score > 1e-9 or future_impact > 1e-9:
        cls = "soft_time_window_risky"
    else:
        cls = "feasible_normal"

    return {
        "action": [int(k), int(j)],
        "class": cls,
        "hard_infeasible": bool(hard_reason),
        "soft_time_window_risky": bool(cls == "soft_time_window_risky"),
        "feasible_urgent": bool(cls == "feasible_urgent"),
        "feasible_normal": bool(cls == "feasible_normal"),
        "reject_reason": hard_reason,
        "lateness_risk_score": float(late_score),
        "future_impact_score": float(future_impact),
        "prediction": pred,
    }
