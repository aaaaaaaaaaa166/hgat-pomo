from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.env.instance_gen import REQUEST_DELIVERY, REQUEST_PICKUP


OVERALL_FIELDS = [
    "model_name",
    "total_orders",
    "accepted_orders",
    "rejected_orders",
    "acceptance_rate",
    "on_time_orders",
    "late_orders",
    "on_time_rate",
    "late_rate",
    "average_lateness",
    "max_lateness",
    "total_lateness",
    "total_delivery_time",
    "average_delivery_time",
    "average_waiting_time",
    "total_flight_distance",
    "average_flight_distance_per_order",
    "total_energy_consumption",
    "average_energy_per_order",
    "average_drone_utilization",
    "total_constraint_violations",
]

ORDER_DETAIL_FIELDS = [
    "model_name",
    "order_id",
    "accepted",
    "rejection_reason",
    "drone_id",
    "order_time",
    "pickup_time",
    "planned_delivery_time",
    "actual_delivery_time",
    "on_time",
    "late",
    "lateness_duration",
    "delivery_duration",
    "waiting_duration",
    "flight_distance",
    "energy_consumption",
    "time_window_violation",
    "battery_violation",
    "capacity_violation",
    "range_violation",
    "other_violation",
]

DRONE_DETAIL_FIELDS = [
    "model_name",
    "drone_id",
    "assigned_orders",
    "completed_orders",
    "total_flight_distance",
    "total_flight_time",
    "total_energy_consumption",
    "remaining_battery",
    "average_energy_per_order",
    "working_time",
    "idle_time",
    "utilization_rate",
    "battery_violation",
    "constraint_violation_count",
]

COMPARISON_FIELDS = [
    "baseline_model",
    "retrained_model",
    "acceptance_rate_change",
    "on_time_rate_change",
    "late_orders_change",
    "average_lateness_change",
    "max_lateness_change",
    "total_energy_consumption_change",
    "average_energy_per_order_change",
    "total_flight_distance_change",
    "average_drone_utilization_change",
    "total_constraint_violations_change",
    "new_model_overall_better",
]


def safe_div(num: float, den: float) -> float:
    den = float(den)
    return 0.0 if abs(den) <= 1e-12 else float(num) / den


def finite_or_blank(value: float) -> Any:
    value = float(value)
    if not math.isfinite(value):
        return ""
    return value


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _num(value: Any, default: float = 0.0) -> float:
    if value in {"", None}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _final_obs_from_traj(traj: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in reversed(traj):
        if isinstance(item.get("obs2"), dict):
            return item["obs2"]
        if isinstance(item.get("obs"), dict):
            return item["obs"]
    return None


def _new_served_nodes(item: Dict[str, Any]) -> List[int]:
    info = item.get("info", {})
    served_nodes = info.get("served_nodes", [])
    if served_nodes:
        return [int(x) for x in served_nodes]

    obs = item.get("obs", {})
    obs2 = item.get("obs2", {})
    prev = np.asarray(obs.get("served", []), dtype=np.float32)
    nxt = np.asarray(obs2.get("served", []), dtype=np.float32)
    if prev.size == 0 or nxt.size == 0:
        return []
    return [int(x) for x in np.where((nxt > 0.5) & (prev <= 0.5))[0] if int(x) > 0]


def _diagnose_rejection(env: Any, node: int, makespan: float) -> str:
    reasons: List[str] = []
    node = int(node)
    demand = float(env.demand[node])
    if int(env.is_dynamic[node]) > 0 and makespan > float(env.decision_deadline[node]) + 1e-9:
        reasons.append("decision_deadline_expired")
    if demand > float(env.cfg.truck_capacity) + 1e-9:
        reasons.append("capacity_limit")
    if int(env.drone_eligible[node]) == 0:
        reasons.append("drone_not_eligible")
    if demand > float(env.cfg.QD) + 1e-9:
        reasons.append("drone_capacity_limit")

    depot_drone_time = (float(env.dist_mat[0, node]) * 2.0) / max(1e-9, float(env.cfg.vD)) + float(env.cfg.sD)
    if depot_drone_time > float(env.cfg.B) + 1e-9:
        reasons.append("max_range_limit")
    depot_drone_energy = float(env._drone_energy(0, node, 0))
    if depot_drone_energy > max(0.0, float(env.cfg.soc_init) - float(env.cfg.soc_min_reserve)) + 1e-9:
        reasons.append("battery_limit")

    due_t = float(env.due[node])
    if math.isfinite(due_t):
        truck_direct = float(env._tau_truck(0, node, apply_traffic=False)) + float(env.cfg.sT)
        drone_direct = depot_drone_time
        earliest = float(env.release[node]) + min(truck_direct, drone_direct)
        if earliest > due_t + 1e-9:
            reasons.append("time_window_unreachable")

    return ";".join(reasons) if reasons else "unknown"


def analyze_episode(
    env: Any,
    traj: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    instance_id: int,
    objective_cost: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Convert one executed trajectory into order, drone and aggregate metrics."""
    final_obs = _final_obs_from_traj(traj)
    if final_obs is None:
        final_obs = env.reset()

    final_accepted = np.asarray(final_obs.get("accepted", np.zeros(env.N + 1)), dtype=np.float32)
    final_rejected = np.asarray(final_obs.get("rejected", np.zeros(env.N + 1)), dtype=np.float32)
    makespan = float(final_obs.get("t", 0.0))

    records: Dict[int, Dict[str, Any]] = {}
    for node in range(1, env.N + 1):
        due = float(env.due[node])
        req_type = int(env.request_type[node])
        records[node] = {
            "model_name": model_name,
            "instance_id": int(instance_id),
            "order_id": f"inst_{int(instance_id):04d}_order_{node:03d}",
            "node_id": int(node),
            "request_type": "pickup" if req_type == REQUEST_PICKUP else "delivery",
            "is_dynamic": bool(int(env.is_dynamic[node]) > 0),
            "accepted": bool(int(env.is_dynamic[node]) == 0),
            "rejection_reason": "",
            "drone_id": "",
            "order_time": float(env.release[node]),
            "pickup_time": "",
            "planned_delivery_time": finite_or_blank(due),
            "actual_delivery_time": "",
            "on_time": False,
            "late": False,
            "lateness_duration": 0.0,
            "slack_duration": "",
            "delivery_duration": "",
            "waiting_duration": "",
            "flight_distance": 0.0,
            "energy_consumption": 0.0,
            "service_mode": "",
            "time_window_violation": False,
            "battery_violation": False,
            "capacity_violation": False,
            "range_violation": False,
            "other_violation": False,
        }

    drone_distance = 0.0
    drone_time = 0.0
    drone_energy = 0.0
    truck_distance = 0.0
    truck_energy = 0.0
    drone_assigned = 0
    drone_completed = 0
    battery_violation_count = 0
    capacity_violation_count = 0
    range_violation_count = 0
    availability_conflicts = 0
    other_conflicts = 0

    for item in traj:
        info = item.get("info", {})
        phase = str(info.get("phase", ""))
        if item.get("action") == ("TIMEOUT",):
            other_conflicts += 1
            continue

        for node in info.get("expired_nodes", []) or []:
            node = int(node)
            if node in records and not records[node]["rejection_reason"]:
                records[node]["rejection_reason"] = "decision_deadline_expired"

        if phase == "decision":
            node = int(info.get("decision_node", -1))
            if node in records:
                if info.get("decision") == "accept":
                    records[node]["accepted"] = True
                    records[node]["rejection_reason"] = ""
                elif info.get("decision") == "reject":
                    records[node]["accepted"] = False
                    records[node]["rejection_reason"] = "policy_reject"
            continue

        i = int(info.get("i", item.get("obs", {}).get("i", 0)))
        j = int(info.get("j", 0))
        k = int(info.get("k", env.K_NONE))
        truck_distance += float(info.get("road_distance", 0.0))
        truck_energy += float(info.get("truck_energy_use", 0.0))

        if k != env.K_NONE:
            drone_assigned += 1
            step_drone_dist = float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
            step_drone_time = float(info.get("drone_time", 0.0))
            step_drone_energy = float(info.get("energy_use", 0.0))
            drone_distance += step_drone_dist
            drone_time += step_drone_time
            drone_energy += step_drone_energy
            if step_drone_energy > max(0.0, float(info.get("soc_prev", 0.0)) - float(env.cfg.soc_min_reserve)) + 1e-9:
                battery_violation_count += 1
            if step_drone_time > float(env.cfg.B) + 1e-9:
                range_violation_count += 1
        else:
            drone_energy += float(info.get("drone_idle_energy_use", 0.0))

        if max(float(info.get("truck_load_prev", 0.0)), float(info.get("truck_load_next", 0.0))) > float(env.cfg.truck_capacity) + 1e-9:
            capacity_violation_count += 1

        finish_times = info.get("service_finish_times", {}) or {}
        late_by_node = info.get("service_lateness", {}) or {}
        for node in _new_served_nodes(item):
            if node not in records:
                continue
            rec = records[node]
            rec["accepted"] = True
            rec["rejection_reason"] = ""
            finish_t = _num(finish_times.get(str(node)), _num(item.get("obs", {}).get("t"), 0.0) + _num(info.get("dt"), 0.0))
            late = max(0.0, _num(late_by_node.get(str(node)), 0.0))
            rec["actual_delivery_time"] = finish_t
            rec["lateness_duration"] = late
            rec["time_window_violation"] = late > 1e-9
            rec["late"] = late > 1e-9
            rec["on_time"] = not rec["late"]
            due_t = float(env.due[node])
            rec["slack_duration"] = "" if not math.isfinite(due_t) else max(0.0, due_t - finish_t)
            rec["delivery_duration"] = max(0.0, finish_t - float(env.release[node]))

            if int(env.request_type[node]) == REQUEST_PICKUP:
                rec["pickup_time"] = finish_t
                rec["waiting_duration"] = max(0.0, finish_t - float(env.release[node]))
            else:
                rec["pickup_time"] = 0.0 if float(env.release[node]) <= 1e-9 else float(env.release[node])
                rec["waiting_duration"] = max(0.0, float(rec["pickup_time"]) - float(env.release[node]))

            if node == k:
                drone_completed += 1
                rec["drone_id"] = "drone_0"
                rec["service_mode"] = "drone"
                rec["flight_distance"] = float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
                rec["energy_consumption"] = float(info.get("energy_use", 0.0))
                rec["battery_violation"] = (
                    float(info.get("energy_use", 0.0))
                    > max(0.0, float(info.get("soc_prev", 0.0)) - float(env.cfg.soc_min_reserve)) + 1e-9
                )
                rec["range_violation"] = float(info.get("drone_time", 0.0)) > float(env.cfg.B) + 1e-9
            elif node == j:
                rec["service_mode"] = "truck"
                rec["flight_distance"] = float(info.get("road_distance", 0.0))
                rec["energy_consumption"] = float(info.get("truck_energy_use", 0.0))
            else:
                rec["service_mode"] = "unknown"

            rec["capacity_violation"] = max(
                float(info.get("truck_load_prev", 0.0)),
                float(info.get("truck_load_next", 0.0)),
            ) > float(env.cfg.truck_capacity) + 1e-9

    for node, rec in records.items():
        accepted = bool(node < len(final_accepted) and final_accepted[node] > 0.5 and not (node < len(final_rejected) and final_rejected[node] > 0.5))
        rejected = bool(node < len(final_rejected) and final_rejected[node] > 0.5)
        rec["accepted"] = accepted
        if rejected:
            rec["accepted"] = False
            if not rec["rejection_reason"]:
                rec["rejection_reason"] = _diagnose_rejection(env, node, makespan)
        elif accepted and rec["actual_delivery_time"] == "":
            rec["late"] = True
            rec["on_time"] = False
            rec["other_violation"] = True
            rec["rejection_reason"] = "accepted_but_unserved_timeout"
            due_t = float(env.due[node])
            rec["lateness_duration"] = 0.0 if not math.isfinite(due_t) else max(0.0, makespan - due_t)

    order_rows = [records[node] for node in range(1, env.N + 1)]
    accepted_rows = [r for r in order_rows if _bool(r["accepted"])]
    late_rows = [r for r in accepted_rows if _bool(r["late"])]
    on_time_rows = [r for r in accepted_rows if _bool(r["on_time"])]
    completed_rows = [r for r in accepted_rows if r["actual_delivery_time"] != ""]

    total_delivery_time = sum(_num(r["delivery_duration"]) for r in completed_rows)
    total_waiting_time = sum(_num(r["waiting_duration"]) for r in completed_rows)
    total_lateness = sum(_num(r["lateness_duration"]) for r in late_rows)
    max_lateness = max([0.0] + [_num(r["lateness_duration"]) for r in late_rows])
    total_flight_distance = truck_distance + drone_distance
    total_energy = truck_energy + drone_energy
    time_window_violations = sum(1 for r in order_rows if _bool(r["time_window_violation"]))
    battery_violations = sum(1 for r in order_rows if _bool(r["battery_violation"])) + battery_violation_count
    capacity_violations = sum(1 for r in order_rows if _bool(r["capacity_violation"])) + capacity_violation_count
    range_violations = sum(1 for r in order_rows if _bool(r["range_violation"])) + range_violation_count
    other_violations = sum(1 for r in order_rows if _bool(r["other_violation"])) + other_conflicts + availability_conflicts
    total_constraint_violations = (
        time_window_violations
        + battery_violations
        + capacity_violations
        + range_violations
        + other_violations
    )
    utilization = safe_div(drone_time, makespan)

    episode_summary = {
        "model_name": model_name,
        "instance_id": int(instance_id),
        "objective_cost": float(objective_cost),
        "total_orders": int(env.N),
        "accepted_orders": int(len(accepted_rows)),
        "rejected_orders": int(env.N - len(accepted_rows)),
        "acceptance_rate": safe_div(len(accepted_rows), env.N),
        "on_time_orders": int(len(on_time_rows)),
        "late_orders": int(len(late_rows)),
        "on_time_rate": safe_div(len(on_time_rows), len(accepted_rows)),
        "late_rate": safe_div(len(late_rows), len(accepted_rows)),
        "average_lateness": safe_div(total_lateness, len(late_rows)),
        "max_lateness": float(max_lateness),
        "total_lateness": float(total_lateness),
        "total_delivery_time": float(total_delivery_time),
        "average_delivery_time": safe_div(total_delivery_time, len(completed_rows)),
        "average_waiting_time": safe_div(total_waiting_time, len(completed_rows)),
        "total_flight_distance": float(total_flight_distance),
        "average_flight_distance_per_order": safe_div(total_flight_distance, len(accepted_rows)),
        "total_energy_consumption": float(total_energy),
        "average_energy_per_order": safe_div(total_energy, len(accepted_rows)),
        "average_drone_utilization": float(utilization),
        "total_constraint_violations": int(total_constraint_violations),
        "time_window_violations": int(time_window_violations),
        "battery_violations": int(battery_violations),
        "capacity_violations": int(capacity_violations),
        "range_violations": int(range_violations),
        "availability_conflicts": int(availability_conflicts),
        "other_conflicts": int(other_violations),
    }

    drone_row = {
        "model_name": model_name,
        "instance_id": int(instance_id),
        "drone_id": "drone_0",
        "assigned_orders": int(drone_assigned),
        "completed_orders": int(drone_completed),
        "total_flight_distance": float(drone_distance),
        "total_flight_time": float(drone_time),
        "total_energy_consumption": float(drone_energy),
        "remaining_battery": float(final_obs.get("soc", 0.0)),
        "average_energy_per_order": safe_div(drone_energy, drone_completed),
        "working_time": float(drone_time),
        "idle_time": max(0.0, float(makespan) - float(drone_time)),
        "utilization_rate": float(utilization),
        "battery_violation": bool(battery_violation_count > 0),
        "constraint_violation_count": int(battery_violation_count + range_violation_count),
    }
    return episode_summary, order_rows, drone_row


def aggregate_model(
    model_name: str,
    episode_summaries: Sequence[Dict[str, Any]],
    order_rows: Sequence[Dict[str, Any]],
    drone_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_orders = sum(int(s["total_orders"]) for s in episode_summaries)
    accepted_orders = sum(int(s["accepted_orders"]) for s in episode_summaries)
    rejected_orders = sum(int(s["rejected_orders"]) for s in episode_summaries)
    on_time_orders = sum(int(s["on_time_orders"]) for s in episode_summaries)
    late_orders = sum(int(s["late_orders"]) for s in episode_summaries)
    total_lateness = sum(float(s["total_lateness"]) for s in episode_summaries)
    max_lateness = max([0.0] + [float(s["max_lateness"]) for s in episode_summaries])
    total_delivery_time = sum(float(s["total_delivery_time"]) for s in episode_summaries)
    completed_orders = sum(1 for r in order_rows if _bool(r.get("accepted")) and r.get("actual_delivery_time") != "")
    total_waiting_time = sum(_num(r.get("waiting_duration")) for r in order_rows if _bool(r.get("accepted")) and r.get("actual_delivery_time") != "")
    total_flight_distance = sum(float(s["total_flight_distance"]) for s in episode_summaries)
    total_energy = sum(float(s["total_energy_consumption"]) for s in episode_summaries)
    total_constraint_violations = sum(int(s["total_constraint_violations"]) for s in episode_summaries)
    avg_drone_util = safe_div(sum(float(s["average_drone_utilization"]) for s in episode_summaries), len(episode_summaries))

    overall = {
        "model_name": model_name,
        "total_orders": int(total_orders),
        "accepted_orders": int(accepted_orders),
        "rejected_orders": int(rejected_orders),
        "acceptance_rate": safe_div(accepted_orders, total_orders),
        "on_time_orders": int(on_time_orders),
        "late_orders": int(late_orders),
        "on_time_rate": safe_div(on_time_orders, accepted_orders),
        "late_rate": safe_div(late_orders, accepted_orders),
        "average_lateness": safe_div(total_lateness, late_orders),
        "max_lateness": float(max_lateness),
        "total_lateness": float(total_lateness),
        "total_delivery_time": float(total_delivery_time),
        "average_delivery_time": safe_div(total_delivery_time, completed_orders),
        "average_waiting_time": safe_div(total_waiting_time, completed_orders),
        "total_flight_distance": float(total_flight_distance),
        "average_flight_distance_per_order": safe_div(total_flight_distance, accepted_orders),
        "total_energy_consumption": float(total_energy),
        "average_energy_per_order": safe_div(total_energy, accepted_orders),
        "average_drone_utilization": float(avg_drone_util),
        "total_constraint_violations": int(total_constraint_violations),
    }

    assigned = sum(int(r["assigned_orders"]) for r in drone_rows)
    completed = sum(int(r["completed_orders"]) for r in drone_rows)
    drone_energy = sum(float(r["total_energy_consumption"]) for r in drone_rows)
    working = sum(float(r["working_time"]) for r in drone_rows)
    idle = sum(float(r["idle_time"]) for r in drone_rows)
    remaining = safe_div(sum(float(r["remaining_battery"]) for r in drone_rows), len(drone_rows))
    drone_detail = {
        "model_name": model_name,
        "drone_id": "drone_0",
        "assigned_orders": int(assigned),
        "completed_orders": int(completed),
        "total_flight_distance": float(sum(float(r["total_flight_distance"]) for r in drone_rows)),
        "total_flight_time": float(sum(float(r["total_flight_time"]) for r in drone_rows)),
        "total_energy_consumption": float(drone_energy),
        "remaining_battery": float(remaining),
        "average_energy_per_order": safe_div(drone_energy, completed),
        "working_time": float(working),
        "idle_time": float(idle),
        "utilization_rate": safe_div(working, working + idle),
        "battery_violation": any(_bool(r["battery_violation"]) for r in drone_rows),
        "constraint_violation_count": int(sum(int(r["constraint_violation_count"]) for r in drone_rows)),
    }
    return overall, drone_detail


def compare_overall(baseline: Dict[str, Any], retrained: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "baseline_model": baseline["model_name"],
        "retrained_model": retrained["model_name"],
        "acceptance_rate_change": float(retrained["acceptance_rate"]) - float(baseline["acceptance_rate"]),
        "on_time_rate_change": float(retrained["on_time_rate"]) - float(baseline["on_time_rate"]),
        "late_orders_change": int(retrained["late_orders"]) - int(baseline["late_orders"]),
        "average_lateness_change": float(retrained["average_lateness"]) - float(baseline["average_lateness"]),
        "max_lateness_change": float(retrained["max_lateness"]) - float(baseline["max_lateness"]),
        "total_energy_consumption_change": float(retrained["total_energy_consumption"]) - float(baseline["total_energy_consumption"]),
        "average_energy_per_order_change": float(retrained["average_energy_per_order"]) - float(baseline["average_energy_per_order"]),
        "total_flight_distance_change": float(retrained["total_flight_distance"]) - float(baseline["total_flight_distance"]),
        "average_drone_utilization_change": float(retrained["average_drone_utilization"]) - float(baseline["average_drone_utilization"]),
        "total_constraint_violations_change": int(retrained["total_constraint_violations"]) - int(baseline["total_constraint_violations"]),
    }
    row["new_model_overall_better"] = bool(
        row["acceptance_rate_change"] >= -1e-9
        and row["on_time_rate_change"] >= -1e-9
        and row["late_orders_change"] <= 0
        and row["average_lateness_change"] <= 1e-9
        and row["total_constraint_violations_change"] <= 0
    )
    return row


def _fieldnames(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> List[str]:
    names = list(preferred)
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(key)
    return names


def write_csv(path: str, rows: Sequence[Dict[str, Any]], preferred_fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = _fieldnames(rows, preferred_fields)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def write_markdown_report(
    path: str,
    overall_rows: Sequence[Dict[str, Any]],
    comparison: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metrics = [
        "acceptance_rate",
        "on_time_rate",
        "late_orders",
        "average_lateness",
        "max_lateness",
        "total_energy_consumption",
        "average_energy_per_order",
        "total_flight_distance",
        "average_drone_utilization",
        "total_constraint_violations",
    ]
    by_name = {row["model_name"]: row for row in overall_rows}
    baseline = by_name.get(comparison["baseline_model"], {})
    retrained = by_name.get(comparison["retrained_model"], {})
    lines = [
        "# Baseline vs Retrained Main Model Report",
        "",
        f"- Baseline model: `{comparison['baseline_model']}`",
        f"- Retrained model: `{comparison['retrained_model']}`",
        f"- Overall better: `{comparison['new_model_overall_better']}`",
        f"- Train data: `{metadata.get('train_split_file', '')}`",
        f"- Eval data: `{metadata.get('eval_split_file', '')}`",
        "",
        "| Metric | Baseline | Retrained | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric in metrics:
        base_val = float(baseline.get(metric, 0.0))
        new_val = float(retrained.get(metric, 0.0))
        change = new_val - base_val
        lines.append(f"| {metric} | {base_val:.6f} | {new_val:.6f} | {change:.6f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `time_window_violation` is counted when an accepted order is delivered after its due time; the current environment treats due time as a soft constraint with lateness penalty.",
            "- The current environment has one drone resource (`drone_0`); drone details are aggregated over all evaluated instances.",
            "- Rejection reasons are explicit policy rejects or decision-deadline expirations when observed; otherwise they are diagnostic labels inferred from the final instance state.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_report_bundle(
    output_dir: str,
    *,
    overall_rows: Sequence[Dict[str, Any]],
    order_rows: Sequence[Dict[str, Any]],
    drone_rows: Sequence[Dict[str, Any]],
    comparison: Dict[str, Any],
    episode_summaries: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "overall_summary_csv": os.path.join(output_dir, "overall_summary.csv"),
        "overall_summary_json": os.path.join(output_dir, "overall_summary.json"),
        "order_details_csv": os.path.join(output_dir, "order_details.csv"),
        "order_details_json": os.path.join(output_dir, "order_details.json"),
        "drone_details_csv": os.path.join(output_dir, "drone_details.csv"),
        "drone_details_json": os.path.join(output_dir, "drone_details.json"),
        "comparison_report_csv": os.path.join(output_dir, "comparison_report.csv"),
        "comparison_report_json": os.path.join(output_dir, "comparison_report.json"),
        "comparison_report_md": os.path.join(output_dir, "comparison_report.md"),
        "episode_summaries_json": os.path.join(output_dir, "episode_summaries.json"),
    }
    write_csv(paths["overall_summary_csv"], overall_rows, OVERALL_FIELDS)
    write_json(paths["overall_summary_json"], {"metadata": metadata, "rows": overall_rows})
    write_csv(paths["order_details_csv"], order_rows, ORDER_DETAIL_FIELDS)
    write_json(paths["order_details_json"], {"metadata": metadata, "rows": order_rows})
    write_csv(paths["drone_details_csv"], drone_rows, DRONE_DETAIL_FIELDS)
    write_json(paths["drone_details_json"], {"metadata": metadata, "rows": drone_rows})
    write_csv(paths["comparison_report_csv"], [comparison], COMPARISON_FIELDS)
    write_json(paths["comparison_report_json"], {"metadata": metadata, "comparison": comparison})
    write_json(paths["episode_summaries_json"], {"metadata": metadata, "rows": episode_summaries})
    write_markdown_report(paths["comparison_report_md"], overall_rows, comparison, metadata)
    return paths

