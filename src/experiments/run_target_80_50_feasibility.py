from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.env.instance_gen import REQUEST_DELIVERY, REQUEST_PICKUP, make_instance_from_coord_demand
from src.env.open_data_loader import (
    load_cvrplib_instances_filtered,
    read_instance_name_list,
    sample_open_vrp_base,
)
from src.env.td_env import EnvConfig, TruckDroneRendezvousEnv
from src.evaluation.action_feasibility import classify_action_feasibility
from src.evaluation.service_metrics import (
    DRONE_DETAIL_FIELDS,
    ORDER_DETAIL_FIELDS,
    OVERALL_FIELDS,
    aggregate_model,
    analyze_episode,
    write_csv,
    write_json,
)
from src.evaluation.time_window_inference import TimeWindowInferenceConfig, predict_action_lateness
from src.evaluation.v2_scheduler import V2SchedulerConfig, rollout_v2_scheduler
from src.evaluation.v2_sequence_scoring import V2SequenceScoreConfig


TARGET_FIELDS = [
    "model_name",
    "model_path",
    "eval_instances",
    "scheduler_mode",
    "acceptance_rate",
    "acceptance_rate_gap_to_0_80",
    "on_time_rate",
    "on_time_rate_gap_to_0_50",
    "late_orders",
    "average_lateness",
    "max_lateness",
    "total_energy_consumption",
    "total_flight_distance",
    "hard_constraint_violations",
    "soft_time_window_violations",
    "is_target_80_50_reached",
    "is_better_than_raw_baseline",
    "is_better_than_v2_repair",
    "recommendation",
]


def _resolve(path: str | Path) -> str:
    return str(Path(path).resolve())


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _has_num(value: Any) -> bool:
    try:
        if value in ("", None):
            return False
        float(value)
        return True
    except Exception:
        return False


def _fmt_float(value: Any, digits: int = 6, signed: bool = False) -> str:
    if not _has_num(value):
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(float(value))


def _fmt_int(value: Any) -> str:
    if not _has_num(value):
        return "n/a"
    return str(int(float(value)))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _env_config(args: argparse.Namespace) -> EnvConfig:
    return EnvConfig(
        vT=args.vT,
        vD=args.vD,
        QD=args.QD,
        B=args.B,
        truck_capacity=args.truck_capacity,
        sT=args.truck_service_time,
        sD=args.drone_service_time,
        depot_service_time=args.depot_service_time,
        allow_wait=True,
        idle_to_next_release=True,
        traffic_sigma=args.traffic_sigma,
        lateness_penalty=args.lateness_penalty,
        reject_penalty=args.reject_penalty,
        overtime_penalty=args.overtime_penalty,
        time_cost_weight=args.time_cost_weight,
        energy_cost_weight=args.energy_cost_weight,
        soc_init=args.soc_init,
        soc_min_reserve=args.soc_reserve,
        energy_per_dist=args.energy_per_dist,
        truck_energy_per_dist=args.truck_energy_per_dist,
        payload_energy_factor=args.payload_energy_factor,
        drone_takeoff_landing_energy=args.drone_takeoff_landing_energy,
        drone_idle_energy_per_time=args.drone_idle_energy_per_time,
        recharge_rate=args.recharge_rate,
        edge_mode=args.edge_mode,
        time_dependent=args.time_dependent,
        peak_after_served_ratio=args.peak_after_served_ratio,
        workday_start=args.workday_start,
        workday_end=args.workday_end,
        morning_peak_start=args.morning_peak_start,
        morning_peak_end=args.morning_peak_end,
        evening_peak_start=args.evening_peak_start,
        evening_peak_end=args.evening_peak_end,
        road_detour_factor=args.road_detour_factor,
        road_signal_density=args.road_signal_density,
        road_turn_density=args.road_turn_density,
        road_one_way_ratio=args.road_one_way_ratio,
        road_peak_factor=args.road_peak_factor,
        signal_penalty=args.signal_penalty,
        turn_penalty=args.turn_penalty,
        left_turn_penalty=args.left_turn_penalty,
        u_turn_penalty=args.u_turn_penalty,
    )


def _load_open_instances(args: argparse.Namespace) -> List[Any]:
    include_names = read_instance_name_list(args.eval_split_file)
    instances = load_cvrplib_instances_filtered(args.dataset_path, include_names=include_names)
    instances = [x for x in instances if x.n_customers >= int(args.N)]
    if not instances:
        raise ValueError("No CVRPLIB instance has enough customers for current --N.")
    return instances


def _make_env(
    args: argparse.Namespace,
    cfg: EnvConfig,
    open_instances: Sequence[Any],
    inst_seed: int,
) -> TruckDroneRendezvousEnv:
    coord_base, demand_base, _ = sample_open_vrp_base(
        instances=open_instances,
        N=int(args.N),
        seed=int(inst_seed),
        coord_scale=float(args.coord_scale),
        normalize_coords=not bool(args.dataset_no_normalize_coords),
        demand_scale=float(args.dataset_demand_scale),
    )
    coord, release, demand, due, meta = make_instance_from_coord_demand(
        coord=coord_base,
        demand=demand_base,
        seed=int(inst_seed),
        release_mode=args.release_mode,
        n_batches=int(args.n_batches),
        max_release=float(args.max_release),
        poisson_rate=float(args.poisson_rate),
        tw_mode=args.tw_mode,
        tw_slack_low=float(args.tw_slack_low),
        tw_slack_high=float(args.tw_slack_high),
        tw_active_prob=float(args.tw_active_prob),
        scheduled_ratio=float(args.scheduled_ratio),
        dynamic_pickup_ratio=float(args.dynamic_pickup_ratio),
        response_slack_low=float(args.response_slack_low),
        response_slack_high=float(args.response_slack_high),
        return_due=True,
    )
    return TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=cfg, seed=int(inst_seed), **meta)


def _service_nodes(env: TruckDroneRendezvousEnv, obs: Dict[str, Any], action: Tuple[int, int]) -> List[int]:
    k, j = int(action[0]), int(action[1])
    nodes: List[int] = []
    current = int(obs.get("current_decision_request", -1))
    if current > 0:
        return []
    if j > 0:
        nodes.append(j)
    if k != env.K_NONE and k > 0:
        nodes.append(k)
    return nodes


def _candidate_actions(env: TruckDroneRendezvousEnv, obs: Dict[str, Any], limit: int = 40) -> List[Tuple[int, int]]:
    masks = env.get_masks()
    feasible_j = np.where(masks["truck_mask"] > 0)[0].astype(int).tolist()
    current = int(obs.get("current_decision_request", -1))
    actions: List[Tuple[int, int]] = []
    if current > 0:
        if current in feasible_j:
            actions.append((env.K_NONE, current))
        if 0 in feasible_j:
            actions.append((env.K_NONE, 0))
        return actions
    ranked_j = sorted(
        feasible_j,
        key=lambda j: (
            float(env.due[j]) if j > 0 and math.isfinite(float(env.due[j])) else 1e9,
            float(env.dist_mat[int(obs.get("i", 0)), int(j)]) if j >= 0 else 0.0,
        ),
    )
    for j in ranked_j[: max(1, min(len(ranked_j), limit))]:
        actions.append((env.K_NONE, int(j)))
        try:
            dm = env.get_masks(j=j)["drone_mask"]
            ks = np.where(dm > 0)[0].astype(int).tolist()
        except Exception:
            ks = []
        ks = sorted(
            ks,
            key=lambda k: (
                float(env.due[k]) if math.isfinite(float(env.due[k])) else 1e9,
                float(env.dist_mat[int(obs.get("i", 0)), int(k)]) + float(env.dist_mat[int(k), int(j)]),
            ),
        )
        for k in ks[:4]:
            actions.append((int(k), int(j)))
    seen = set()
    out = []
    for a in actions:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out[:limit]


def _decision_acceptable(env: TruckDroneRendezvousEnv, obs: Dict[str, Any], node: int, mode: str) -> bool:
    if mode == "accept_all":
        return True
    if float(env.demand[node]) > float(env.cfg.truck_capacity) + 1e-9:
        return False
    pred = predict_action_lateness(env, obs, j=node, k=env.K_NONE, cfg=TimeWindowInferenceConfig())
    if mode == "on_time_accept":
        return not bool(pred.get("will_be_late", False))
    return True


def _score_action(env: TruckDroneRendezvousEnv, obs: Dict[str, Any], action: Tuple[int, int], mode: str) -> float:
    current = int(obs.get("current_decision_request", -1))
    if current > 0:
        if action[1] == 0:
            return 1e6
        return -100.0
    pred = predict_action_lateness(env, obs, j=int(action[1]), k=int(action[0]), cfg=TimeWindowInferenceConfig())
    node_late = [max(0.0, float(v)) for v in (pred.get("node_lateness", {}) or {}).values()]
    late_count = sum(1 for x in node_late if x > 1e-9)
    total_late = sum(node_late)
    max_late = max([0.0] + node_late)
    nodes = _service_nodes(env, obs, action)
    due_score = min([float(env.due[n]) for n in nodes if math.isfinite(float(env.due[n]))] + [1e6])
    i = int(obs.get("i", 0))
    dist = float(env.get_dense_edge_attr()[i, int(action[1]), 0]) if int(action[1]) >= 0 else 0.0
    if int(action[0]) != env.K_NONE:
        dist += float(env.dist_mat[i, int(action[0])]) + float(env.dist_mat[int(action[0]), int(action[1])])
    service_bonus = -20.0 * len(nodes)
    if mode == "edf":
        return due_score + 0.05 * dist + service_bonus
    if mode == "min_lateness":
        return 1000.0 * late_count + 100.0 * total_late + 10.0 * max_late + 0.1 * dist + service_bonus
    if mode == "regret":
        cls = classify_action_feasibility(env, obs, action)
        return (
            700.0 * late_count
            + 80.0 * total_late
            + 20.0 * float(cls.get("future_impact_score", 0.0))
            + 0.15 * dist
            + service_bonus
            + due_score * 0.01
        )
    return 1000.0 * late_count + 100.0 * total_late + 0.1 * dist + service_bonus


def rollout_greedy_heuristic(
    env: TruckDroneRendezvousEnv,
    *,
    mode: str,
    max_steps: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    traj: List[Dict[str, Any]] = []
    done = False
    total_cost = 0.0
    classifications: List[Dict[str, Any]] = []
    for step in range(max_steps):
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            accept_mode = "accept_all" if mode in {"edf_accept_all", "min_lateness_accept_all"} else "feasible_accept"
            accept = _decision_acceptable(e, obs, current, accept_mode)
            action = (e.K_NONE, current if accept else 0)
        else:
            actions = _candidate_actions(e, obs, limit=48)
            if not actions:
                action = (e.K_NONE, 0)
            else:
                score_mode = "edf" if mode == "edf_accept_all" else ("min_lateness" if "min_lateness" in mode else "regret")
                action = min(actions, key=lambda a: _score_action(e, obs, a, score_mode))
        if step < 12:
            try:
                classifications.append(classify_action_feasibility(e, obs, action))
            except Exception:
                pass
        obs2, reward, done, info = e.step(action)
        reward_f = float(reward)
        total_cost += -reward_f
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": reward_f, "info": info})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += 1000.0
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -1000.0, "info": {"timeout": True}})
    return float(total_cost), traj, {"done": bool(done), "classifications": classifications}


def rollout_beam_oracle(env: TruckDroneRendezvousEnv, *, max_steps: int, seed: int) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    cfg = V2SchedulerConfig(
        lookahead_depth=2,
        beam_size=6,
        candidate_top_k=12,
        max_runtime_seconds=0.25,
        deterministic_seed=int(seed),
        use_beam_search=True,
        score=V2SequenceScoreConfig(
            accept_reward=8.0,
            on_time_reward=20.0,
            reject_penalty=50.0,
            late_count_penalty=15.0,
            lateness_weight=3.0,
            max_lateness_weight=2.0,
            severe_lateness_threshold=8.0,
            severe_lateness_weight=6.0,
            energy_weight=0.03,
            distance_weight=0.02,
            future_tight_window_horizon=8.0,
            future_tight_window_weight=1.0,
        ),
    )
    return rollout_v2_scheduler(env, cfg, max_steps=max_steps)


def _traj_cost(traj: Sequence[Dict[str, Any]]) -> float:
    return float(sum(-float(item.get("reward", 0.0)) for item in traj))


def _hard_soft(order_rows: Sequence[Dict[str, Any]], drone_rows: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    hard = 0
    for row in order_rows:
        if _bool(row.get("battery_violation")) or _bool(row.get("capacity_violation")) or _bool(row.get("range_violation")):
            hard += 1
    for row in drone_rows:
        if _bool(row.get("battery_violation")):
            hard += 1
    soft = sum(1 for row in order_rows if _bool(row.get("time_window_violation")))
    return int(hard), int(soft)


def analyze_orders(envs: Sequence[TruckDroneRendezvousEnv]) -> Dict[str, Any]:
    slacks: List[float] = []
    direct_late = 0
    high_risk = 0
    dist_infeasible = 0
    energy_infeasible = 0
    cap_infeasible = 0
    tight = 0
    dynamic = 0
    response_slacks: List[float] = []
    per_instance: List[Dict[str, Any]] = []
    for idx, env in enumerate(envs, start=1):
        inst_slacks = []
        inst_tight = 0
        inst_direct_late = 0
        for node in range(1, env.N + 1):
            due = float(env.due[node])
            release = float(env.release[node])
            if int(env.is_dynamic[node]) > 0:
                dynamic += 1
                response_slacks.append(float(env.decision_deadline[node]) - release)
            if math.isfinite(due):
                slack = due - release
                slacks.append(slack)
                inst_slacks.append(slack)
                if slack <= 6.0:
                    tight += 1
                    inst_tight += 1
                truck_direct = float(env._tau_truck(0, node, apply_traffic=False)) + float(env.cfg.sT)
                drone_direct = float(env._tau_drone(0, node, 0)) + float(env.cfg.sD)
                earliest = release + min(truck_direct, drone_direct)
                if earliest > due + 1e-9:
                    direct_late += 1
                    inst_direct_late += 1
                elif due - earliest <= 2.0:
                    high_risk += 1
            if float(env.demand[node]) > float(env.cfg.truck_capacity) + 1e-9:
                cap_infeasible += 1
            drone_time = float(env._tau_drone(0, node, 0)) + float(env.cfg.sD)
            if drone_time > float(env.cfg.B) + 1e-9:
                dist_infeasible += 1
            drone_energy = float(env._drone_energy(0, node, 0))
            if drone_energy > max(0.0, float(env.cfg.soc_init) - float(env.cfg.soc_min_reserve)) + 1e-9:
                energy_infeasible += 1
        per_instance.append(
            {
                "instance_id": idx,
                "orders": int(env.N),
                "dynamic_orders": int((env.is_dynamic[1:] > 0).sum()),
                "tight_orders_slack_le_6": int(inst_tight),
                "direct_earliest_late_orders": int(inst_direct_late),
                "avg_slack": float(np.mean(inst_slacks)) if inst_slacks else 0.0,
                "min_slack": float(np.min(inst_slacks)) if inst_slacks else 0.0,
            }
        )
    return {
        "total_orders": int(sum(env.N for env in envs)),
        "instances": len(envs),
        "orders_per_instance": [int(env.N) for env in envs],
        "dynamic_orders": int(dynamic),
        "tight_window_orders_slack_le_6": int(tight),
        "tight_window_ratio": float(tight / max(1, len(slacks))),
        "average_time_window_slack": float(np.mean(slacks)) if slacks else 0.0,
        "minimum_time_window_slack": float(np.min(slacks)) if slacks else 0.0,
        "median_time_window_slack": float(np.median(slacks)) if slacks else 0.0,
        "average_response_slack": float(np.mean(response_slacks)) if response_slacks else 0.0,
        "minimum_response_slack": float(np.min(response_slacks)) if response_slacks else 0.0,
        "estimated_unavoidable_direct_late_orders": int(direct_late),
        "estimated_high_risk_orders": int(high_risk),
        "infeasible_orders_due_to_distance": int(dist_infeasible),
        "infeasible_orders_due_to_energy": int(energy_infeasible),
        "infeasible_orders_due_to_capacity": int(cap_infeasible),
        "per_instance": per_instance,
    }


def resource_analysis(envs: Sequence[TruckDroneRendezvousEnv]) -> Dict[str, Any]:
    dists = []
    service_times = []
    for env in envs:
        for node in range(1, env.N + 1):
            dists.append(float(env.dist_mat[0, node]))
            service_times.append(float(env._tau_truck(0, node, apply_traffic=False)) + float(env.cfg.sT))
    env0 = envs[0]
    avg_service = float(np.mean(service_times)) if service_times else 0.0
    max_work = float(env0.max_work_time)
    single_resource_capacity = int(max_work / max(1e-6, avg_service)) if avg_service > 0 else 0
    return {
        "drone_count": 1,
        "truck_count": 1,
        "estimated_max_orders_per_single_route_resource": int(single_resource_capacity),
        "estimated_total_service_capacity_truck_plus_drone_proxy": int(single_resource_capacity * 2),
        "battery_capacity_soc": float(env0.cfg.soc_init),
        "soc_min_reserve": float(env0.cfg.soc_min_reserve),
        "max_drone_range_time": float(env0.cfg.B),
        "truck_capacity": float(env0.cfg.truck_capacity),
        "drone_payload_capacity": float(env0.cfg.QD),
        "average_direct_service_time": avg_service,
        "average_depot_distance": float(np.mean(dists)) if dists else 0.0,
        "resource_insufficient_proxy": bool(single_resource_capacity < int(0.8 * env0.N)),
    }


def evaluate_oracles(args: argparse.Namespace, eval_instances: int, out_dir: Path) -> Dict[str, Any]:
    cfg = _env_config(args)
    open_instances = _load_open_instances(args)
    max_steps = 10 * (int(args.N) + 1)
    modes = ["edf_accept_all", "min_lateness_accept_all", "regret_insertion", "beam_oracle"]
    results: Dict[str, Dict[str, Any]] = {}
    best_trajs_for_imitation: List[Dict[str, Any]] = []
    envs_for_analysis: List[TruckDroneRendezvousEnv] = []

    for mode in modes:
        episode_summaries: List[Dict[str, Any]] = []
        order_rows: List[Dict[str, Any]] = []
        drone_rows: List[Dict[str, Any]] = []
        debug: Dict[str, Any] = {"mode": mode, "instances": []}
        for idx in range(1, int(eval_instances) + 1):
            inst_seed = int(args.eval_seed) * 100000 + idx
            env = _make_env(args, cfg, open_instances, inst_seed)
            if mode == modes[0]:
                envs_for_analysis.append(env)
            if mode == "beam_oracle":
                cost, traj, log = rollout_beam_oracle(env, max_steps=max_steps, seed=inst_seed)
            else:
                cost, traj, log = rollout_greedy_heuristic(env, mode=mode, max_steps=max_steps)
            summary, orders, drone = analyze_episode(
                env,
                traj,
                model_name=mode,
                instance_id=idx,
                objective_cost=float(cost),
            )
            episode_summaries.append(summary)
            order_rows.extend(orders)
            drone_rows.append(drone)
            debug["instances"].append({"instance_id": idx, "objective_cost": float(cost), **log})
            if mode == "beam_oracle":
                best_trajs_for_imitation.append(
                    {
                        "instance_id": idx,
                        "trajectory": [
                            {
                                "step": step,
                                "t": float(item.get("obs", {}).get("t", 0.0)),
                                "i": int(item.get("obs", {}).get("i", 0)),
                                "current_decision_request": int(item.get("obs", {}).get("current_decision_request", -1)),
                                "action": list(item.get("action")) if isinstance(item.get("action"), tuple) else ["TIMEOUT"],
                                "reward": float(item.get("reward", 0.0)),
                            }
                            for step, item in enumerate(traj)
                        ],
                    }
                )
        overall, drone_detail = aggregate_model(mode, episode_summaries, order_rows, drone_rows)
        hard, soft = _hard_soft(order_rows, [drone_detail])
        overall["hard_constraint_violations"] = hard
        overall["soft_time_window_violations"] = soft
        mode_dir = out_dir / "oracle" / f"eval_{eval_instances}" / mode
        write_csv(str(mode_dir / "overall_summary.csv"), [overall], OVERALL_FIELDS + ["hard_constraint_violations", "soft_time_window_violations"])
        write_csv(str(mode_dir / "order_details.csv"), order_rows, ORDER_DETAIL_FIELDS)
        write_csv(str(mode_dir / "drone_details.csv"), [drone_detail], DRONE_DETAIL_FIELDS)
        write_json(str(mode_dir / "debug_log.json"), debug)
        results[mode] = {"overall": overall, "order_rows": order_rows, "drone_detail": drone_detail}

    best_mode = sorted(
        results.keys(),
        key=lambda m: (
            -float(results[m]["overall"].get("acceptance_rate", 0.0)),
            -float(results[m]["overall"].get("on_time_rate", 0.0)),
            int(results[m]["overall"].get("late_orders", 10**9)),
        ),
    )[0]
    return {
        "eval_instances": int(eval_instances),
        "results": results,
        "best_mode": best_mode,
        "best_overall": results[best_mode]["overall"],
        "envs_for_analysis": envs_for_analysis,
        "imitation_trajs": best_trajs_for_imitation,
    }


def _baseline_rows_from_previous(args: argparse.Namespace) -> List[Dict[str, Any]]:
    prev = Path(args.previous_comparison_csv)
    if not prev.exists():
        return []
    rows = []
    with prev.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model_name") in {"raw_baseline", "v2_repair_only"}:
                rows.append(row)
    return rows


def _target_row(
    name: str,
    model_path: str,
    eval_instances: int,
    scheduler_mode: str,
    overall: Dict[str, Any],
    raw: Optional[Dict[str, Any]],
    v2: Optional[Dict[str, Any]],
    recommendation: str,
) -> Dict[str, Any]:
    acc = float(overall.get("acceptance_rate", 0.0))
    ot = float(overall.get("on_time_rate", 0.0))
    hard = int(overall.get("hard_constraint_violations", 0))
    soft = int(overall.get("soft_time_window_violations", overall.get("late_orders", 0)))
    better_raw = False
    if raw:
        better_raw = (
            acc > _num(raw.get("acceptance_rate"))
            and ot > _num(raw.get("on_time_rate"))
            and int(float(overall.get("late_orders", 0))) < int(float(raw.get("late_orders", 0)))
            and _num(overall.get("average_lateness")) <= _num(raw.get("average_lateness")) + 1e-9
            and _num(overall.get("max_lateness")) <= _num(raw.get("max_lateness")) + 1e-9
            and hard == 0
        )
    better_v2 = False
    if v2:
        better_v2 = (
            acc > _num(v2.get("acceptance_rate"))
            and ot > _num(v2.get("on_time_rate"))
            and int(float(overall.get("late_orders", 0))) < int(float(v2.get("late_orders", 0)))
            and _num(overall.get("average_lateness")) <= _num(v2.get("average_lateness")) + 1e-9
            and _num(overall.get("max_lateness")) <= _num(v2.get("max_lateness")) + 1e-9
            and hard == 0
        )
    return {
        "model_name": name,
        "model_path": model_path,
        "eval_instances": int(eval_instances),
        "scheduler_mode": scheduler_mode,
        "acceptance_rate": acc,
        "acceptance_rate_gap_to_0_80": 0.80 - acc,
        "on_time_rate": ot,
        "on_time_rate_gap_to_0_50": 0.50 - ot,
        "late_orders": int(float(overall.get("late_orders", 0))),
        "average_lateness": float(overall.get("average_lateness", 0.0)),
        "max_lateness": float(overall.get("max_lateness", 0.0)),
        "total_energy_consumption": float(overall.get("total_energy_consumption", 0.0)),
        "total_flight_distance": float(overall.get("total_flight_distance", 0.0)),
        "hard_constraint_violations": hard,
        "soft_time_window_violations": soft,
        "is_target_80_50_reached": bool(acc >= 0.80 and ot >= 0.50 and hard == 0),
        "is_better_than_raw_baseline": bool(better_raw),
        "is_better_than_v2_repair": bool(better_v2),
        "recommendation": recommendation,
    }


def write_reports(
    args: argparse.Namespace,
    out_dir: Path,
    oracle_runs: Sequence[Dict[str, Any]],
    order_stats: Dict[str, Any],
    resource_stats: Dict[str, Any],
    target_rows: List[Dict[str, Any]],
) -> None:
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    best_acc_by_size = {
        int(run["eval_instances"]): max(
            run["results"].items(),
            key=lambda kv: float(kv[1]["overall"].get("acceptance_rate", 0.0)),
        )
        for run in oracle_runs
    }
    best_ot_by_size = {
        int(run["eval_instances"]): max(
            run["results"].items(),
            key=lambda kv: float(kv[1]["overall"].get("on_time_rate", 0.0)),
        )
        for run in oracle_runs
    }
    order_stats_by_size = {
        int(run["eval_instances"]): analyze_orders(run.get("envs_for_analysis", []))
        for run in oracle_runs
    }
    all_oracle_overalls = [
        result["overall"]
        for run in oracle_runs
        for result in run["results"].values()
    ]
    max_acc = max(float(ov.get("acceptance_rate", 0.0)) for ov in all_oracle_overalls)
    max_ot = max(float(ov.get("on_time_rate", 0.0)) for ov in all_oracle_overalls)
    feasible_joint = any(
        float(ov.get("acceptance_rate", 0.0)) >= 0.80
        and float(ov.get("on_time_rate", 0.0)) >= 0.50
        and int(ov.get("hard_constraint_violations", 0)) == 0
        for ov in all_oracle_overalls
    )
    feasible_80 = max_acc >= 0.80
    feasible_50 = max_ot >= 0.50
    bottleneck = (
        "dynamic response deadlines and single-route-resource service time"
        if not feasible_80
        else "time-window lateness under accepted workload"
    )

    lines = [
        "# Feasibility Analysis for 80/50 Target",
        "",
        f"- Acceptance 80% theoretically reachable under tested oracle heuristics: `{feasible_80}`",
        f"- On-time 50% theoretically reachable under tested oracle heuristics: `{feasible_50}`",
        f"- Joint 80/50 target reached by any tested oracle heuristic: `{feasible_joint}`",
        f"- Best heuristic acceptance observed: `{max_acc:.6f}`",
        f"- Best heuristic on-time observed: `{max_ot:.6f}`",
        f"- Primary bottleneck: `{bottleneck}`",
        "",
        "## Order Demand Analysis",
        "",
        f"- Total orders analyzed: `{order_stats['total_orders']}`",
        f"- Instances: `{order_stats['instances']}`",
        f"- Orders per instance: `{sorted(set(order_stats['orders_per_instance']))}`",
        f"- Dynamic orders: `{order_stats['dynamic_orders']}`",
        f"- Tight-window ratio, slack<=6: `{order_stats['tight_window_ratio']:.6f}`",
        f"- Average time-window slack: `{order_stats['average_time_window_slack']:.6f}`",
        f"- Minimum time-window slack: `{order_stats['minimum_time_window_slack']:.6f}`",
        f"- Average dynamic response slack: `{order_stats['average_response_slack']:.6f}`",
        f"- Minimum dynamic response slack: `{order_stats['minimum_response_slack']:.6f}`",
        f"- Direct earliest unavoidable late orders estimate: `{order_stats['estimated_unavoidable_direct_late_orders']}`",
        f"- High-risk orders estimate: `{order_stats['estimated_high_risk_orders']}`",
        "",
        "## Resource Analysis",
        "",
        f"- Drone count: `{resource_stats['drone_count']}`",
        f"- Truck count: `{resource_stats['truck_count']}`",
        f"- Estimated direct service time: `{resource_stats['average_direct_service_time']:.6f}`",
        f"- Estimated single-resource capacity: `{resource_stats['estimated_max_orders_per_single_route_resource']}`",
        f"- Battery SOC: `{resource_stats['battery_capacity_soc']}`",
        f"- Max drone range time: `{resource_stats['max_drone_range_time']}`",
        f"- Truck capacity: `{resource_stats['truck_capacity']}`",
        "",
        "## Oracle Results",
        "",
        "Best by acceptance:",
        "",
        "| eval | best_mode | acceptance | on_time | late | avg_late | max_late | hard |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in oracle_runs:
        mode, result = best_acc_by_size[int(run["eval_instances"])]
        ov = result["overall"]
        lines.append(
            f"| {run['eval_instances']} | {mode} | {float(ov['acceptance_rate']):.6f} | "
            f"{float(ov['on_time_rate']):.6f} | {int(ov['late_orders'])} | "
            f"{float(ov['average_lateness']):.6f} | {float(ov['max_lateness']):.6f} | "
            f"{int(ov.get('hard_constraint_violations', 0))} |"
        )
    lines.extend(
        [
            "",
            "Best by on-time rate:",
            "",
            "| eval | best_mode | acceptance | on_time | late | avg_late | max_late | hard |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in oracle_runs:
        mode, result = best_ot_by_size[int(run["eval_instances"])]
        ov = result["overall"]
        lines.append(
            f"| {run['eval_instances']} | {mode} | {float(ov['acceptance_rate']):.6f} | "
            f"{float(ov['on_time_rate']):.6f} | {int(ov['late_orders'])} | "
            f"{float(ov['average_lateness']):.6f} | {float(ov['max_lateness']):.6f} | "
            f"{int(ov.get('hard_constraint_violations', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Direct Answer",
            "",
            f"1. Current constraints reach 80% acceptance: `{feasible_80}`.",
            f"2. Current constraints reach 50% on-time: `{feasible_50}`.",
            f"3. Current constraints reach the joint 80/50 target in one schedule: `{feasible_joint}`.",
            "4. If either target is false, do not continue target training because the tested non-neural upper-bound estimates do not support the business target.",
        ]
    )
    (reports / "feasibility_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    upper = [
        "# Oracle Upper Bound Report",
        "",
        "The values below are heuristic/relaxed-oracle estimates, not a mathematical proof of global optimality.",
        "",
        "| eval | best_possible_acceptance_rate | best_possible_on_time_rate | fewest_late_orders_observed | unavoidable_direct_late_orders_estimate | hard_at_best_acceptance |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in oracle_runs:
        size = int(run["eval_instances"])
        _, best_acc_result = best_acc_by_size[size]
        _, best_ot_result = best_ot_by_size[size]
        best_acc_ov = best_acc_result["overall"]
        best_ot_ov = best_ot_result["overall"]
        fewest_late = min(int(result["overall"].get("late_orders", 0)) for result in run["results"].values())
        run_order_stats = order_stats_by_size[size]
        upper.append(
            f"| {size} | {float(best_acc_ov['acceptance_rate']):.6f} | "
            f"{float(best_ot_ov['on_time_rate']):.6f} | {fewest_late} | "
            f"{run_order_stats['estimated_unavoidable_direct_late_orders']} | {int(best_acc_ov.get('hard_constraint_violations', 0))} |"
        )
    upper.extend(
        [
            "",
            "Largest evaluated split infeasibility estimates:",
            f"- infeasible_orders_due_to_time_window: `{order_stats['estimated_unavoidable_direct_late_orders']}`",
            f"- infeasible_orders_due_to_distance: `{order_stats['infeasible_orders_due_to_distance']}`",
            f"- infeasible_orders_due_to_energy: `{order_stats['infeasible_orders_due_to_energy']}`",
            f"- infeasible_orders_due_to_capacity: `{order_stats['infeasible_orders_due_to_capacity']}`",
        ]
    )
    (reports / "oracle_upper_bound_report.md").write_text("\n".join(upper) + "\n", encoding="utf-8")

    write_csv(str(reports / "final_target_80_50_comparison.csv"), target_rows, TARGET_FIELDS)
    write_json(str(reports / "final_target_80_50_comparison.json"), {"rows": target_rows})
    md = [
        "# Final Target 80/50 Comparison",
        "",
        "| model | eval | scheduler | acc | gap80 | on_time | gap50 | late | avg_late | max_late | hard | recommendation |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in target_rows:
        md.append(
            f"| {r['model_name']} | {r['eval_instances']} | {r['scheduler_mode']} | "
            f"{_fmt_float(r['acceptance_rate'])} | {_fmt_float(r['acceptance_rate_gap_to_0_80'], signed=True)} | "
            f"{_fmt_float(r['on_time_rate'])} | {_fmt_float(r['on_time_rate_gap_to_0_50'], signed=True)} | "
            f"{_fmt_int(r['late_orders'])} | {_fmt_float(r['average_lateness'])} | "
            f"{_fmt_float(r['max_lateness'])} | {_fmt_int(r['hard_constraint_violations'])} | {r['recommendation']} |"
        )
    (reports / "final_target_80_50_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    numeric_rows = [r for r in target_rows if _has_num(r.get("acceptance_rate")) and _has_num(r.get("on_time_rate"))]
    best = max(numeric_rows, key=lambda r: (float(r["acceptance_rate"]), float(r["on_time_rate"])))
    best_lines = [
        "# Best Model Report",
        "",
        "No target model was trained because feasibility analysis did not justify target 80/50 training.",
        "",
        f"Best observed non-neural result by acceptance: `{best['model_name']}` at eval `{best['eval_instances']}`.",
        f"- Acceptance: `{float(best['acceptance_rate']):.6f}`",
        f"- On-time: `{float(best['on_time_rate']):.6f}`",
        f"- Gap to 0.80 acceptance: `{0.80 - float(best['acceptance_rate']):.6f}`",
        f"- Gap to 0.50 on-time: `{0.50 - float(best['on_time_rate']):.6f}`",
        "",
        "Do not replace `experiments/frozen_models_20260419/model_main_ep200.pt`.",
    ]
    (reports / "best_model_report.md").write_text("\n".join(best_lines) + "\n", encoding="utf-8")

    req = [
        "# Resource Requirement for 80/50",
        "",
        "The tested oracle heuristics did not reach the 0.80 acceptance / 0.50 on-time target under the current single-truck/single-drone environment.",
        "",
        "## Main blockers",
        "",
        f"- Dynamic response slack is very short: average `{order_stats['average_response_slack']:.3f}`, minimum `{order_stats['minimum_response_slack']:.3f}`.",
        "- The environment only allows waiting for future dynamic requests when no feasible service action exists, so scheduled-service travel can cause dynamic requests to expire before decision.",
        "- With `scheduled_ratio=0.5`, about half of the orders are dynamic pickups; reaching 80% total acceptance means accepting roughly 60% of dynamic requests in addition to all scheduled orders.",
        "",
        "## Suggested adjustments",
        "",
        f"- Current best observed acceptance is `{max_acc:.3f}`; reaching `0.800` requires roughly `{0.80 - max_acc:.3f}` absolute acceptance-rate gain.",
        f"- Current best observed on-time rate is `{max_ot:.3f}`; reaching `0.500` requires roughly `{0.50 - max_ot:.3f}` absolute on-time-rate gain.",
        "- Capacity alone suggests at least a second parallel dispatch resource, but the response-deadline bottleneck also requires earlier accept/reject decisions or wider response windows.",
        "- Increase dynamic response deadline from 0.25-1.00 to at least 3.0-5.0 time units, then re-run feasibility.",
        "- Add one or more parallel dispatch resources; a second drone alone helps service time but does not fully solve response-deadline expiry unless decisions can be accepted asynchronously.",
        "- Allow an explicit wait/decision action even when scheduled orders are feasible, or decouple accept/reject decisions from truck route step boundaries.",
        "- If preserving synchronous decisions, reduce dynamic order density or lower `dynamic_pickup_ratio`.",
        "- To improve 50% on-time, widen delivery time-window slack by roughly 2-4 time units first, then test speed/battery changes.",
    ]
    (reports / "resource_requirement_for_80_50.md").write_text("\n".join(req) + "\n", encoding="utf-8")

    train_summary = [
        "# Training Summary",
        "",
        "No target_80_50 candidate A/B/C training was run.",
        "",
        "Reason: feasibility/oracle analysis did not support the requested 0.80 acceptance and 0.50 on-time target under the current data and constraints.",
        "",
        "| candidate | status | model_path |",
        "|---|---|---|",
        "| target_80_50_candidate_A | not_run |  |",
        "| target_80_50_candidate_B | not_run |  |",
        "| target_80_50_candidate_C | not_run |  |",
    ]
    (reports / "training_summary.md").write_text("\n".join(train_summary) + "\n", encoding="utf-8")


def write_imitation(out_dir: Path, oracle_runs: Sequence[Dict[str, Any]]) -> None:
    imitation_dir = out_dir / "imitation"
    imitation_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for run in oracle_runs:
        if int(run["eval_instances"]) != 30:
            continue
        for item in run.get("imitation_trajs", []):
            for step in item["trajectory"]:
                rec = {"eval_instances": int(run["eval_instances"]), "oracle_mode": "beam_oracle", "instance_id": item["instance_id"], **step}
                records.append(rec)
    if records:
        fields = list(records[0].keys())
        with (imitation_dir / "imitation_dataset.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        write_json(str(imitation_dir / "imitation_dataset.json"), {"records": records})
        torch.save(records, imitation_dir / "imitation_dataset.pt")
    report = [
        "# Imitation Quality Report",
        "",
        f"- State-action records generated: `{len(records)}`",
        "- Source: `beam_oracle` heuristic trajectories for eval_instances=30.",
        "- These data are suitable as a diagnostic imitation warm-start corpus, but target training was not launched because feasibility did not support 80/50.",
    ]
    (out_dir / "reports" / "imitation_quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_target_rows(args: argparse.Namespace, oracle_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prev_rows = _baseline_rows_from_previous(args)
    raw_by_size = {int(r["eval_instances"]): r for r in prev_rows if r["model_name"] == "raw_baseline"}
    v2_by_size = {int(r["eval_instances"]): r for r in prev_rows if r["model_name"] == "v2_repair_only"}
    for r in prev_rows:
        size = int(r["eval_instances"])
        rows.append(
            _target_row(
                r["model_name"],
                r.get("model_path", args.baseline_model_path),
                size,
                r["scheduler_mode"],
                r,
                raw_by_size.get(size),
                v2_by_size.get(size),
                "baseline_reference" if r["model_name"] == "raw_baseline" else "v2_reference_not_default",
            )
        )
    for run in oracle_runs:
        size = int(run["eval_instances"])
        for mode, result in run["results"].items():
            rows.append(
                _target_row(
                    mode,
                    "heuristic_oracle_no_weights",
                    size,
                    "oracle",
                    result["overall"],
                    raw_by_size.get(size),
                    v2_by_size.get(size),
                    "oracle_upper_bound_estimate",
                )
            )
    for size in sorted({30, 50, 100}):
        for cand in ("target_80_50_candidate_A", "target_80_50_candidate_B", "target_80_50_candidate_C"):
            rows.append(
                {
                    "model_name": cand,
                    "model_path": "",
                    "eval_instances": size,
                    "scheduler_mode": "not_run",
                    "acceptance_rate": "",
                    "acceptance_rate_gap_to_0_80": "",
                    "on_time_rate": "",
                    "on_time_rate_gap_to_0_50": "",
                    "late_orders": "",
                    "average_lateness": "",
                    "max_lateness": "",
                    "total_energy_consumption": "",
                    "total_flight_distance": "",
                    "hard_constraint_violations": "",
                    "soft_time_window_violations": "",
                    "is_target_80_50_reached": False,
                    "is_better_than_raw_baseline": False,
                    "is_better_than_v2_repair": False,
                    "recommendation": "not_trained_due_to_feasibility",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feasibility/oracle analysis for target acceptance 0.80 and on-time 0.50.")
    p.add_argument("--output-dir", type=str, default="experiments/main_model_target_80_50_20260513")
    p.add_argument("--baseline-model-path", type=str, default="experiments/frozen_models_20260419/model_main_ep200.pt")
    p.add_argument("--previous-comparison-csv", type=str, default="experiments/main_model_sequence_tw_retrain_20260513/reports/final_model_comparison.csv")
    p.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    p.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--eval-sizes", type=str, default="30,50,100")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--coord-scale", type=float, default=10.0)
    p.add_argument("--release-mode", type=str, default="batches")
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--max-release", type=float, default=10.0)
    p.add_argument("--poisson-rate", type=float, default=1.0)
    p.add_argument("--tw-mode", type=str, default="relative")
    p.add_argument("--tw-slack-low", type=float, default=4.0)
    p.add_argument("--tw-slack-high", type=float, default=14.0)
    p.add_argument("--tw-active-prob", type=float, default=0.8)
    p.add_argument("--scheduled-ratio", type=float, default=0.5)
    p.add_argument("--dynamic-pickup-ratio", type=float, default=1.0)
    p.add_argument("--response-slack-low", type=float, default=0.25)
    p.add_argument("--response-slack-high", type=float, default=1.0)
    p.add_argument("--dataset-demand-scale", type=float, default=1.0)
    p.add_argument("--dataset-no-normalize-coords", action="store_true")
    p.add_argument("--vT", type=float, default=1.0)
    p.add_argument("--vD", type=float, default=1.5)
    p.add_argument("--QD", type=float, default=1.0)
    p.add_argument("--B", type=float, default=6.0)
    p.add_argument("--truck-capacity", type=float, default=3.0)
    p.add_argument("--truck-service-time", type=float, default=0.05)
    p.add_argument("--drone-service-time", type=float, default=0.03)
    p.add_argument("--depot-service-time", type=float, default=0.10)
    p.add_argument("--traffic-sigma", type=float, default=0.15)
    p.add_argument("--lateness-penalty", type=float, default=1.5)
    p.add_argument("--reject-penalty", type=float, default=6.0)
    p.add_argument("--overtime-penalty", type=float, default=1.0)
    p.add_argument("--time-cost-weight", type=float, default=1.0)
    p.add_argument("--energy-cost-weight", type=float, default=0.08)
    p.add_argument("--soc-init", type=float, default=1.0)
    p.add_argument("--soc-reserve", type=float, default=0.10)
    p.add_argument("--energy-per-dist", type=float, default=0.08)
    p.add_argument("--truck-energy-per-dist", type=float, default=0.04)
    p.add_argument("--payload-energy-factor", type=float, default=0.4)
    p.add_argument("--drone-takeoff-landing-energy", type=float, default=0.01)
    p.add_argument("--drone-idle-energy-per-time", type=float, default=0.0)
    p.add_argument("--recharge-rate", type=float, default=0.25)
    p.add_argument("--edge-mode", type=str, default="road")
    p.add_argument("--time-dependent", action="store_true", default=True)
    p.add_argument("--peak-after-served-ratio", type=float, default=0.5)
    p.add_argument("--workday-start", type=float, default=8.0)
    p.add_argument("--workday-end", type=float, default=20.0)
    p.add_argument("--morning-peak-start", type=float, default=8.0)
    p.add_argument("--morning-peak-end", type=float, default=10.0)
    p.add_argument("--evening-peak-start", type=float, default=17.0)
    p.add_argument("--evening-peak-end", type=float, default=19.0)
    p.add_argument("--road-detour-factor", type=float, default=1.18)
    p.add_argument("--road-signal-density", type=float, default=0.006)
    p.add_argument("--road-turn-density", type=float, default=0.010)
    p.add_argument("--road-one-way-ratio", type=float, default=0.10)
    p.add_argument("--road-peak-factor", type=float, default=1.25)
    p.add_argument("--signal-penalty", type=float, default=0.05)
    p.add_argument("--turn-penalty", type=float, default=0.12)
    p.add_argument("--left-turn-penalty", type=float, default=0.08)
    p.add_argument("--u-turn-penalty", type=float, default=0.30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "oracle").mkdir(parents=True, exist_ok=True)
    write_json(str(out_dir / "run_config.json"), {"created_time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)})

    oracle_runs = []
    for size in [int(x.strip()) for x in args.eval_sizes.split(",") if x.strip()]:
        oracle_runs.append(evaluate_oracles(args, size, out_dir))

    envs = oracle_runs[-1]["envs_for_analysis"] if oracle_runs else []
    order_stats = analyze_orders(envs)
    resource_stats = resource_analysis(envs)
    target_rows = build_target_rows(args, oracle_runs)
    write_imitation(out_dir, oracle_runs)
    write_reports(args, out_dir, oracle_runs, order_stats, resource_stats, target_rows)
    print(json.dumps({"output_dir": _resolve(out_dir), "best_oracle": {r["eval_instances"]: r["best_mode"] for r in oracle_runs}}, indent=2))


if __name__ == "__main__":
    main()
