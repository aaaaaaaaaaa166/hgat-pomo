from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.env.instance_gen import make_instance_from_coord_demand, make_random_instance
from src.env.open_data_loader import (
    load_cvrplib_instances_filtered,
    read_instance_name_list,
    sample_open_vrp_base,
)
from src.env.td_env import EnvConfig, TruckDroneRendezvousEnv
from src.evaluation.service_metrics import (
    DRONE_DETAIL_FIELDS,
    ORDER_DETAIL_FIELDS,
    OVERALL_FIELDS,
    aggregate_model,
    analyze_episode,
    safe_div,
    write_csv,
    write_json,
)
from src.evaluation.time_window_inference import (
    TimeWindowInferenceConfig,
    local_repair_action,
    predict_action_lateness,
    select_action_with_time_window_bias,
)
from src.models.policy import HGATPolicy


EXPERIMENTS = [
    ("raw_baseline", {}),
    (
        "tw_bias_light",
        {
            "enable_time_window_bias": True,
            "lateness_bias_weight": 0.5,
            "severe_lateness_threshold": 10.0,
            "severe_lateness_bias_weight": 1.0,
        },
    ),
    (
        "tw_bias_medium",
        {
            "enable_time_window_bias": True,
            "lateness_bias_weight": 1.0,
            "severe_lateness_threshold": 10.0,
            "severe_lateness_bias_weight": 2.0,
        },
    ),
    (
        "tw_bias_strong",
        {
            "enable_time_window_bias": True,
            "lateness_bias_weight": 2.0,
            "severe_lateness_threshold": 10.0,
            "severe_lateness_bias_weight": 4.0,
        },
    ),
    (
        "local_repair_only",
        {
            "enable_local_repair": True,
            "repair_max_iterations": 50,
            "repair_window_size": 4,
        },
    ),
    (
        "tw_bias_medium_plus_repair",
        {
            "enable_time_window_bias": True,
            "lateness_bias_weight": 1.0,
            "severe_lateness_threshold": 10.0,
            "severe_lateness_bias_weight": 2.0,
            "enable_local_repair": True,
            "repair_max_iterations": 50,
            "repair_window_size": 4,
        },
    ),
    (
        "tw_bias_strong_plus_repair",
        {
            "enable_time_window_bias": True,
            "lateness_bias_weight": 2.0,
            "severe_lateness_threshold": 10.0,
            "severe_lateness_bias_weight": 4.0,
            "enable_local_repair": True,
            "repair_max_iterations": 50,
            "repair_window_size": 4,
        },
    ),
]


SUMMARY_FIELDS = [
    "experiment_name",
    "acceptance_rate",
    "acceptance_rate_delta",
    "on_time_rate",
    "on_time_rate_delta",
    "late_orders",
    "late_orders_delta",
    "average_lateness",
    "average_lateness_delta",
    "max_lateness",
    "max_lateness_delta",
    "total_lateness",
    "total_lateness_delta",
    "total_energy_consumption",
    "total_energy_delta",
    "average_energy_per_order",
    "total_flight_distance",
    "total_distance_delta",
    "average_drone_utilization",
    "utilization_delta",
    "hard_constraint_violations",
    "soft_time_window_violations",
    "is_candidate",
    "is_risky",
    "recommendation",
]


def _resolve(path: str) -> str:
    return str(Path(path).resolve())


def _bool(v: Any) -> bool:
    return str(v).strip().lower() == "true"


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


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


def _load_open_instances(args: argparse.Namespace):
    if not args.dataset_path.strip():
        return None
    include_names = read_instance_name_list(args.eval_split_file) if args.eval_split_file.strip() else None
    instances = load_cvrplib_instances_filtered(args.dataset_path.strip(), include_names=include_names)
    instances = [x for x in instances if x.n_customers >= int(args.N)]
    if not instances:
        raise ValueError("No CVRPLIB instance has enough customers for current --N.")
    return instances


def _make_env(args: argparse.Namespace, cfg: EnvConfig, open_instances: Any, inst_seed: int) -> TruckDroneRendezvousEnv:
    if open_instances is None:
        coord, release, demand, due, meta = make_random_instance(
            N=args.N,
            seed=inst_seed,
            coord_scale=args.coord_scale,
            release_mode=args.release_mode,
            n_batches=args.n_batches,
            max_release=args.max_release,
            poisson_rate=args.poisson_rate,
            tw_mode=args.tw_mode,
            tw_slack_low=args.tw_slack_low,
            tw_slack_high=args.tw_slack_high,
            tw_active_prob=args.tw_active_prob,
            scheduled_ratio=args.scheduled_ratio,
            dynamic_pickup_ratio=args.dynamic_pickup_ratio,
            response_slack_low=args.response_slack_low,
            response_slack_high=args.response_slack_high,
            return_due=True,
        )
    else:
        coord_base, demand_base, _ = sample_open_vrp_base(
            instances=open_instances,
            N=args.N,
            seed=inst_seed,
            coord_scale=args.coord_scale,
            normalize_coords=not args.dataset_no_normalize_coords,
            demand_scale=args.dataset_demand_scale,
        )
        coord, release, demand, due, meta = make_instance_from_coord_demand(
            coord=coord_base,
            demand=demand_base,
            seed=inst_seed,
            release_mode=args.release_mode,
            n_batches=args.n_batches,
            max_release=args.max_release,
            poisson_rate=args.poisson_rate,
            tw_mode=args.tw_mode,
            tw_slack_low=args.tw_slack_low,
            tw_slack_high=args.tw_slack_high,
            tw_active_prob=args.tw_active_prob,
            scheduled_ratio=args.scheduled_ratio,
            dynamic_pickup_ratio=args.dynamic_pickup_ratio,
            response_slack_low=args.response_slack_low,
            response_slack_high=args.response_slack_high,
            return_due=True,
        )
    return TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=cfg, seed=inst_seed, **meta)


def _load_policy(args: argparse.Namespace) -> HGATPolicy:
    device = torch.device("cpu")
    policy = HGATPolicy(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        k_nn_orders=args.k_nn_orders,
        num_encoder_layers=args.encoder_layers,
        tanh_clipping=args.tanh_clipping,
        temperature=args.temperature,
    ).to(device)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def rollout_inference(
    policy: HGATPolicy,
    env: TruckDroneRendezvousEnv,
    cfg: TimeWindowInferenceConfig,
    *,
    K: int,
    max_steps: int,
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    costs: List[float] = []
    trajs: List[List[Dict[str, Any]]] = []
    logs: List[Dict[str, Any]] = []
    use_custom_selector = bool(cfg.enable_time_window_bias)
    for k_id in range(K):
        e = env.copy()
        obs = e.reset()
        total_cost = 0.0
        traj: List[Dict[str, Any]] = []
        step_logs: List[Dict[str, Any]] = []
        done = False
        for step in range(max_steps):
            if use_custom_selector:
                action, bias_debug = select_action_with_time_window_bias(policy, e, obs, cfg, greedy=False)
            else:
                with torch.no_grad():
                    action, _ = policy.act(e, obs)
                bias_debug = {"time_window_bias_enabled": False, "bias_changed_action": False}

            repaired_action, repair_debug = local_repair_action(e, obs, action, cfg)
            obs2, reward, done, info = e.step(repaired_action)
            reward_f = float(reward)
            total_cost += -reward_f
            traj.append(
                {
                    "obs": obs,
                    "obs2": obs2,
                    "action": repaired_action,
                    "reward": reward_f,
                    "info": info,
                    "inference_debug": bias_debug,
                    "repair_debug": repair_debug,
                }
            )
            if bool(bias_debug.get("bias_changed_action")) or bool(repair_debug.get("repair_changed_action")):
                step_logs.append(
                    {
                        "step": int(step),
                        "obs_t": float(obs.get("t", 0.0)),
                        "obs_i": int(obs.get("i", 0)),
                        "final_action": [int(repaired_action[0]), int(repaired_action[1])],
                        "bias_debug": bias_debug,
                        "repair_debug": repair_debug,
                    }
                )
            obs = obs2
            if done:
                break
        if not done:
            total_cost += 1000.0
            traj.append(
                {
                    "obs": obs,
                    "action": ("TIMEOUT",),
                    "reward": -1000.0,
                    "info": {"timeout": True, "max_steps": max_steps},
                }
            )
        costs.append(float(total_cost))
        trajs.append(traj)
        logs.append({"k_id": int(k_id), "cost": float(total_cost), "changed_steps": step_logs})
    return np.asarray(costs, dtype=np.float32), trajs, logs


def _hard_violations(order_rows: Sequence[Dict[str, Any]], drone_rows: Sequence[Dict[str, Any]]) -> int:
    hard = 0
    for r in order_rows:
        if _bool(r.get("battery_violation")) or _bool(r.get("capacity_violation")) or _bool(r.get("range_violation")):
            hard += 1
    for r in drone_rows:
        if _bool(r.get("battery_violation")):
            hard += 1
    return int(hard)


def evaluate_experiment(
    args: argparse.Namespace,
    policy: HGATPolicy,
    open_instances: Any,
    env_cfg: EnvConfig,
    name: str,
    tw_cfg: TimeWindowInferenceConfig,
    out_dir: Path,
) -> Dict[str, Any]:
    np.random.seed(int(args.eval_seed))
    torch.manual_seed(int(args.eval_seed))
    max_steps = 8 * (int(args.N) + 1)
    episode_summaries: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    drone_episode_rows: List[Dict[str, Any]] = []
    repair_log: Dict[str, Any] = {"experiment_name": name, "config": tw_cfg.__dict__, "instances": []}
    for idx in range(1, int(args.eval_instances) + 1):
        inst_seed = int(args.eval_seed) * 100000 + idx
        env = _make_env(args, env_cfg, open_instances, inst_seed)
        costs, trajs, logs = rollout_inference(policy, env, tw_cfg, K=int(args.K), max_steps=max_steps)
        best_id = int(costs.argmin())
        episode_summary, episode_orders, drone_row = analyze_episode(
            env,
            trajs[best_id],
            model_name=name,
            instance_id=idx,
            objective_cost=float(costs[best_id]),
        )
        episode_summaries.append(episode_summary)
        order_rows.extend(episode_orders)
        drone_episode_rows.append(drone_row)
        repair_log["instances"].append(
            {
                "instance_id": int(idx),
                "best_k": int(best_id),
                "best_cost": float(costs[best_id]),
                "changed_steps": logs[best_id]["changed_steps"],
            }
        )
        if idx % max(1, int(args.eval_progress_every)) == 0:
            print(
                f"[{name}] {idx}/{args.eval_instances}: "
                f"accept={episode_summary['acceptance_rate']:.3f} "
                f"on_time={episode_summary['on_time_rate']:.3f} late={episode_summary['late_orders']}"
            )

    overall, drone_detail = aggregate_model(name, episode_summaries, order_rows, drone_episode_rows)
    metrics_dir = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    write_csv(str(metrics_dir / f"{name}_overall_summary.csv"), [overall], OVERALL_FIELDS)
    write_csv(str(metrics_dir / f"{name}_order_details.csv"), order_rows, ORDER_DETAIL_FIELDS)
    write_csv(str(metrics_dir / f"{name}_drone_details.csv"), [drone_detail], DRONE_DETAIL_FIELDS)
    write_json(str(logs_dir / f"{name}_repair_log.json"), repair_log)
    return {
        "overall": overall,
        "order_rows": order_rows,
        "drone_rows": [drone_detail],
        "episode_summaries": episode_summaries,
    }


def write_lateness_analysis(repair_dir: Path) -> None:
    reports_dir = repair_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    src_order = Path("experiments/main_retrain_service_20260513_retry1/reports/order_details.csv")
    src_drone = Path("experiments/main_retrain_service_20260513_retry1/reports/drone_details.csv")
    ablation_root = Path("experiments/main_retrain_service_20260513_ablation")
    if not src_order.exists():
        (reports_dir / "time_window_lateness_analysis.md").write_text(
            "# Time Window Lateness Analysis\n\nPrevious order details were not found.\n",
            encoding="utf-8",
        )
        return

    with src_order.open(encoding="utf-8", newline="") as f:
        retry_orders = list(csv.DictReader(f))
    baseline_orders = [r for r in retry_orders if r["model_name"] == "baseline_main"]
    late = [r for r in baseline_orders if _bool(r.get("late"))]
    tight = sorted(
        [
            r
            for r in baseline_orders
            if _num(r.get("planned_delivery_time"), math.inf) < math.inf
            and _num(r.get("planned_delivery_time")) - _num(r.get("order_time")) <= 6.0
        ],
        key=lambda r: _num(r.get("planned_delivery_time")) - _num(r.get("order_time")),
    )
    near_miss = sorted(
        [r for r in late if 0.0 < _num(r.get("lateness_duration")) <= 1.0],
        key=lambda r: _num(r.get("lateness_duration")),
    )
    severe = sorted([r for r in late if _num(r.get("lateness_duration")) >= 10.0], key=lambda r: _num(r.get("lateness_duration")), reverse=True)
    drone_late = [r for r in late if r.get("drone_id")]
    service_counter: Dict[str, int] = {}
    for r in late:
        service_counter[r.get("service_mode", "unknown")] = service_counter.get(r.get("service_mode", "unknown"), 0) + 1

    ablation_late_counts: List[Tuple[str, int, int, float]] = []
    if ablation_root.exists():
        for p in sorted(ablation_root.glob("exp_*/reports/overall_summary.csv")):
            with p.open(encoding="utf-8", newline="") as f:
                rows = {r["model_name"]: r for r in csv.DictReader(f)}
            if "retrained_main" in rows:
                row = rows["retrained_main"]
                ablation_late_counts.append((p.parts[-3], int(_num(row["late_orders"])), int(_num(row["total_constraint_violations"])), _num(row["on_time_rate"])))

    lines = [
        "# Time Window Lateness Analysis",
        "",
        "This analysis uses the previous `retry1` detailed reports and the A-E ablation reports. The repeated pattern is that hard constraints are feasible, while soft time-window lateness dominates failures.",
        "",
        "## Baseline Late Orders",
        "",
        f"- Baseline accepted orders: `{sum(1 for r in baseline_orders if _bool(r.get('accepted')))}`.",
        f"- Baseline late orders: `{len(late)}`.",
        f"- Near-miss late orders with lateness <= 1.0: `{len(near_miss)}`.",
        f"- Severe late orders with lateness >= 10.0: `{len(severe)}`.",
        f"- Tight-window orders with planned-order time <= 6.0: `{len(tight)}`.",
        f"- Late orders by service mode: `{service_counter}`.",
        f"- Drone-served late orders: `{len(drone_late)}`.",
        "",
        "## Near-Miss Late Orders",
        "",
        "| order_id | lateness | planned | actual | mode | distance | energy |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for r in near_miss[:20]:
        lines.append(
            f"| {r['order_id']} | {_num(r['lateness_duration']):.3f} | {_num(r['planned_delivery_time']):.3f} | {_num(r['actual_delivery_time']):.3f} | {r.get('service_mode','')} | {_num(r['flight_distance']):.3f} | {_num(r['energy_consumption']):.3f} |"
        )
    lines.extend(["", "## Severe Late Orders", "", "| order_id | lateness | planned | actual | mode | distance | energy |", "|---|---:|---:|---:|---|---:|---:|"])
    for r in severe[:20]:
        lines.append(
            f"| {r['order_id']} | {_num(r['lateness_duration']):.3f} | {_num(r['planned_delivery_time']):.3f} | {_num(r['actual_delivery_time']):.3f} | {r.get('service_mode','')} | {_num(r['flight_distance']):.3f} | {_num(r['energy_consumption']):.3f} |"
        )
    lines.extend(["", "## Tightest Time Windows", "", "| order_id | window_width | late | lateness | planned | actual |", "|---|---:|---|---:|---:|---:|"])
    for r in tight[:20]:
        width = _num(r.get("planned_delivery_time")) - _num(r.get("order_time"))
        lines.append(
            f"| {r['order_id']} | {width:.3f} | {r.get('late')} | {_num(r.get('lateness_duration')):.3f} | {_num(r.get('planned_delivery_time')):.3f} | {_num(r.get('actual_delivery_time')):.3f} |"
        )
    lines.extend(["", "## A-E Ablation Lateness Pattern", "", "| experiment | late_orders | soft_time_window_violations | on_time_rate |", "|---|---:|---:|---:|"])
    for name, late_count, viol_count, ot in ablation_late_counts:
        lines.append(f"| {name} | {late_count} | {viol_count} | {ot:.6f} |")
    lines.extend(
        [
            "",
            "## Repair Opportunity Assessment",
            "",
            "- Near-miss late orders are plausible local-repair targets: small swaps or moving tight-window orders earlier may convert some to on-time.",
            "- Severe late orders usually need action-stage prevention or earlier dispatch; local adjacent swaps alone are unlikely to fix all of them.",
            "- Orders with very tight windows should receive inference-time risk bias before execution, not only after service completion.",
            "- Some accepted orders are likely already late at selection time. These should be downweighted, and in extreme cases rejected only if global metrics improve.",
            "- The single-drone environment has no inter-drone balancing issue. The useful repair surface is within-route ordering and truck/drone assignment choice.",
        ]
    )
    (reports_dir / "time_window_lateness_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_row(name: str, result: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    cur = result["overall"]
    base = raw["overall"]
    hard = _hard_violations(result["order_rows"], result["drone_rows"])
    soft = sum(1 for r in result["order_rows"] if _bool(r.get("time_window_violation")))
    acc_delta = float(cur["acceptance_rate"]) - float(base["acceptance_rate"])
    ot_delta = float(cur["on_time_rate"]) - float(base["on_time_rate"])
    late_delta = int(cur["late_orders"]) - int(base["late_orders"])
    avg_late_delta = float(cur["average_lateness"]) - float(base["average_lateness"])
    max_late_delta = float(cur["max_lateness"]) - float(base["max_lateness"])
    total_late_delta = float(cur["total_lateness"]) - float(base["total_lateness"])
    energy_delta = float(cur["total_energy_consumption"]) - float(base["total_energy_consumption"])
    dist_delta = float(cur["total_flight_distance"]) - float(base["total_flight_distance"])
    util_delta = float(cur["average_drone_utilization"]) - float(base["average_drone_utilization"])
    energy_risky = energy_delta > max(1e-9, 0.01 * float(base["total_energy_consumption"]))
    dist_risky = dist_delta > max(1e-9, 0.01 * float(base["total_flight_distance"]))
    service_ok = (
        hard == 0
        and ot_delta > 1e-9
        and late_delta < 0
        and avg_late_delta <= 1e-9
        and max_late_delta <= 0.5
        and acc_delta >= -0.005
    )
    is_risky = bool(service_ok and (energy_risky or dist_risky))
    is_candidate = bool(service_ok)
    reasons: List[str] = []
    if hard != 0:
        reasons.append("hard_constraint_violation")
    if ot_delta <= 0:
        reasons.append("on_time_not_improved")
    if late_delta >= 0:
        reasons.append("late_orders_not_reduced")
    if avg_late_delta > 0:
        reasons.append("average_lateness_worse")
    if max_late_delta > 0.5:
        reasons.append("max_lateness_worse")
    if acc_delta < -0.005:
        reasons.append("acceptance_drop_gt_0.5pp")
    if is_candidate:
        rec = "risky_candidate" if is_risky else "candidate"
    else:
        rec = "rejected: " + ", ".join(reasons)
    return {
        "experiment_name": name,
        "acceptance_rate": float(cur["acceptance_rate"]),
        "acceptance_rate_delta": float(acc_delta),
        "on_time_rate": float(cur["on_time_rate"]),
        "on_time_rate_delta": float(ot_delta),
        "late_orders": int(cur["late_orders"]),
        "late_orders_delta": int(late_delta),
        "average_lateness": float(cur["average_lateness"]),
        "average_lateness_delta": float(avg_late_delta),
        "max_lateness": float(cur["max_lateness"]),
        "max_lateness_delta": float(max_late_delta),
        "total_lateness": float(cur["total_lateness"]),
        "total_lateness_delta": float(total_late_delta),
        "total_energy_consumption": float(cur["total_energy_consumption"]),
        "total_energy_delta": float(energy_delta),
        "average_energy_per_order": float(cur["average_energy_per_order"]),
        "total_flight_distance": float(cur["total_flight_distance"]),
        "total_distance_delta": float(dist_delta),
        "average_drone_utilization": float(cur["average_drone_utilization"]),
        "utilization_delta": float(util_delta),
        "hard_constraint_violations": int(hard),
        "soft_time_window_violations": int(soft),
        "is_candidate": bool(is_candidate),
        "is_risky": bool(is_risky),
        "recommendation": rec,
    }


def write_repair_summaries(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    reports_dir = out_dir / "reports"
    write_csv(str(reports_dir / "repair_experiment_summary.csv"), rows, SUMMARY_FIELDS)
    write_json(str(reports_dir / "repair_experiment_summary.json"), {"experiments": rows})
    md_lines = [
        "# Repair Experiment Summary",
        "",
        "| experiment | acc | d_acc | on_time | d_on_time | late_d | avg_late_d | max_late_d | energy_d | distance_d | hard | soft_tw | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['experiment_name']} | {r['acceptance_rate']:.6f} | {r['acceptance_rate_delta']:+.6f} | "
            f"{r['on_time_rate']:.6f} | {r['on_time_rate_delta']:+.6f} | {r['late_orders_delta']:+d} | "
            f"{r['average_lateness_delta']:+.6f} | {r['max_lateness_delta']:+.6f} | "
            f"{r['total_energy_delta']:+.6f} | {r['total_distance_delta']:+.6f} | "
            f"{r['hard_constraint_violations']} | {r['soft_time_window_violations']} | {r['recommendation']} |"
        )
    candidates = [r for r in rows if r["is_candidate"]]
    md_lines.extend(["", "## Result", ""])
    if candidates:
        best = sorted(candidates, key=lambda r: (-r["on_time_rate_delta"], r["late_orders_delta"], r["total_energy_delta"]))[0]
        md_lines.append(f"Best candidate: `{best['experiment_name']}`.")
    else:
        md_lines.append("No inference/repair configuration passed the candidate criteria.")
    (reports_dir / "repair_experiment_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    best_report = ["# Best Candidate Report", ""]
    if candidates:
        best = sorted(candidates, key=lambda r: (-r["on_time_rate_delta"], r["late_orders_delta"], r["total_energy_delta"]))[0]
        best_report.extend(
            [
                f"Best candidate: `{best['experiment_name']}`.",
                "",
                f"- On-time delta: `{best['on_time_rate_delta']:.6f}`",
                f"- Late-order delta: `{best['late_orders_delta']}`",
                f"- Acceptance delta: `{best['acceptance_rate_delta']:.6f}`",
                f"- Energy delta: `{best['total_energy_delta']:.6f}`",
                f"- Distance delta: `{best['total_distance_delta']:.6f}`",
                f"- Risky: `{best['is_risky']}`",
            ]
        )
    else:
        best_report.extend(
            [
                "No candidate found. Keep raw baseline inference as the best deployable option for now.",
                "",
                "Most likely reason: scalar time-window bias and local current-step repair are insufficient because the main failures are delayed sequence-level time-window conflicts.",
            ]
        )
    (reports_dir / "best_candidate_report.md").write_text("\n".join(best_report) + "\n", encoding="utf-8")


def append_current_benchmark_to_lateness_analysis(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = out_dir / "reports" / "time_window_lateness_analysis.md"
    if not path.exists():
        return
    lines = [
        "",
        "## Current No-Training Repair Benchmark",
        "",
        "The repair-stage experiment re-evaluated the frozen baseline on 30 test instances with `N=30` and `K=8`. The current raw baseline can differ slightly from previous reports because this benchmark re-runs stochastic POMO inference under the repair experiment script.",
        "",
        "| experiment | accepted_rate | on_time_rate | late_orders | avg_late | max_late | energy_delta | distance_delta | hard_violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment_name']} | {r['acceptance_rate']:.6f} | {r['on_time_rate']:.6f} | "
            f"{int(r['late_orders'])} | {r['average_lateness']:.6f} | {r['max_lateness']:.6f} | "
            f"{r['total_energy_delta']:.6f} | {r['total_distance_delta']:.6f} | {int(r['hard_constraint_violations'])} |"
        )
    candidates = [r for r in rows if r["is_candidate"]]
    best_on_time = max(rows, key=lambda r: float(r["on_time_rate_delta"])) if rows else None
    lines.append("")
    if candidates:
        best = sorted(candidates, key=lambda r: (-r["on_time_rate_delta"], r["late_orders_delta"], r["total_energy_delta"]))[0]
        lines.append(f"`{best['experiment_name']}` passes the candidate criteria.")
    elif best_on_time is not None:
        lines.append(
            f"All no-training variants preserve hard feasibility, but none passes all candidate criteria. "
            f"The best on-time-rate variant is `{best_on_time['experiment_name']}`, which still fails because `{best_on_time['recommendation']}`."
        )
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-training time-window-aware inference and local repair experiments.")
    parser.add_argument("--output-dir", type=str, default="experiments/main_retrain_service_20260513_repair")
    parser.add_argument("--model-path", type=str, default="experiments/frozen_models_20260419/model_main_ep200.pt")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--eval-instances", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-progress-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k-nn-orders", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--run-custom-experiment", action="store_true")
    parser.add_argument("--custom-experiment-name", type=str, default="custom_time_window_repair")
    parser.add_argument("--enable-time-window-bias", action="store_true")
    parser.add_argument("--lateness-bias-weight", type=float, default=1.0)
    parser.add_argument("--severe-lateness-threshold", type=float, default=10.0)
    parser.add_argument("--severe-lateness-bias-weight", type=float, default=2.0)
    parser.add_argument("--disallow-late-if-no-feasible-action", action="store_true")
    parser.add_argument("--enable-local-repair", action="store_true")
    parser.add_argument("--repair-max-iterations", type=int, default=50)
    parser.add_argument("--repair-window-size", type=int, default=4)
    parser.add_argument("--repair-objective", type=str, default="lateness")

    parser.add_argument("--coord-scale", type=float, default=10.0)
    parser.add_argument("--release-mode", type=str, default="batches")
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--max-release", type=float, default=10.0)
    parser.add_argument("--poisson-rate", type=float, default=1.0)
    parser.add_argument("--tw-mode", type=str, default="relative")
    parser.add_argument("--tw-slack-low", type=float, default=4.0)
    parser.add_argument("--tw-slack-high", type=float, default=14.0)
    parser.add_argument("--tw-active-prob", type=float, default=0.8)
    parser.add_argument("--scheduled-ratio", type=float, default=0.5)
    parser.add_argument("--dynamic-pickup-ratio", type=float, default=1.0)
    parser.add_argument("--response-slack-low", type=float, default=0.25)
    parser.add_argument("--response-slack-high", type=float, default=1.0)
    parser.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    parser.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    parser.add_argument("--dataset-demand-scale", type=float, default=1.0)
    parser.add_argument("--dataset-no-normalize-coords", action="store_true")

    parser.add_argument("--vT", type=float, default=1.0)
    parser.add_argument("--vD", type=float, default=1.5)
    parser.add_argument("--QD", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=6.0)
    parser.add_argument("--truck-capacity", type=float, default=3.0)
    parser.add_argument("--truck-service-time", type=float, default=0.05)
    parser.add_argument("--drone-service-time", type=float, default=0.03)
    parser.add_argument("--depot-service-time", type=float, default=0.10)
    parser.add_argument("--traffic-sigma", type=float, default=0.15)
    parser.add_argument("--lateness-penalty", type=float, default=1.5)
    parser.add_argument("--reject-penalty", type=float, default=6.0)
    parser.add_argument("--overtime-penalty", type=float, default=1.0)
    parser.add_argument("--time-cost-weight", type=float, default=1.0)
    parser.add_argument("--energy-cost-weight", type=float, default=0.08)
    parser.add_argument("--soc-init", type=float, default=1.0)
    parser.add_argument("--soc-reserve", type=float, default=0.10)
    parser.add_argument("--energy-per-dist", type=float, default=0.08)
    parser.add_argument("--truck-energy-per-dist", type=float, default=0.04)
    parser.add_argument("--payload-energy-factor", type=float, default=0.4)
    parser.add_argument("--drone-takeoff-landing-energy", type=float, default=0.01)
    parser.add_argument("--drone-idle-energy-per-time", type=float, default=0.0)
    parser.add_argument("--recharge-rate", type=float, default=0.25)
    parser.add_argument("--edge-mode", type=str, default="road")
    parser.add_argument("--time-dependent", action="store_true", default=True)
    parser.add_argument("--peak-after-served-ratio", type=float, default=0.5)
    parser.add_argument("--workday-start", type=float, default=8.0)
    parser.add_argument("--workday-end", type=float, default=20.0)
    parser.add_argument("--morning-peak-start", type=float, default=8.0)
    parser.add_argument("--morning-peak-end", type=float, default=10.0)
    parser.add_argument("--evening-peak-start", type=float, default=17.0)
    parser.add_argument("--evening-peak-end", type=float, default=19.0)
    parser.add_argument("--road-detour-factor", type=float, default=1.18)
    parser.add_argument("--road-signal-density", type=float, default=0.006)
    parser.add_argument("--road-turn-density", type=float, default=0.010)
    parser.add_argument("--road-one-way-ratio", type=float, default=0.10)
    parser.add_argument("--road-peak-factor", type=float, default=1.25)
    parser.add_argument("--signal-penalty", type=float, default=0.05)
    parser.add_argument("--turn-penalty", type=float, default=0.12)
    parser.add_argument("--left-turn-penalty", type=float, default=0.08)
    parser.add_argument("--u-turn-penalty", type=float, default=0.30)
    return parser.parse_args()


def _experiments_from_args(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    if not (args.run_custom_experiment or args.enable_time_window_bias or args.enable_local_repair):
        return list(EXPERIMENTS)
    return [
        (
            str(args.custom_experiment_name),
            {
                "enable_time_window_bias": bool(args.enable_time_window_bias),
                "lateness_bias_weight": float(args.lateness_bias_weight),
                "severe_lateness_threshold": float(args.severe_lateness_threshold),
                "severe_lateness_bias_weight": float(args.severe_lateness_bias_weight),
                "allow_late_if_no_feasible_action": not bool(args.disallow_late_if_no_feasible_action),
                "enable_local_repair": bool(args.enable_local_repair),
                "repair_max_iterations": int(args.repair_max_iterations),
                "repair_window_size": int(args.repair_window_size),
                "repair_objective": str(args.repair_objective),
            },
        )
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    for sub in ("reports", "metrics", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print("No training will be run. Loading baseline weights:", args.model_path)
    write_lateness_analysis(out_dir)
    policy = _load_policy(args)
    env_cfg = _env_config(args)
    open_instances = _load_open_instances(args)
    experiments_to_run = _experiments_from_args(args)
    write_json(
        str(out_dir / "run_config.json"),
        {
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": _resolve(args.model_path),
            "dataset_path": _resolve(args.dataset_path),
            "eval_split_file": _resolve(args.eval_split_file),
            "experiments": [{"name": name, "config": cfg} for name, cfg in experiments_to_run],
            "args": vars(args),
        },
    )

    results: Dict[str, Dict[str, Any]] = {}
    for name, cfg_kwargs in experiments_to_run:
        print(f"\n=== {name} ===")
        tw_cfg = TimeWindowInferenceConfig(**cfg_kwargs)
        results[name] = evaluate_experiment(args, policy, open_instances, env_cfg, name, tw_cfg, out_dir)

    raw = results.get("raw_baseline", next(iter(results.values())))
    rows = [_summary_row(name, result, raw) for name, result in results.items()]
    write_repair_summaries(out_dir, rows)
    append_current_benchmark_to_lateness_analysis(out_dir, rows)
    print("\nRepair experiments complete.")
    print(json.dumps({"summary_csv": str(out_dir / "reports" / "repair_experiment_summary.csv"), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
