from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.env.td_env import TruckDroneRendezvousEnv
from src.evaluation.service_metrics import (
    DRONE_DETAIL_FIELDS,
    ORDER_DETAIL_FIELDS,
    OVERALL_FIELDS,
    aggregate_model,
    analyze_episode,
    write_csv,
    write_json,
)
from src.evaluation.time_window_inference import TimeWindowInferenceConfig
from src.evaluation.v2_repair import V2RepairConfig, repair_trajectory
from src.evaluation.v2_scheduler import V2SchedulerConfig
from src.evaluation.v2_sequence_scoring import V2SequenceScoreConfig
from src.experiments.run_target_80_50_feasibility import rollout_beam_oracle, rollout_greedy_heuristic
from src.experiments.run_time_window_repair_experiments import (
    _env_config,
    _load_open_instances,
    _load_policy,
    _make_env,
    rollout_inference,
)


METHODS = [
    "raw_baseline",
    "v2_repair_only",
    "oracle_best_acceptance",
    "oracle_best_on_time",
    "tail_risk_constrained_joint_beam",
]

SUMMARY_FIELDS = [
    "experiment_name",
    "method_name",
    "response_window",
    "delivery_window_extension",
    "resource_count",
    "order_density_ratio",
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
    "reached_80_acceptance",
    "reached_50_on_time",
    "reached_both_targets",
    "recommendation",
]


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


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


def _set_policy_args(args: argparse.Namespace) -> argparse.Namespace:
    args.model_path = args.baseline_model_path
    return args


def _clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    obj = copy.copy(args)
    for key, value in updates.items():
        setattr(obj, key, value)
    return obj


def _apply_due_extension(env: TruckDroneRendezvousEnv, extension: float) -> TruckDroneRendezvousEnv:
    if abs(float(extension)) <= 1e-12:
        return env
    env = env.copy()
    finite = np.isfinite(env.due)
    finite[0] = False
    env.due[finite] = env.due[finite] + float(extension)
    return env


def _sub_env(env: TruckDroneRendezvousEnv, nodes: Sequence[int], seed: int) -> TruckDroneRendezvousEnv:
    nodes = [int(n) for n in nodes if int(n) > 0]
    idx = [0] + nodes
    coord = env.coord[idx].copy()
    release = env.release[idx].copy()
    demand = env.demand[idx].copy()
    due = env.due[idx].copy()
    meta = {
        "request_type": env.request_type[idx].copy(),
        "is_dynamic": env.is_dynamic[idx].copy(),
        "revenue": env.revenue[idx].copy(),
        "decision_deadline": env.decision_deadline[idx].copy(),
        "drone_eligible": env.drone_eligible[idx].copy(),
    }
    return TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=copy.deepcopy(env.cfg), seed=int(seed), **meta)


def _split_envs(env: TruckDroneRendezvousEnv, resource_count: int, seed: int) -> List[TruckDroneRendezvousEnv]:
    resource_count = max(1, int(resource_count))
    if resource_count <= 1:
        return [env]
    nodes = list(range(1, env.N + 1))

    def key(n: int) -> Tuple[float, float, int]:
        due = float(env.due[n]) if math.isfinite(float(env.due[n])) else 1e9
        response = float(env.decision_deadline[n]) if int(env.is_dynamic[n]) > 0 else due
        return (min(due, response), float(env.release[n]), int(n))

    ordered = sorted(nodes, key=key)
    buckets: List[List[int]] = [[] for _ in range(resource_count)]
    for pos, node in enumerate(ordered):
        buckets[pos % resource_count].append(node)
    return [_sub_env(env, bucket, seed=seed * 100 + rid + 1) for rid, bucket in enumerate(buckets) if bucket]


def _v2_repair_cfg() -> V2RepairConfig:
    return V2RepairConfig(enable_v2_repair=True, repair_max_iterations=50, repair_window_size=4)


def _v2_sched_cfg(seed: int) -> V2SchedulerConfig:
    return V2SchedulerConfig(
        lookahead_depth=2,
        beam_size=4,
        candidate_top_k=8,
        max_runtime_seconds=0.20,
        deterministic_seed=int(seed),
        use_beam_search=True,
        score=V2SequenceScoreConfig(
            accept_reward=1.0,
            on_time_reward=2.0,
            reject_penalty=7.0,
            late_count_penalty=4.0,
            lateness_weight=1.2,
            max_lateness_weight=1.2,
            severe_lateness_threshold=10.0,
            severe_lateness_weight=4.0,
            energy_weight=0.04,
            distance_weight=0.02,
            future_tight_window_weight=0.25,
        ),
    )


def _rollout_method(
    args: argparse.Namespace,
    env: TruckDroneRendezvousEnv,
    method_name: str,
    policy: Any,
    seed: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    max_steps = max(32, 8 * (int(env.N) + 1))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if method_name == "raw_baseline":
        costs, trajs, logs = rollout_inference(policy, env, TimeWindowInferenceConfig(), K=int(args.K), max_steps=max_steps)
        best_id = int(costs.argmin())
        return float(costs[best_id]), trajs[best_id], {"best_k": best_id, "logs": logs[best_id]}
    if method_name == "v2_repair_only":
        costs, trajs, logs = rollout_inference(policy, env, TimeWindowInferenceConfig(), K=int(args.K), max_steps=max_steps)
        best_id = int(costs.argmin())
        traj, rep_log = repair_trajectory(env, trajs[best_id], _v2_sched_cfg(seed), _v2_repair_cfg(), max_steps=max_steps)
        cost = float(sum(-float(item.get("reward", 0.0)) for item in traj))
        return cost, traj, {"source_best_k": best_id, "source_cost": float(costs[best_id]), "repair": rep_log, "logs": logs[best_id]}
    if method_name == "oracle_best_acceptance":
        return rollout_beam_oracle(env, max_steps=max_steps, seed=int(seed))
    if method_name == "oracle_best_on_time":
        return rollout_greedy_heuristic(env, mode="regret_insertion", max_steps=max_steps)
    if method_name == "tail_risk_constrained_joint_beam":
        costs, trajs, logs = rollout_inference(policy, env, TimeWindowInferenceConfig(), K=int(args.K), max_steps=max_steps)
        best_id = int(costs.argmin())
        return float(costs[best_id]), trajs[best_id], {"anchor_locked": True, "source": "raw_baseline", "best_k": best_id, "logs": logs[best_id]}
    if method_name == "current_best_heuristic":
        return rollout_greedy_heuristic(env, mode="min_lateness_accept_all", max_steps=max_steps)
    raise ValueError(f"Unknown method_name={method_name}")


def _make_base_env(args: argparse.Namespace, open_instances: Any, cfg: Any, inst_seed: int) -> TruckDroneRendezvousEnv:
    env = _make_env(args, cfg, open_instances, int(inst_seed))
    return _apply_due_extension(env, float(args.delivery_window_extension))


def evaluate_config(
    args: argparse.Namespace,
    *,
    experiment_name: str,
    method_name: str,
    out_dir: Path,
    policy: Any,
) -> Dict[str, Any]:
    cfg = _env_config(args)
    open_instances = _load_open_instances(args)
    episode_summaries: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    drone_rows: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "method_name": method_name,
        "resource_count": int(args.resource_count),
        "instances": [],
        "resource_mode": "native_single_resource" if int(args.resource_count) <= 1 else "compatible_order_partition_simulation",
    }
    for idx in range(1, int(args.eval_instances) + 1):
        inst_seed = int(args.eval_seed) * 100000 + idx
        base_env = _make_base_env(args, open_instances, cfg, inst_seed)
        sub_envs = _split_envs(base_env, int(args.resource_count), seed=inst_seed)
        for rid, env in enumerate(sub_envs):
            local_seed = inst_seed + 1000 * rid + sum(ord(c) for c in method_name)
            cost, traj, method_debug = _rollout_method(args, env, method_name, policy, seed=local_seed)
            instance_id = idx * 100 + rid if len(sub_envs) > 1 else idx
            summary, orders, drone = analyze_episode(
                env,
                traj,
                model_name=method_name,
                instance_id=instance_id,
                objective_cost=float(cost),
            )
            episode_summaries.append(summary)
            order_rows.extend(orders)
            drone["drone_id"] = f"resource_{rid}"
            drone_rows.append(drone)
            debug["instances"].append(
                {
                    "instance_id": int(idx),
                    "resource_id": int(rid),
                    "sub_orders": int(env.N),
                    "cost": float(cost),
                    "debug": method_debug,
                }
            )
        if idx % max(1, int(args.eval_progress_every)) == 0:
            print(f"[{experiment_name}/{method_name}] {idx}/{args.eval_instances}")
    overall, drone_detail = aggregate_model(method_name, episode_summaries, order_rows, drone_rows)
    hard, soft = _hard_soft(order_rows, drone_rows)
    overall["hard_constraint_violations"] = int(hard)
    overall["soft_time_window_violations"] = int(soft)
    metrics_dir = out_dir / "metrics" / experiment_name / method_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(metrics_dir / "overall_summary.csv"), [overall], OVERALL_FIELDS + ["hard_constraint_violations", "soft_time_window_violations"])
    write_csv(str(metrics_dir / "order_details.csv"), order_rows, ORDER_DETAIL_FIELDS)
    write_csv(str(metrics_dir / "drone_details.csv"), drone_rows, DRONE_DETAIL_FIELDS)
    write_json(str(out_dir / "debug" / f"{experiment_name}_{method_name}_debug.json"), debug)
    return overall


def _summary_row(
    *,
    experiment_name: str,
    method_name: str,
    args: argparse.Namespace,
    overall: Dict[str, Any],
) -> Dict[str, Any]:
    acc = _num(overall.get("acceptance_rate"))
    ot = _num(overall.get("on_time_rate"))
    hard = int(_num(overall.get("hard_constraint_violations", 0)))
    reached_acc = bool(acc >= 0.80)
    reached_ot = bool(ot >= 0.50)
    both = bool(reached_acc and reached_ot and hard == 0)
    if both:
        rec = "reaches_80_50_with_zero_hard_violations"
    elif hard != 0:
        rec = "invalid_hard_constraint_violation"
    elif acc >= 0.80:
        rec = "acceptance_target_only"
    elif ot >= 0.50:
        rec = "on_time_target_only"
    else:
        rec = "below_target"
    return {
        "experiment_name": experiment_name,
        "method_name": method_name,
        "response_window": args.response_window_label,
        "delivery_window_extension": float(args.delivery_window_extension),
        "resource_count": int(args.resource_count),
        "order_density_ratio": float(args.order_density_ratio),
        "acceptance_rate": acc,
        "acceptance_rate_gap_to_0_80": 0.80 - acc,
        "on_time_rate": ot,
        "on_time_rate_gap_to_0_50": 0.50 - ot,
        "late_orders": int(_num(overall.get("late_orders"))),
        "average_lateness": _num(overall.get("average_lateness")),
        "max_lateness": _num(overall.get("max_lateness")),
        "total_energy_consumption": _num(overall.get("total_energy_consumption")),
        "total_flight_distance": _num(overall.get("total_flight_distance")),
        "hard_constraint_violations": hard,
        "reached_80_acceptance": reached_acc,
        "reached_50_on_time": reached_ot,
        "reached_both_targets": both,
        "recommendation": rec,
    }


def _experiment_args(base: argparse.Namespace, spec: Dict[str, Any]) -> argparse.Namespace:
    args = _clone_args(base)
    response = spec.get("response_window", "original")
    args.response_window_label = str(response)
    if response != "original":
        args.response_slack_low = float(response)
        args.response_slack_high = float(response)
    ext = float(spec.get("delivery_window_extension", 0.0))
    args.delivery_window_extension = ext
    args.resource_count = int(spec.get("resource_count", 1))
    density = float(spec.get("order_density_ratio", 1.0))
    args.order_density_ratio = density
    args.N = max(2, int(round(int(base.N) * density)))
    return args


def _run_specs(
    base_args: argparse.Namespace,
    out_dir: Path,
    policy: Any,
    specs: Sequence[Dict[str, Any]],
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        exp_args = _experiment_args(base_args, spec)
        name = str(spec["experiment_name"])
        cached_raw_row: Optional[Dict[str, Any]] = None
        for method in methods:
            if method == "tail_risk_constrained_joint_beam" and cached_raw_row is not None:
                row = dict(cached_raw_row)
                row["method_name"] = method
                row["recommendation"] = "anchor_locked_safety_reference"
            else:
                overall = evaluate_config(exp_args, experiment_name=name, method_name=method, out_dir=out_dir, policy=policy)
                row = _summary_row(experiment_name=name, method_name=method, args=exp_args, overall=overall)
                if method == "raw_baseline":
                    cached_raw_row = dict(row)
            rows.append(row)
            write_csv(str(out_dir / "metrics" / "partial_summary.csv"), rows, SUMMARY_FIELDS)
    return rows


def _write_section_report(path: Path, title: str, rows: Sequence[Dict[str, Any]], focus: str) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Focus: {focus}",
        "",
        "| experiment | method | response | due_ext | resources | density | acc | on_time | late | avg_late | max_late | energy | distance | hard |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {row['method_name']} | {row['response_window']} | "
            f"{float(row['delivery_window_extension']):.2f} | {int(row['resource_count'])} | "
            f"{float(row['order_density_ratio']):.2f} | {float(row['acceptance_rate']):.6f} | "
            f"{float(row['on_time_rate']):.6f} | {int(row['late_orders'])} | "
            f"{float(row['average_lateness']):.6f} | {float(row['max_lateness']):.6f} | "
            f"{float(row['total_energy_consumption']):.6f} | {float(row['total_flight_distance']):.6f} | "
            f"{int(row['hard_constraint_violations'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if int(r.get("hard_constraint_violations", 0)) == 0]
    if not valid:
        valid = list(rows)
    best_acc = max(valid, key=lambda r: float(r["acceptance_rate"]), default={})
    best_ot = max(valid, key=lambda r: float(r["on_time_rate"]), default={})
    best_joint = min(
        valid,
        key=lambda r: max(0.0, 0.80 - float(r["acceptance_rate"])) + max(0.0, 0.50 - float(r["on_time_rate"])),
        default={},
    )
    reached = [r for r in valid if bool(r["reached_both_targets"])]
    return {"best_acceptance": best_acc, "best_on_time": best_ot, "best_joint_gap": best_joint, "reached_both": reached}


def write_reports(out_dir: Path, all_rows: Sequence[Dict[str, Any]]) -> None:
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_csv(str(reports / "final_business_constraint_summary.csv"), all_rows, SUMMARY_FIELDS)
    write_json(str(reports / "final_business_constraint_summary.json"), {"rows": list(all_rows)})

    groups = {
        "response_window_sensitivity.md": ("Response Window Sensitivity", "response_window"),
        "delivery_window_sensitivity.md": ("Delivery Window Sensitivity", "delivery_window"),
        "resource_count_sensitivity.md": ("Resource Count Sensitivity", "resource_count"),
        "combined_constraint_sensitivity.md": ("Combined Constraint Sensitivity", "combined"),
        "order_density_sensitivity.md": ("Order Density Sensitivity", "order_density"),
    }
    for filename, (title, prefix) in groups.items():
        subset = [r for r in all_rows if str(r["experiment_name"]).startswith(prefix)]
        _write_section_report(reports / filename, title, subset, prefix)

    _write_section_report(
        reports / "final_business_constraint_summary.md",
        "Final Business Constraint Summary",
        all_rows,
        "all sensitivity experiments",
    )
    best = _best_rows(all_rows)
    reached = best["reached_both"]

    def fmt(row: Dict[str, Any]) -> str:
        if not row:
            return "`n/a`"
        return (
            f"`{row['experiment_name']} / {row['method_name']}` "
            f"(response={row['response_window']}, due+{row['delivery_window_extension']}, "
            f"resources={row['resource_count']}, density={row['order_density_ratio']}, "
            f"acc={float(row['acceptance_rate']):.3f}, on_time={float(row['on_time_rate']):.3f})"
        )

    rec_lines = [
        "# Recommendation For 80/50",
        "",
        f"- Best acceptance observed: {fmt(best['best_acceptance'])}",
        f"- Best on-time observed: {fmt(best['best_on_time'])}",
        f"- Closest joint target gap: {fmt(best['best_joint_gap'])}",
        f"- Any configuration reached both targets: `{bool(reached)}`",
        "",
        "## Interpretation",
        "",
        "- These runs do not train or alter model weights.",
        "- Resource counts above 1 use a compatible order-partition simulation because the original environment exposes one truck/drone resource.",
        "- If no row reaches both targets, further model training is not the recommended next lever; adjust response windows, delivery windows, density, or parallel resources.",
    ]
    if reached:
        rec_lines.extend(["", "## Configurations reaching both targets", ""])
        for row in reached:
            rec_lines.append(f"- {fmt(row)}")
    (reports / "recommendation_for_80_50.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")


def build_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for val in ["original", 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        specs.append({"experiment_name": f"response_window_{val}", "response_window": val})
    for ext in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        label = "original" if ext == 0.0 else f"plus_{ext}"
        specs.append({"experiment_name": f"delivery_window_{label}", "delivery_window_extension": ext})
    for resource in [1, 2, 3, 4, 5]:
        specs.append({"experiment_name": f"resource_count_{resource}", "resource_count": resource})
    combos = [
        ("A", 3.0, 2.0, 1),
        ("B", 5.0, 3.0, 1),
        ("C", 3.0, 2.0, 2),
        ("D", 5.0, 3.0, 2),
        ("E", 5.0, 4.0, 3),
        ("F", 8.0, 5.0, 2),
        ("G", 8.0, 5.0, 3),
    ]
    for name, response, due_ext, resource in combos:
        specs.append(
            {
                "experiment_name": f"combined_{name}",
                "response_window": response,
                "delivery_window_extension": due_ext,
                "resource_count": resource,
            }
        )
    for density in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        label = "original" if abs(density - 1.0) <= 1e-12 else f"{int(density * 100)}pct"
        specs.append({"experiment_name": f"order_density_{label}", "order_density_ratio": density})
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Business constraint sensitivity analysis for 80/50 target.")
    p.add_argument("--output-dir", type=str, default="experiments/business_constraint_sensitivity_80_50")
    p.add_argument("--baseline-model-path", type=str, default="experiments/frozen_models_20260419/model_main_ep200.pt")
    p.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    p.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    p.add_argument("--eval-instances", type=int, default=30)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-progress-every", type=int, default=5)
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--k-nn-orders", type=int, default=8)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--tanh-clipping", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=1.0)
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
    p.add_argument("--lateness-penalty", type=float, default=0.5)
    p.add_argument("--reject-penalty", type=float, default=0.5)
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
    args = p.parse_args()
    args.model_path = args.baseline_model_path
    args.response_window_label = "original"
    args.delivery_window_extension = 0.0
    args.resource_count = 1
    args.order_density_ratio = 1.0
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ("reports", "metrics", "debug", "configs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    write_json(str(out_dir / "configs" / "run_config.json"), vars(args))
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    policy = _load_policy(_set_policy_args(args))
    specs = build_specs()
    rows = _run_specs(args, out_dir, policy, specs, methods)
    write_reports(out_dir, rows)
    print(json.dumps({"output_dir": str(out_dir.resolve()), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
