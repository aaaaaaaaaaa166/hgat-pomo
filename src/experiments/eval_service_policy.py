from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.service_metrics import (
    DRONE_DETAIL_FIELDS,
    ORDER_DETAIL_FIELDS,
    OVERALL_FIELDS,
    aggregate_model,
    analyze_episode,
    write_csv,
    write_json,
)
from src.experiments.eval_acceptance_insertion import (
    INSERTION_METHODS,
    SUMMARY_FIELDS,
    _hard_soft,
    _summary_row,
    evaluate_method,
)
from src.experiments.run_time_window_repair_experiments import _env_config, _load_open_instances, _load_policy, _make_env
from src.models.service_policy import ServicePolicy


SERVICE_METHODS = ["service_policy_imitation", "service_policy_rl"]
ALL_METHODS = ["raw_baseline", "v2_repair_only"] + INSERTION_METHODS + SERVICE_METHODS


def _load_service_policy(path: str, device: torch.device) -> ServicePolicy:
    state = torch.load(path, map_location=device, weights_only=False)
    dims = state.get("dims", {})
    cfg = state.get("config", {})
    model = ServicePolicy(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        heads=int(cfg.get("heads", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
        k_nn_orders=int(cfg.get("k_nn_orders", 8)),
        num_encoder_layers=int(cfg.get("encoder_layers", 2)),
        order_feature_dim=int(dims.get("order_feature_dim", 22)),
        truck_feature_dim=int(dims.get("truck_feature_dim", 6)),
        drone_feature_dim=int(dims.get("drone_feature_dim", 6)),
    ).to(device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def evaluate_service_model(args: argparse.Namespace, method: str, model: ServicePolicy, out_dir: Path) -> Dict[str, Any]:
    env_cfg = _env_config(args)
    env_cfg.feature_mode = "service_v2"
    env_cfg.decision_mode = str(args.decision_mode)
    env_cfg.response_window = float(args.response_window)
    open_instances = _load_open_instances(args)
    max_steps = 8 * (int(args.N) + 1)
    episode_summaries: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    drone_rows: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"method_name": method, "instances": []}
    for idx in range(1, int(args.eval_instances) + 1):
        seed = int(args.eval_seed) * 100000 + idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = _make_env(args, env_cfg, open_instances, seed)
        e = env.copy()
        obs = e.reset()
        traj: List[Dict[str, Any]] = []
        total_cost = 0.0
        done = False
        step_debug: List[Dict[str, Any]] = []
        for step in range(max_steps):
            try:
                action, policy_debug = model.act(e, obs, greedy=True)
                obs2, reward, done, info = e.step(action)
            except Exception as exc:
                action = (e.K_NONE, 0)
                obs2, reward, done, info = e.step(action)
                policy_debug = {"fallback": "reject_or_wait", "error": str(exc)}
            total_cost += -float(reward)
            traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "policy_debug": policy_debug})
            if step < 30:
                step_debug.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": policy_debug})
            obs = obs2
            if done:
                break
        if not done:
            traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -1_000_000.0, "info": {"timeout": True}})
            total_cost += 1_000_000.0
        summary, orders, drone = analyze_episode(e, traj, model_name=method, instance_id=idx, objective_cost=total_cost)
        episode_summaries.append(summary)
        order_rows.extend(orders)
        drone_rows.append(drone)
        debug["instances"].append({"instance_id": idx, "cost": total_cost, "debug_steps": step_debug})
        if idx % max(1, int(args.eval_progress_every)) == 0:
            print(f"[{method}] {idx}/{args.eval_instances}: acc={summary['acceptance_rate']:.3f} on_time={summary['on_time_rate']:.3f} late={summary['late_orders']}")

    overall, drone_detail = aggregate_model(method, episode_summaries, order_rows, drone_rows)
    hard, soft = _hard_soft(order_rows, [drone_detail])
    overall["hard_constraint_violations"] = int(hard)
    overall["soft_time_window_violations"] = int(soft)
    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    write_csv(str(method_dir / "overall_summary.csv"), [overall], OVERALL_FIELDS + ["hard_constraint_violations", "soft_time_window_violations"])
    write_csv(str(method_dir / "order_details.csv"), order_rows, ORDER_DETAIL_FIELDS)
    write_csv(str(method_dir / "drone_details.csv"), [drone_detail], DRONE_DETAIL_FIELDS)
    write_json(str(method_dir / "debug_log.json"), debug)
    reasons: Dict[str, int] = {}
    for row in order_rows:
        reason = str(row.get("rejection_reason", "") or "")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    with (method_dir / "rejection_reason_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rejection_reason", "count"])
        writer.writeheader()
        for reason, count in sorted(reasons.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow({"rejection_reason": reason, "count": count})
    return overall


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ServicePolicy against baseline, repair, and insertion teachers.")
    p.add_argument("--output-dir", type=str, default="experiments/service_v2/evaluation/service_policy")
    p.add_argument("--service-model-path", type=str, default="experiments/service_v2/models/service_policy_imitation_best.pt")
    p.add_argument("--baseline-model-path", type=str, default="experiments/frozen_models_20260419/model_main_ep200.pt")
    p.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    p.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    p.add_argument("--eval-instances", type=int, default=5)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-progress-every", type=int, default=5)
    p.add_argument("--methods", type=str, default="raw_baseline,edd_insertion,service_policy_imitation")
    p.add_argument("--decision-mode", type=str, default="legacy")
    p.add_argument("--feature-mode", type=str, default="legacy")
    p.add_argument("--response-window", type=float, default=0.0)
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
    p.add_argument("--accept-reward", type=float, default=0.0)
    p.add_argument("--reject-feasible-penalty", type=float, default=0.0)
    p.add_argument("--reject-infeasible-penalty", type=float, default=0.0)
    p.add_argument("--expired-order-penalty", type=float, default=0.0)
    p.add_argument("--on-time-reward", type=float, default=0.0)
    p.add_argument("--late-order-penalty", type=float, default=0.0)
    p.add_argument("--lateness-duration-penalty", type=float, default=0.5)
    p.add_argument("--severe-lateness-threshold", type=float, default=10.0)
    p.add_argument("--severe-lateness-penalty", type=float, default=0.0)
    p.add_argument("--max-lateness-penalty", type=float, default=0.0)
    p.add_argument("--overtime-penalty", type=float, default=1.0)
    p.add_argument("--time-cost-weight", type=float, default=1.0)
    p.add_argument("--energy-cost-weight", type=float, default=0.08)
    p.add_argument("--distance-cost-weight", type=float, default=0.0)
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
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    baseline_policy = _load_policy(args)
    service_model = None
    if any(m in SERVICE_METHODS for m in methods):
        service_model = _load_service_policy(args.service_model_path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        if method in SERVICE_METHODS:
            assert service_model is not None
            results[method] = evaluate_service_model(args, method, service_model, out_dir)
        else:
            results[method] = evaluate_method(args, method, baseline_policy, out_dir)
    raw = results.get("raw_baseline") or next(iter(results.values()))
    rows = [_summary_row(method, int(args.eval_instances), overall, raw) for method, overall in results.items()]
    write_csv(str(out_dir / "overall_summary.csv"), rows, SUMMARY_FIELDS)
    write_json(str(out_dir / "overall_summary.json"), {"rows": rows})
    lines = [
        "# ServicePolicy Evaluation",
        "",
        f"- eval_instances: `{args.eval_instances}`",
        "",
        "| method | acc | on_time | late | avg_late | max_late | energy | distance | hard | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_name']} | {row['acceptance_rate']:.6f} | {row['on_time_rate']:.6f} | "
            f"{row['late_orders']} | {row['average_lateness']:.6f} | {row['max_lateness']:.6f} | "
            f"{row['total_energy_consumption']:.6f} | {row['total_flight_distance']:.6f} | "
            f"{row['hard_constraint_violations']} | {row['is_small_data_pass']} |"
        )
    (out_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir.resolve()), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

