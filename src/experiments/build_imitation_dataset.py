from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.evaluation.service_metrics import aggregate_model, analyze_episode, write_csv
from src.experiments.eval_acceptance_insertion import _insertion_cfg
from src.experiments.run_time_window_repair_experiments import _env_config, _load_open_instances, _make_env
from src.graph.build_graph_pyg import build_hgat_heterodata
from src.schedulers.acceptance_insertion import select_acceptance_insertion_action
from src.schedulers.feasibility import classify_order_feasibility


def _sample_from_state(env: Any, obs: Dict[str, Any], action: Any, debug: Dict[str, Any], k_nn_orders: int) -> Dict[str, Any]:
    data, extra = build_hgat_heterodata(env, obs, k_nn_orders=k_nn_orders)
    current = int(obs.get("current_decision_request", -1))
    k, j = int(action[0]), int(action[1])
    accept_label = -100
    best_next_order = -100
    target_node = current if current > 0 else j
    if current > 0:
        accept_label = 1 if j == current else 0
    elif j > 0:
        best_next_order = j
    lateness_label = 0.0
    if target_node > 0:
        lateness_label = float(classify_order_feasibility(env, target_node).predicted_lateness)
    score = float(debug.get("selected_score", debug.get("accept_score", 0.0)) or 0.0)
    return {
        "data": data.to("cpu"),
        "extra": {key: value.cpu() if torch.is_tensor(value) else value for key, value in extra.items()},
        "current_order_id": int(current),
        "action_k": int(k),
        "action_j": int(j),
        "accept_label": int(accept_label),
        "reject_label": int(1 - accept_label) if accept_label in {0, 1} else -100,
        "best_next_order": int(best_next_order),
        "predicted_lateness_label": float(lateness_label),
        "insertion_score_label": float(score),
        "t": float(obs.get("t", 0.0)),
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    env_cfg = _env_config(args)
    env_cfg.feature_mode = "service_v2"
    env_cfg.decision_mode = str(args.decision_mode)
    env_cfg.response_window = float(args.response_window)
    open_instances = _load_open_instances(args)
    scheduler_cfg = _insertion_cfg(str(args.teacher_method))
    samples: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    drone_rows: List[Dict[str, Any]] = []
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else 8 * (int(args.N) + 1)

    for idx in range(1, int(args.instances) + 1):
        seed = int(args.eval_seed) * 100000 + idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = _make_env(args, env_cfg, open_instances, seed)
        e = env.copy()
        obs = e.reset()
        traj: List[Dict[str, Any]] = []
        total_cost = 0.0
        done = False
        for _ in range(max_steps):
            action, debug = select_acceptance_insertion_action(e, obs, scheduler_cfg)
            samples.append(_sample_from_state(e, obs, action, debug, int(args.k_nn_orders)))
            obs2, reward, done, info = e.step(action)
            total_cost += -float(reward)
            traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "teacher_debug": debug})
            obs = obs2
            if done:
                break
        if not done:
            traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -1_000_000.0, "info": {"timeout": True}})
            total_cost += 1_000_000.0
        summary, orders, drone = analyze_episode(e, traj, model_name=str(args.teacher_method), instance_id=idx, objective_cost=total_cost)
        episode_summaries.append(summary)
        order_rows.extend(orders)
        drone_rows.append(drone)
        if idx % max(1, int(args.progress_every)) == 0:
            print(f"[imitation] {idx}/{args.instances}: samples={len(samples)} acc={summary['acceptance_rate']:.3f} on_time={summary['on_time_rate']:.3f}")

    overall, _ = aggregate_model(str(args.teacher_method), episode_summaries, order_rows, drone_rows)
    return {"samples": samples, "overall": overall, "episode_summaries": episode_summaries}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ServicePolicy imitation dataset from acceptance-insertion teacher.")
    p.add_argument("--output-path", type=str, default="experiments/service_v2/imitation/imitation_dataset.pt")
    p.add_argument("--report-path", type=str, default="experiments/service_v2/reports/imitation_quality.md")
    p.add_argument("--teacher-method", type=str, default="edd_insertion", choices=["edd_insertion", "regret_insertion", "min_lateness_insertion", "hybrid_score_insertion"])
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    p.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--decision-mode", type=str, default="legacy")
    p.add_argument("--response-window", type=float, default=0.0)
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--k-nn-orders", type=int, default=8)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    report_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dataset(args)
    torch.save(payload, output_path)
    overall = payload["overall"]
    lines = [
        "# Imitation Quality",
        "",
        f"- teacher_method: `{args.teacher_method}`",
        f"- instances: `{args.instances}`",
        f"- samples: `{len(payload['samples'])}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| acceptance_rate | {float(overall.get('acceptance_rate', 0.0)):.6f} |",
        f"| on_time_rate | {float(overall.get('on_time_rate', 0.0)):.6f} |",
        f"| late_orders | {int(overall.get('late_orders', 0))} |",
        f"| average_lateness | {float(overall.get('average_lateness', 0.0)):.6f} |",
        f"| max_lateness | {float(overall.get('max_lateness', 0.0)):.6f} |",
        "",
        "This dataset should not be used for long training unless the teacher passes the small-data gate.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "report_path": str(report_path), "samples": len(payload["samples"]), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()

