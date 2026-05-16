from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from src.evaluation.service_metrics import aggregate_model, analyze_episode, write_csv
from src.experiments.business_env_profiles import BUSINESS_ENV_PROFILES, apply_business_env_profile_to_args, profile_names
from src.experiments.eval_acceptance_insertion import _hard_soft, _insertion_cfg
from src.experiments.run_business_constraint_sensitivity_80_50 import _apply_due_extension, _split_envs
from src.experiments.run_target_80_50_feasibility import _candidate_actions, _decision_acceptable, _score_action
from src.experiments.run_time_window_repair_experiments import _env_config, _load_open_instances, _make_env
from src.graph.build_graph_pyg import build_hgat_heterodata
from src.schedulers.acceptance_insertion import select_acceptance_insertion_action
from src.schedulers.feasibility import classify_order_feasibility


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sample_target_node(sample: Dict[str, Any]) -> int:
    current = int(sample.get("current_order_id", -1))
    if current > 0:
        return current
    for key in ("best_next_order", "action_j"):
        node = int(sample.get(key, -1))
        if node > 0:
            return node
    return -1


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
    teacher_on_time = 1 if target_node > 0 and lateness_label <= 1e-9 else 0
    return {
        "data": data.to("cpu"),
        "extra": {key: value.cpu() if torch.is_tensor(value) else value for key, value in extra.items()},
        "state_features": "hgat_heterodata",
        "current_order_id": int(current),
        "action_k": int(k),
        "action_j": int(j),
        "teacher_action": [int(k), int(j)],
        "accept_label": int(accept_label),
        "reject_label": int(1 - accept_label) if accept_label in {0, 1} else -100,
        "best_next_order": int(best_next_order),
        "next_service_order_label": int(best_next_order),
        "assignment_label": int(k),
        "order_priority_label": int(best_next_order),
        "predicted_lateness_label": float(lateness_label),
        "lateness_risk_label": float(lateness_label),
        "insertion_score_label": float(score),
        "teacher_score": float(score),
        "teacher_on_time_label": int(teacher_on_time),
        "t": float(obs.get("t", 0.0)),
    }


def _candidate_labels(env: Any, obs: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    current = int(obs.get("current_decision_request", -1))
    if current > 0:
        for j in [current, 0]:
            score = _score_action(env, obs, (env.K_NONE, int(j)), "regret") if j > 0 else 1e6
            late = float(classify_order_feasibility(env, j).predicted_lateness) if j > 0 else 0.0
            labels.append(
                {
                    "k": int(env.K_NONE),
                    "j": int(j),
                    "phase": "acceptance",
                    "candidate_accept_label": int(j == current),
                    "candidate_score": float(score),
                    "predicted_lateness": float(late),
                    "candidate_on_time": int(j > 0 and late <= 1e-9),
                }
            )
        return labels

    for k, j in _candidate_actions(env, obs, limit=max(1, int(limit))):
        if len(labels) >= int(limit):
            break
        late = float(classify_order_feasibility(env, int(j)).predicted_lateness) if int(j) > 0 else 0.0
        labels.append(
            {
                "k": int(k),
                "j": int(j),
                "phase": "route",
                "candidate_score": float(_score_action(env, obs, (int(k), int(j)), "regret")),
                "predicted_lateness": float(late),
                "candidate_on_time": int(int(j) > 0 and late <= 1e-9),
            }
        )
    return labels


def _enrich_sample_with_step(sample: Dict[str, Any], env: Any, obs: Dict[str, Any], action: Any, info: Dict[str, Any]) -> None:
    k, j = int(action[0]), int(action[1])
    i = int(obs.get("i", 0))
    drone_distance = 0.0
    if k != env.K_NONE and k > 0 and j > 0:
        drone_distance = float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
    truck_distance = float(info.get("road_distance", 0.0))
    truck_energy = float(info.get("truck_energy_use", 0.0))
    drone_energy = float(info.get("energy_use", 0.0)) + float(info.get("drone_idle_energy_use", 0.0))
    served_nodes = [int(x) for x in (info.get("served_nodes", []) or [])]
    service_lateness = info.get("service_lateness", {}) or {}
    target = int(sample.get("current_order_id", -1))
    if target <= 0:
        target = int(sample.get("best_next_order", -1))
    target_late = float(service_lateness.get(target, 0.0) or 0.0) if target > 0 else 0.0
    sample.update(
        {
            "served_nodes": served_nodes,
            "step_lateness": float(info.get("lateness", target_late) or 0.0),
            "actual_lateness_label": float(target_late),
            "actual_on_time_label": int(target > 0 and target in served_nodes and target_late <= 1e-9),
            "step_energy_use": float(truck_energy + drone_energy),
            "step_truck_energy_use": truck_energy,
            "step_drone_energy_use": drone_energy,
            "step_flight_distance": float(truck_distance + drone_distance),
            "step_truck_distance": truck_distance,
            "step_drone_distance": drone_distance,
        }
    )


def _apply_episode_outcomes(
    samples: Sequence[Dict[str, Any]],
    order_rows: Sequence[Dict[str, Any]],
    *,
    relabel_risky_accepts: bool,
    safe_accept_lateness_threshold: float,
) -> Dict[str, int]:
    by_node = {int(row.get("node_id", -1)): row for row in order_rows}
    counts = {
        "outcome_labeled_samples": 0,
        "outcome_late_samples": 0,
        "accept_relabels": 0,
        "teacher_reject_labels": 0,
    }
    for sample in samples:
        node = _sample_target_node(sample)
        row = by_node.get(int(node))
        if row is None:
            continue
        accepted = _bool(row.get("accepted"))
        late = _bool(row.get("late"))
        lateness = float(row.get("lateness_duration", 0.0) or 0.0)
        sample.update(
            {
                "final_order_accepted": int(accepted),
                "final_order_late": int(late),
                "final_order_on_time": int(accepted and not late),
                "actual_lateness_label": float(lateness),
                "actual_on_time_label": int(accepted and not late),
                "outcome_rejection_reason": str(row.get("rejection_reason", "")),
            }
        )
        counts["outcome_labeled_samples"] += 1
        counts["outcome_late_samples"] += int(lateness > float(safe_accept_lateness_threshold))

        if int(sample.get("accept_label", -100)) != -100 and int(sample.get("current_order_id", -1)) > 0:
            if not accepted:
                sample["accept_label"] = 0
                sample["reject_label"] = 1
                sample["accept_label_source"] = "final_reject"
                counts["teacher_reject_labels"] += 1
            elif relabel_risky_accepts and lateness > float(safe_accept_lateness_threshold):
                sample["accept_label"] = 0
                sample["reject_label"] = 1
                sample["accept_label_source"] = "risky_late_relabel"
                counts["accept_relabels"] += 1
            else:
                sample["accept_label"] = 1
                sample["reject_label"] = 0
                sample["accept_label_source"] = "teacher_accept"
    return counts


def _rollout_teacher(args: argparse.Namespace, env: Any, teacher_method: str, seed: int, max_steps: int) -> tuple[float, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    traj: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    total_cost = 0.0
    done = False
    if teacher_method == "oracle_best_acceptance":
        raise ValueError("oracle_best_acceptance is an evaluation teacher only; build imitation from oracle_best_on_time.")
    for _ in range(max_steps):
        if teacher_method == "oracle_best_on_time":
            current = int(obs.get("current_decision_request", -1))
            if current > 0:
                accept = _decision_acceptable(e, obs, current, "feasible_accept")
                action = (e.K_NONE, current if accept else 0)
                debug = {"teacher": "oracle_best_on_time", "phase": "acceptance", "decision": "accept" if accept else "reject"}
            else:
                actions = _candidate_actions(e, obs, limit=48)
                if not actions:
                    action = (e.K_NONE, 0)
                    debug = {"teacher": "oracle_best_on_time", "phase": "route", "selected_score": 0.0}
                else:
                    scored = [(a, _score_action(e, obs, a, "regret")) for a in actions]
                    action, score = min(scored, key=lambda x: x[1])
                    debug = {"teacher": "oracle_best_on_time", "phase": "route", "selected_score": float(score)}
        else:
            scheduler_cfg = _insertion_cfg(str(teacher_method))
            action, debug = select_acceptance_insertion_action(e, obs, scheduler_cfg)
        sample = _sample_from_state(e, obs, action, debug, int(args.k_nn_orders))
        if bool(getattr(args, "include_candidate_labels", False)):
            candidates = _candidate_labels(e, obs, int(args.candidate_label_limit))
            sample["candidate_labels"] = candidates
            sample["candidate_count"] = int(len(candidates))
            sample["candidate_late_count"] = int(sum(1 for c in candidates if float(c.get("predicted_lateness", 0.0)) > 1e-9))
        obs2, reward, done, info = e.step(action)
        sample["step_reward"] = float(reward)
        _enrich_sample_with_step(sample, e, obs, action, info)
        samples.append(sample)
        total_cost += -float(reward)
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "teacher_debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -1_000_000.0, "info": {"timeout": True}})
        total_cost += 1_000_000.0
    return float(total_cost), traj, samples, {"done": bool(done), "steps": len(traj)}


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    args = apply_business_env_profile_to_args(args, str(getattr(args, "env_profile", "")))
    env_cfg = _env_config(args)
    env_cfg.feature_mode = "service_v2"
    env_cfg.decision_mode = str(args.decision_mode)
    env_cfg.response_window = float(args.response_window)
    open_instances = _load_open_instances(args)
    samples: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    drone_rows: List[Dict[str, Any]] = []
    augmentation_counts = {
        "outcome_labeled_samples": 0,
        "outcome_late_samples": 0,
        "accept_relabels": 0,
        "teacher_reject_labels": 0,
    }

    for idx in range(1, int(args.instances) + 1):
        seed = int(args.eval_seed) * 100000 + idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        base_env = _make_env(args, env_cfg, open_instances, seed)
        base_env = _apply_due_extension(base_env, float(args.delivery_window_extension))
        sub_envs = _split_envs(base_env, int(args.resource_count), seed=seed)
        for rid, env in enumerate(sub_envs):
            local_seed = seed + 1000 * rid + sum(ord(c) for c in str(args.teacher_method))
            max_steps = int(args.max_steps) if int(args.max_steps) > 0 else max(32, 8 * (int(env.N) + 1))
            total_cost, traj, teacher_samples, _ = _rollout_teacher(args, env, str(args.teacher_method), local_seed, max_steps)
            instance_id = idx * 100 + rid if len(sub_envs) > 1 else idx
            summary, orders, drone = analyze_episode(env, traj, model_name=str(args.teacher_method), instance_id=instance_id, objective_cost=total_cost)
            if bool(getattr(args, "augment_outcome_labels", False)):
                local_counts = _apply_episode_outcomes(
                    teacher_samples,
                    orders,
                    relabel_risky_accepts=bool(getattr(args, "relabel_risky_accepts", False)),
                    safe_accept_lateness_threshold=float(args.safe_accept_lateness_threshold),
                )
                for key, value in local_counts.items():
                    augmentation_counts[key] = int(augmentation_counts.get(key, 0)) + int(value)
            samples.extend(teacher_samples)
            episode_summaries.append(summary)
            order_rows.extend(orders)
            drone["drone_id"] = f"resource_{rid}"
            drone_rows.append(drone)
        if idx % max(1, int(args.progress_every)) == 0:
            print(f"[imitation] {idx}/{args.instances}: samples={len(samples)} acc={summary['acceptance_rate']:.3f} on_time={summary['on_time_rate']:.3f}")

    overall, _ = aggregate_model(str(args.teacher_method), episode_summaries, order_rows, drone_rows)
    hard, soft = _hard_soft(order_rows, drone_rows)
    overall["hard_constraint_violations"] = int(hard)
    overall["soft_time_window_violations"] = int(soft)
    return {
        "samples": samples,
        "overall": overall,
        "episode_summaries": episode_summaries,
        "env_profile": str(args.env_profile),
        "augmentation": augmentation_counts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ServicePolicy imitation dataset from acceptance-insertion teacher.")
    p.add_argument("--output-path", type=str, default="experiments/service_v2/imitation/imitation_dataset.pt")
    p.add_argument("--report-path", type=str, default="experiments/service_v2/reports/imitation_quality.md")
    p.add_argument(
        "--teacher-method",
        type=str,
        default="edd_insertion",
        choices=[
            "edd_insertion",
            "regret_insertion",
            "min_lateness_insertion",
            "hybrid_score_insertion",
            "oracle_best_acceptance",
            "oracle_best_on_time",
        ],
    )
    p.add_argument("--env-profile", type=str, default="", choices=[""] + sorted(BUSINESS_ENV_PROFILES), help=f"Business profile: {profile_names()}.")
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--augment-outcome-labels", type=_as_bool, default=False)
    p.add_argument("--relabel-risky-accepts", type=_as_bool, default=False)
    p.add_argument("--safe-accept-lateness-threshold", type=float, default=1.0)
    p.add_argument("--include-candidate-labels", type=_as_bool, default=False)
    p.add_argument("--candidate-label-limit", type=int, default=16)
    p.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    p.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--decision-mode", type=str, default="legacy")
    p.add_argument("--response-window", type=float, default=0.0)
    p.add_argument("--delivery-window-extension", type=float, default=0.0)
    p.add_argument("--resource-count", type=int, default=1)
    p.add_argument("--order-density-ratio", type=float, default=1.0)
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
    augmentation = payload.get("augmentation", {}) or {}
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
        f"| hard_constraint_violations | {int(overall.get('hard_constraint_violations', 0))} |",
        f"| soft_time_window_violations | {int(overall.get('soft_time_window_violations', 0))} |",
        f"| outcome_labeled_samples | {int(augmentation.get('outcome_labeled_samples', 0))} |",
        f"| outcome_late_samples | {int(augmentation.get('outcome_late_samples', 0))} |",
        f"| accept_relabels | {int(augmentation.get('accept_relabels', 0))} |",
        f"| teacher_reject_labels | {int(augmentation.get('teacher_reject_labels', 0))} |",
        "",
        "This dataset should not be used for long training unless the teacher passes the small-data gate.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "report_path": str(report_path), "samples": len(payload["samples"]), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
