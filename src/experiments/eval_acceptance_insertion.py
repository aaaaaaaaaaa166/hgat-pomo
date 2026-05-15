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
from src.evaluation.time_window_inference import TimeWindowInferenceConfig
from src.evaluation.v2_repair import V2RepairConfig, repair_trajectory
from src.evaluation.v2_scheduler import V2SchedulerConfig
from src.evaluation.v2_sequence_scoring import V2SequenceScoreConfig
from src.evaluation.v2_scheduler import rollout_v2_scheduler, select_v2_action
from src.experiments.run_time_window_repair_experiments import (
    _env_config,
    _load_open_instances,
    _load_policy,
    _make_env,
    rollout_inference,
)
from src.schedulers.acceptance_insertion import AcceptanceInsertionConfig, rollout_acceptance_insertion
from src.schedulers.feasibility import classify_order_feasibility
from src.schedulers.insertion_objective import InsertionObjectiveConfig
from src.schedulers.joint_accept_route_beam import (
    JointAcceptRouteBeamConfig,
    TailRiskBudget,
    rollout_joint_accept_route_beam,
    rollout_tail_risk_constrained_joint_beam,
    select_joint_accept_route_action,
)
from src.schedulers.joint_beam_objective import JointBeamObjectiveConfig


INSERTION_METHODS = [
    "edd_insertion",
    "regret_insertion",
    "min_lateness_insertion",
    "hybrid_score_insertion",
    "beam_oracle_insertion",
    "deadline_beam_oracle",
    "conservative_deadline_beam",
    "ontime_beam_oracle",
    "guarded_ontime_beam",
    "policy_accept_ontime_beam",
    "joint_accept_route_beam",
    "joint_accept_route_beam_guarded",
    "tail_risk_constrained_joint_beam",
]
ALL_METHODS = ["raw_baseline", "v2_repair_only"] + INSERTION_METHODS

SUMMARY_FIELDS = [
    "method_name",
    "eval_instances",
    "acceptance_rate",
    "acceptance_rate_delta_vs_raw_baseline",
    "on_time_rate",
    "on_time_rate_delta_vs_raw_baseline",
    "late_orders",
    "late_orders_delta_vs_raw_baseline",
    "average_lateness",
    "average_lateness_delta_vs_raw_baseline",
    "max_lateness",
    "max_lateness_delta_vs_raw_baseline",
    "total_energy_consumption",
    "total_energy_delta_vs_raw_baseline",
    "total_flight_distance",
    "total_distance_delta_vs_raw_baseline",
    "hard_constraint_violations",
    "is_small_data_pass",
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


def _v2_sched_cfg(seed: int) -> V2SchedulerConfig:
    return V2SchedulerConfig(
        lookahead_depth=2,
        beam_size=4,
        candidate_top_k=8,
        max_runtime_seconds=0.20,
        deterministic_seed=int(seed),
        use_beam_search=True,
        score=V2SequenceScoreConfig(),
    )


def _oracle_sched_cfg(method: str, seed: int) -> V2SchedulerConfig:
    if method == "conservative_deadline_beam":
        score = V2SequenceScoreConfig(
            accept_reward=0.0,
            on_time_reward=9.0,
            reject_penalty=2.5,
            lateness_weight=4.0,
            late_count_penalty=16.0,
            max_lateness_weight=5.0,
            severe_lateness_threshold=6.0,
            severe_lateness_weight=10.0,
            energy_weight=0.05,
            distance_weight=0.03,
            future_tight_window_weight=2.0,
        )
        return V2SchedulerConfig(
            lookahead_depth=4,
            beam_size=14,
            candidate_top_k=16,
            max_runtime_seconds=1.80,
            deterministic_seed=int(seed),
            use_beam_search=True,
            score=score,
        )
    if method == "ontime_beam_oracle":
        score = V2SequenceScoreConfig(
            accept_reward=0.25,
            on_time_reward=12.0,
            reject_penalty=3.5,
            lateness_weight=5.0,
            late_count_penalty=20.0,
            max_lateness_weight=6.0,
            severe_lateness_threshold=5.0,
            severe_lateness_weight=12.0,
            energy_weight=0.05,
            distance_weight=0.03,
            future_tight_window_weight=2.5,
        )
        return V2SchedulerConfig(
            lookahead_depth=4,
            beam_size=16,
            candidate_top_k=18,
            max_runtime_seconds=2.00,
            deterministic_seed=int(seed),
            use_beam_search=True,
            score=score,
        )
    if method == "deadline_beam_oracle":
        score = V2SequenceScoreConfig(
            accept_reward=0.5,
            on_time_reward=7.0,
            reject_penalty=4.0,
            lateness_weight=2.5,
            late_count_penalty=10.0,
            max_lateness_weight=3.5,
            severe_lateness_threshold=8.0,
            severe_lateness_weight=7.0,
            energy_weight=0.04,
            distance_weight=0.02,
            future_tight_window_weight=1.5,
        )
        return V2SchedulerConfig(
            lookahead_depth=4,
            beam_size=12,
            candidate_top_k=14,
            max_runtime_seconds=1.50,
            deterministic_seed=int(seed),
            use_beam_search=True,
            score=score,
        )
    score = V2SequenceScoreConfig(
        accept_reward=1.5,
        on_time_reward=5.5,
        reject_penalty=6.0,
        lateness_weight=2.0,
        late_count_penalty=8.0,
        max_lateness_weight=2.8,
        severe_lateness_threshold=8.0,
        severe_lateness_weight=6.0,
        energy_weight=0.04,
        distance_weight=0.02,
        future_tight_window_weight=1.0,
    )
    return V2SchedulerConfig(
        lookahead_depth=3,
        beam_size=10,
        candidate_top_k=12,
        max_runtime_seconds=1.00,
        deterministic_seed=int(seed),
        use_beam_search=True,
        score=score,
    )


def _v2_repair_cfg() -> V2RepairConfig:
    return V2RepairConfig(enable_v2_repair=True, repair_max_iterations=50, repair_window_size=4)


def _insertion_cfg(method: str) -> AcceptanceInsertionConfig:
    objective = InsertionObjectiveConfig(
        accept_reward=6.0,
        on_time_reward=8.0,
        reject_penalty=6.0,
        late_order_penalty=8.0,
        lateness_weight=1.2,
        max_lateness_weight=2.0,
        future_impact_weight=1.0,
        energy_weight=0.04,
        distance_weight=0.02,
    )
    if method == "min_lateness_insertion":
        objective.accept_reward = 3.0
        objective.on_time_reward = 10.0
        objective.late_order_penalty = 12.0
    elif method == "regret_insertion":
        objective.future_impact_weight = 2.0
        objective.accept_reward = 5.0
    elif method == "edd_insertion":
        objective.on_time_reward = 9.0
        objective.max_lateness_weight = 2.5
    return AcceptanceInsertionConfig(method=method, candidate_top_k=12, objective=objective)


def _joint_beam_cfg(args: argparse.Namespace, *, guarded: bool = False) -> JointAcceptRouteBeamConfig:
    return JointAcceptRouteBeamConfig(
        beam_size=int(args.joint_beam_size),
        lookahead_depth=int(args.joint_lookahead_depth),
        candidate_top_k=int(args.joint_candidate_top_k),
        max_expanded_states=int(args.joint_max_expanded_states),
        time_limit_seconds=float(args.joint_time_limit_seconds),
        enable_dominance_pruning=bool(args.joint_enable_dominance_pruning),
        enable_acceptance_guard=bool(guarded),
        guard_max_lateness_factor=float(args.joint_guard_max_lateness_factor),
        guard_distance_factor=float(args.joint_guard_distance_factor),
        guard_energy_factor=float(args.joint_guard_energy_factor),
        guard_abs_lateness_slack=float(args.joint_guard_abs_lateness_slack),
        guard_abs_distance_slack=float(args.joint_guard_abs_distance_slack),
        guard_abs_energy_slack=float(args.joint_guard_abs_energy_slack),
        guard_sim_max_steps=int(args.joint_guard_sim_max_steps),
        objective=JointBeamObjectiveConfig(
            accept_weight=float(args.joint_accept_weight),
            on_time_weight=float(args.joint_on_time_weight),
            late_weight=float(args.joint_late_weight),
            lateness_weight=float(args.joint_lateness_weight),
            max_lateness_weight=float(args.joint_max_lateness_weight),
            energy_weight=float(args.joint_energy_weight),
            distance_weight=float(args.joint_distance_weight),
            severe_late_weight=float(args.joint_severe_late_weight),
            severe_lateness_threshold=float(args.severe_lateness_threshold),
            hard_violation_weight=float(args.joint_hard_violation_weight),
        ),
    )


def _severe_late_count(order_rows: Sequence[Dict[str, Any]], threshold: float) -> int:
    return int(
        sum(
            1
            for row in order_rows
            if _bool(row.get("accepted")) and _num(row.get("lateness_duration")) > float(threshold) + 1e-9
        )
    )


def _tail_risk_budget_from_raw(
    args: argparse.Namespace,
    raw_summary: Dict[str, Any],
    raw_order_rows: Sequence[Dict[str, Any]],
) -> TailRiskBudget:
    threshold = float(args.severe_lateness_threshold)
    max_ratio = float(args.tail_risk_max_lateness_ratio)
    hard_cap = float(args.severe_lateness_hard_cap)
    if hard_cap <= 0.0:
        hard_cap = _num(raw_summary.get("max_lateness")) * max_ratio
    return TailRiskBudget(
        baseline_acceptance_rate=_num(raw_summary.get("acceptance_rate")),
        baseline_on_time_rate=_num(raw_summary.get("on_time_rate")),
        baseline_late_orders=int(_num(raw_summary.get("late_orders"))),
        baseline_average_lateness=_num(raw_summary.get("average_lateness")),
        baseline_max_lateness=_num(raw_summary.get("max_lateness")),
        baseline_total_energy=_num(raw_summary.get("total_energy_consumption")),
        baseline_total_distance=_num(raw_summary.get("total_flight_distance")),
        baseline_severe_late_count=_severe_late_count(raw_order_rows, threshold),
        max_lateness_ratio=max_ratio,
        avg_lateness_ratio=float(args.tail_risk_avg_lateness_ratio),
        energy_ratio=float(args.tail_risk_energy_ratio),
        distance_ratio=float(args.tail_risk_distance_ratio),
        severe_lateness_threshold=threshold,
        severe_lateness_hard_cap=hard_cap,
        enable_baseline_anchored_budget=bool(args.enable_baseline_anchored_budget),
    )


def _raw_baseline_budget_for_instance(
    args: argparse.Namespace,
    policy: Any,
    env: Any,
    *,
    seed: int,
    instance_id: int,
    max_steps: int,
) -> Tuple[TailRiskBudget, Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    raw_seed = int(seed) + sum(ord(c) for c in "raw_baseline")
    np.random.seed(raw_seed)
    torch.manual_seed(raw_seed)
    costs, trajs, logs = rollout_inference(policy, env, TimeWindowInferenceConfig(), K=int(args.K), max_steps=max_steps)
    best_id = int(costs.argmin())
    raw_cost = float(costs[best_id])
    raw_traj = trajs[best_id]
    raw_summary, raw_orders, raw_drone = analyze_episode(
        env,
        raw_traj,
        model_name="raw_baseline_budget",
        instance_id=instance_id,
        objective_cost=raw_cost,
    )
    hard, soft = _hard_soft(raw_orders, [raw_drone])
    raw_summary["hard_constraint_violations"] = int(hard)
    raw_summary["soft_time_window_violations"] = int(soft)
    budget = _tail_risk_budget_from_raw(args, raw_summary, raw_orders)
    return budget, {
        "raw_seed": int(raw_seed),
        "raw_best_k": int(best_id),
        "raw_cost": float(raw_cost),
        "raw_summary": raw_summary,
        "raw_actions": [item.get("action") for item in raw_traj],
        "raw_logs": logs[best_id],
        "budget": budget.to_dict(),
    }, [item.get("action") for item in raw_traj], raw_traj


def _guarded_accept(
    env: Any,
    order_id: int,
    *,
    slack_threshold: float = 2.0,
    max_backlog: int = 16,
    max_dynamic_accepts: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    obs = env.get_obs()
    feas = classify_order_feasibility(env, int(order_id))
    accepted = np.asarray(obs.get("accepted", []), dtype=np.float32)
    served = np.asarray(obs.get("served", []), dtype=np.float32)
    rejected = np.asarray(obs.get("rejected", np.zeros_like(accepted)), dtype=np.float32)
    backlog = 0
    if accepted.size and served.size:
        mask = (accepted > 0.5) & (served <= 0.5) & (rejected <= 0.5)
        if mask.size:
            mask[0] = False
        backlog = int(mask.sum())
    accepted_dynamic = 0
    if accepted.size:
        dyn = np.asarray(env.is_dynamic, dtype=np.int8)
        accepted_dynamic = int(((accepted > 0.5) & (dyn > 0)).sum())
    accept = (
        not bool(feas.hard_infeasible)
        and float(feas.predicted_lateness) <= 1e-9
        and float(feas.slack_after_arrival) >= float(slack_threshold)
        and backlog <= int(max_backlog)
        and accepted_dynamic < int(max_dynamic_accepts)
    )
    return bool(accept), {
        "feasibility": feas.to_dict(),
        "backlog": int(backlog),
        "slack_threshold": float(slack_threshold),
        "max_backlog": int(max_backlog),
        "accepted_dynamic": int(accepted_dynamic),
        "max_dynamic_accepts": int(max_dynamic_accepts),
    }


def rollout_guarded_ontime_beam(env: Any, seed: int, *, max_steps: int) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    cfg = _oracle_sched_cfg("ontime_beam_oracle", seed)
    traj: List[Dict[str, Any]] = []
    debug_steps: List[Dict[str, Any]] = []
    total_cost = 0.0
    done = False
    for step in range(int(max_steps)):
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            do_accept, guard = _guarded_accept(e, current, slack_threshold=8.0, max_backlog=12, max_dynamic_accepts=1)
            action = (e.K_NONE, current if do_accept else 0)
            debug = {"mode": "guarded_acceptance", "selected": "accept" if do_accept else "reject", **guard}
        else:
            action, debug = select_v2_action(e, obs, cfg)
            debug = {"mode": "guarded_ontime_route", **debug}
        obs2, reward, done, info = e.step(action)
        total_cost += -float(reward)
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "guarded_beam_debug": debug})
        if step < 30:
            debug_steps.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += 1_000_000.0
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -1_000_000.0, "info": {"timeout": True}})
    return float(total_cost), traj, {"done": bool(done), "steps": len(traj), "debug_steps": debug_steps}


def rollout_policy_accept_ontime_beam(
    policy: Any,
    env: Any,
    seed: int,
    *,
    max_steps: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    cfg = _oracle_sched_cfg("ontime_beam_oracle", seed)
    traj: List[Dict[str, Any]] = []
    debug_steps: List[Dict[str, Any]] = []
    total_cost = 0.0
    done = False
    for step in range(int(max_steps)):
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            try:
                base_action, _ = policy.act(e, obs, greedy=True)
                accept = int(base_action[1]) == current
            except Exception as exc:
                accept = False
                base_action = (e.K_NONE, 0)
                debug_error = str(exc)
            else:
                debug_error = ""
            action = (e.K_NONE, current if accept else 0)
            debug = {
                "mode": "policy_acceptance",
                "selected": "accept" if accept else "reject",
                "baseline_action": [int(base_action[0]), int(base_action[1])],
                "error": debug_error,
            }
        else:
            action, debug = select_v2_action(e, obs, cfg)
            debug = {"mode": "policy_accept_ontime_route", **debug}
        obs2, reward, done, info = e.step(action)
        total_cost += -float(reward)
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "policy_accept_beam_debug": debug})
        if step < 30:
            debug_steps.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += 1_000_000.0
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -1_000_000.0, "info": {"timeout": True}})
    return float(total_cost), traj, {"done": bool(done), "steps": len(traj), "debug_steps": debug_steps}


def rollout_joint_guarded_accept_ontime_route(
    env: Any,
    seed: int,
    cfg: JointAcceptRouteBeamConfig,
    *,
    max_steps: int,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    e = env.copy()
    obs = e.reset()
    route_cfg = _oracle_sched_cfg("ontime_beam_oracle", seed)
    traj: List[Dict[str, Any]] = []
    debug_steps: List[Dict[str, Any]] = []
    total_cost = 0.0
    done = False
    expanded_total = 0
    for step in range(int(max_steps)):
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            action, debug = select_joint_accept_route_action(e, obs, cfg)
            debug = {**debug, "mode": "joint_guarded_acceptance"}
            expanded_total += int(debug.get("expanded_states", 0))
        else:
            action, debug = select_v2_action(e, obs, route_cfg)
            debug = {**debug, "mode": "joint_guarded_ontime_route"}
        obs2, reward, done, info = e.step(action)
        total_cost += -float(reward)
        traj.append({"obs": obs, "obs2": obs2, "action": action, "reward": float(reward), "info": info, "joint_guarded_debug": debug})
        if step < 30:
            debug_steps.append({"step": int(step), "t": float(obs.get("t", 0.0)), "action": [int(action[0]), int(action[1])], "debug": debug})
        obs = obs2
        if done:
            break
    if not done:
        total_cost += float(cfg.objective.hard_violation_weight)
        traj.append({"obs": obs, "action": ("TIMEOUT",), "reward": -float(cfg.objective.hard_violation_weight), "info": {"timeout": True}})
    return float(total_cost), traj, {
        "done": bool(done),
        "steps": len(traj),
        "expanded_states": int(expanded_total),
        "debug_steps": debug_steps,
    }


def evaluate_method(args: argparse.Namespace, method: str, policy: Any, out_dir: Path) -> Dict[str, Any]:
    env_cfg = _env_config(args)
    env_cfg.decision_mode = str(args.decision_mode)
    env_cfg.response_window = float(args.response_window)
    env_cfg.accept_reward = float(args.accept_reward)
    env_cfg.reject_feasible_penalty = float(args.reject_feasible_penalty)
    env_cfg.reject_infeasible_penalty = float(args.reject_infeasible_penalty)
    env_cfg.expired_order_penalty = float(args.expired_order_penalty)
    env_cfg.on_time_reward = float(args.on_time_reward)
    env_cfg.late_count_penalty = float(args.late_order_penalty)
    env_cfg.lateness_penalty = float(args.lateness_duration_penalty)
    env_cfg.severe_lateness_threshold = float(args.severe_lateness_threshold)
    env_cfg.severe_lateness_penalty = float(args.severe_lateness_penalty)
    env_cfg.max_lateness_penalty = float(args.max_lateness_penalty)
    env_cfg.distance_cost_weight = float(args.distance_cost_weight)
    env_cfg.feature_mode = str(args.feature_mode)
    open_instances = _load_open_instances(args)
    max_steps = 8 * (int(args.N) + 1)
    episode_summaries: List[Dict[str, Any]] = []
    order_rows: List[Dict[str, Any]] = []
    drone_rows: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"method_name": method, "instances": []}
    for idx in range(1, int(args.eval_instances) + 1):
        seed = int(args.eval_seed) * 100000 + idx
        np.random.seed(seed + sum(ord(c) for c in method))
        torch.manual_seed(seed + sum(ord(c) for c in method))
        env = _make_env(args, env_cfg, open_instances, seed)
        if method == "raw_baseline":
            costs, trajs, logs = rollout_inference(policy, env, TimeWindowInferenceConfig(), K=int(args.K), max_steps=max_steps)
            best_id = int(costs.argmin())
            cost = float(costs[best_id])
            traj = trajs[best_id]
            method_debug = {"best_k": best_id, "logs": logs[best_id]}
        elif method == "v2_repair_only":
            costs, trajs, logs = rollout_inference(policy, env, TimeWindowInferenceConfig(), K=int(args.K), max_steps=max_steps)
            best_id = int(costs.argmin())
            traj, repair_log = repair_trajectory(env, trajs[best_id], _v2_sched_cfg(seed), _v2_repair_cfg(), max_steps=max_steps)
            cost = float(sum(-float(item.get("reward", 0.0)) for item in traj))
            method_debug = {"best_k": best_id, "repair": repair_log, "logs": logs[best_id]}
        elif method in {
            "beam_oracle_insertion",
            "deadline_beam_oracle",
            "conservative_deadline_beam",
            "ontime_beam_oracle",
        }:
            cost, traj, method_debug = rollout_v2_scheduler(env, _oracle_sched_cfg(method, seed), max_steps=max_steps)
        elif method == "guarded_ontime_beam":
            cost, traj, method_debug = rollout_guarded_ontime_beam(env, seed, max_steps=max_steps)
        elif method == "policy_accept_ontime_beam":
            cost, traj, method_debug = rollout_policy_accept_ontime_beam(policy, env, seed, max_steps=max_steps)
        elif method == "joint_accept_route_beam":
            cost, traj, method_debug = rollout_joint_accept_route_beam(env, _joint_beam_cfg(args), max_steps=max_steps)
        elif method == "joint_accept_route_beam_guarded":
            cost, traj, method_debug = rollout_joint_guarded_accept_ontime_route(
                env,
                seed,
                _joint_beam_cfg(args, guarded=True),
                max_steps=max_steps,
            )
        elif method == "tail_risk_constrained_joint_beam":
            budget, raw_budget_debug, raw_anchor_actions, raw_anchor_traj = _raw_baseline_budget_for_instance(
                args,
                policy,
                env,
                seed=seed,
                instance_id=idx,
                max_steps=max_steps,
            )
            if not bool(args.tail_risk_allow_anchor_deviation):
                cost = float(raw_budget_debug["raw_cost"])
                traj = raw_anchor_traj
                method_debug = {
                    "done": True,
                    "anchor_locked": True,
                    "anchor_policy": "raw_baseline_best_rollout",
                    "reason": "baseline anchor is used unless a no-regret deviation is explicitly enabled",
                    "budget": budget.to_dict(),
                    "anchor_action_count": len(raw_anchor_actions),
                }
            else:
                anchor_seed = int(raw_budget_debug["raw_seed"])
                np.random.seed(anchor_seed)
                torch.manual_seed(anchor_seed)
                cost, traj, method_debug = rollout_tail_risk_constrained_joint_beam(
                    env,
                    _joint_beam_cfg(args, guarded=True),
                    budget,
                    max_steps=max_steps,
                    anchor_actions=raw_anchor_actions,
                    allow_anchor_deviation=True,
                )
            method_debug["raw_budget_reference"] = raw_budget_debug
        else:
            cost, traj, method_debug = rollout_acceptance_insertion(env, _insertion_cfg(method), max_steps=max_steps)
        summary, orders, drone = analyze_episode(env, traj, model_name=method, instance_id=idx, objective_cost=cost)
        episode_summaries.append(summary)
        order_rows.extend(orders)
        drone_rows.append(drone)
        debug["instances"].append({"instance_id": idx, "cost": cost, "debug": method_debug})
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


def _summary_row(method: str, size: int, overall: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    hard = int(_num(overall.get("hard_constraint_violations")))
    passed = (
        _num(overall.get("acceptance_rate")) + 1e-9 >= _num(raw.get("acceptance_rate"))
        and _num(overall.get("on_time_rate")) + 1e-9 >= _num(raw.get("on_time_rate"))
        and int(_num(overall.get("late_orders"))) <= int(_num(raw.get("late_orders")))
        and _num(overall.get("average_lateness")) <= _num(raw.get("average_lateness")) + 1e-9
        and _num(overall.get("max_lateness")) <= _num(raw.get("max_lateness")) + 1e-9
        and _num(overall.get("total_energy_consumption")) <= 1.03 * _num(raw.get("total_energy_consumption")) + 1e-9
        and _num(overall.get("total_flight_distance")) <= 1.03 * _num(raw.get("total_flight_distance")) + 1e-9
        and hard == 0
    )
    return {
        "method_name": method,
        "eval_instances": int(size),
        "acceptance_rate": _num(overall.get("acceptance_rate")),
        "acceptance_rate_delta_vs_raw_baseline": _num(overall.get("acceptance_rate")) - _num(raw.get("acceptance_rate")),
        "on_time_rate": _num(overall.get("on_time_rate")),
        "on_time_rate_delta_vs_raw_baseline": _num(overall.get("on_time_rate")) - _num(raw.get("on_time_rate")),
        "late_orders": int(_num(overall.get("late_orders"))),
        "late_orders_delta_vs_raw_baseline": int(_num(overall.get("late_orders"))) - int(_num(raw.get("late_orders"))),
        "average_lateness": _num(overall.get("average_lateness")),
        "average_lateness_delta_vs_raw_baseline": _num(overall.get("average_lateness")) - _num(raw.get("average_lateness")),
        "max_lateness": _num(overall.get("max_lateness")),
        "max_lateness_delta_vs_raw_baseline": _num(overall.get("max_lateness")) - _num(raw.get("max_lateness")),
        "total_energy_consumption": _num(overall.get("total_energy_consumption")),
        "total_energy_delta_vs_raw_baseline": _num(overall.get("total_energy_consumption")) - _num(raw.get("total_energy_consumption")),
        "total_flight_distance": _num(overall.get("total_flight_distance")),
        "total_distance_delta_vs_raw_baseline": _num(overall.get("total_flight_distance")) - _num(raw.get("total_flight_distance")),
        "hard_constraint_violations": hard,
        "is_small_data_pass": bool(passed),
        "recommendation": "small_gate_pass" if passed else "do_not_train_yet",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate acceptance-insertion schedulers before ServicePolicy training.")
    p.add_argument("--output-dir", type=str, default="experiments/service_v2/evaluation/small_gate")
    p.add_argument("--baseline-model-path", type=str, default="experiments/frozen_models_20260419/model_main_ep200.pt")
    p.add_argument("--dataset-path", type=str, default="datasets/cvrplib")
    p.add_argument("--eval-split-file", type=str, default="datasets/cvrplib/splits/test.txt")
    p.add_argument("--eval-instances", type=int, default=5)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-progress-every", type=int, default=5)
    p.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    p.add_argument("--decision-mode", type=str, default="legacy", choices=["legacy", "accept_then_route"])
    p.add_argument("--feature-mode", type=str, default="legacy", choices=["legacy", "service_v2"])
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
    p.add_argument("--severe-lateness-threshold", type=float, default=30.0)
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
    p.add_argument("--joint-beam-size", type=int, default=16)
    p.add_argument("--joint-lookahead-depth", type=int, default=3)
    p.add_argument("--joint-candidate-top-k", type=int, default=10)
    p.add_argument("--joint-max-expanded-states", type=int, default=5000)
    p.add_argument("--joint-time-limit-seconds", type=float, default=2.0)
    p.add_argument("--joint-accept-weight", type=float, default=20.0)
    p.add_argument("--joint-on-time-weight", type=float, default=30.0)
    p.add_argument("--joint-late-weight", type=float, default=40.0)
    p.add_argument("--joint-lateness-weight", type=float, default=3.0)
    p.add_argument("--joint-max-lateness-weight", type=float, default=8.0)
    p.add_argument("--joint-energy-weight", type=float, default=0.08)
    p.add_argument("--joint-distance-weight", type=float, default=0.04)
    p.add_argument("--joint-severe-late-weight", type=float, default=0.0)
    p.add_argument("--joint-hard-violation-weight", type=float, default=1000000.0)
    p.add_argument("--joint-enable-dominance-pruning", type=_bool, default=True)
    p.add_argument("--joint-disable-dominance-pruning", dest="joint_enable_dominance_pruning", action="store_false")
    p.add_argument("--joint-guard-max-lateness-factor", type=float, default=1.05)
    p.add_argument("--joint-guard-distance-factor", type=float, default=1.10)
    p.add_argument("--joint-guard-energy-factor", type=float, default=1.10)
    p.add_argument("--joint-guard-abs-lateness-slack", type=float, default=1e-6)
    p.add_argument("--joint-guard-abs-distance-slack", type=float, default=1e-6)
    p.add_argument("--joint-guard-abs-energy-slack", type=float, default=1e-6)
    p.add_argument("--joint-guard-sim-max-steps", type=int, default=96)
    p.add_argument("--tail-risk-max-lateness-ratio", type=float, default=1.02)
    p.add_argument("--tail-risk-avg-lateness-ratio", type=float, default=1.02)
    p.add_argument("--tail-risk-energy-ratio", type=float, default=1.03)
    p.add_argument("--tail-risk-distance-ratio", type=float, default=1.03)
    p.add_argument("--severe-lateness-hard-cap", type=float, default=0.0)
    p.add_argument("--enable-baseline-anchored-budget", type=_bool, default=True)
    p.add_argument("--disable-baseline-anchored-budget", dest="enable_baseline_anchored_budget", action="store_false")
    p.add_argument("--tail-risk-allow-anchor-deviation", action="store_true")
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
    policy = _load_policy(args)
    results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        results[method] = evaluate_method(args, method, policy, out_dir)
    raw = results.get("raw_baseline") or next(iter(results.values()))
    rows = [_summary_row(method, int(args.eval_instances), overall, raw) for method, overall in results.items()]
    write_csv(str(out_dir / "overall_summary.csv"), rows, SUMMARY_FIELDS)
    write_json(str(out_dir / "overall_summary.json"), {"rows": rows})
    lines = [
        "# Acceptance Insertion Small-Gate Report",
        "",
        f"- eval_instances: `{args.eval_instances}`",
        f"- decision_mode: `{args.decision_mode}`",
        f"- feature_mode: `{args.feature_mode}`",
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
