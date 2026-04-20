from __future__ import annotations
"""HGAT-POMO 评估入口，支持轻量启发式基线对比。"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch

# Allow both:
# 1) python -m src.main_eval
# 2) python src/main_eval.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.instance_gen import (
    REQUEST_DELIVERY,
    REQUEST_PICKUP,
    make_instance_from_coord_demand,
    make_random_instance,
)
from src.env.open_data_loader import (
    load_cvrplib_instances_filtered,
    read_instance_name_list,
    sample_open_vrp_base,
)
from src.env.td_env import TruckDroneRendezvousEnv, EnvConfig
from src.models.policy import HGATPolicy
from src.rl.pomo_rollout import (
    pomo_rollout,
    _init_episode_stats,
    _update_episode_stats,
    _finalize_episode_stats,
)
from src.baselines.local_search import choose_truck_next_local_search


OPS_METRIC_NAMES = [
    "accept_rate",
    "reject_rate",
    "on_time_rate",
    "avg_lateness",
    "total_revenue",
    "total_energy",
    "depot_return_count",
    "drone_participation_rate",
    "pickup_service_rate",
    "delivery_service_rate",
]


def _safe_div(num: float, den: float) -> float:
    den = float(den)
    if abs(den) <= 1e-9:
        return 0.0
    return float(num) / den


def _raw_stats_to_ops(raw: dict) -> dict:
    served_total = float(raw.get("served_total", 0.0))
    return {
        "accept_rate": _safe_div(raw.get("accepted_dynamic", 0.0), raw.get("dynamic_total", 0.0)),
        "reject_rate": _safe_div(raw.get("rejected_dynamic", 0.0), raw.get("dynamic_total", 0.0)),
        "on_time_rate": _safe_div(raw.get("on_time_count", 0.0), served_total),
        "avg_lateness": _safe_div(raw.get("total_lateness", 0.0), served_total),
        "total_revenue": float(raw.get("revenue_total", 0.0)),
        "total_energy": float(raw.get("energy_total", 0.0)),
        "depot_return_count": float(raw.get("depot_return_count", 0.0)),
        "drone_participation_rate": _safe_div(raw.get("drone_dispatch_count", 0.0), raw.get("route_step_count", 0.0)),
        "pickup_service_rate": _safe_div(raw.get("served_pickup", 0.0), raw.get("pickup_total", 0.0)),
        "delivery_service_rate": _safe_div(raw.get("served_delivery", 0.0), raw.get("delivery_total", 0.0)),
    }


def _init_ops_values() -> dict:
    return {name: [] for name in OPS_METRIC_NAMES}


def _append_ops_values(values: dict, raw_stats: dict) -> None:
    cur = _raw_stats_to_ops(raw_stats)
    for name in OPS_METRIC_NAMES:
        values[name].append(float(cur[name]))


def _make_ops_summary(values: dict) -> dict:
    out = {}
    for name in OPS_METRIC_NAMES:
        arr = values.get(name, [])
        if len(arr) == 0:
            out[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            continue
        m, s, mn, mx = summary(arr)
        out[name] = {"mean": m, "std": s, "min": mn, "max": mx}
    return out


def _print_ops_summary_line(prefix: str, ss: dict) -> None:
    print(
        f"{prefix} accept={ss['accept_rate']['mean']:.3f} "
        f"reject={ss['reject_rate']['mean']:.3f} "
        f"ontime={ss['on_time_rate']['mean']:.3f} "
        f"avg_late={ss['avg_lateness']['mean']:.3f}"
    )
    print(
        f"{prefix} revenue={ss['total_revenue']['mean']:.3f} "
        f"energy={ss['total_energy']['mean']:.3f} "
        f"depot_ret={ss['depot_return_count']['mean']:.3f} "
        f"drone_part={ss['drone_participation_rate']['mean']:.3f}"
    )
    print(
        f"{prefix} pickup_srv={ss['pickup_service_rate']['mean']:.3f} "
        f"delivery_srv={ss['delivery_service_rate']['mean']:.3f}"
    )


def _feasible_js(env: TruckDroneRendezvousEnv) -> np.ndarray:
    """返回当前状态下卡车可选的下一跳节点。"""
    mask = env.get_masks()["truck_mask"]
    return np.where(mask > 0)[0]


def _estimate_profit_accept_score(
    env: TruckDroneRendezvousEnv,
    obs: dict,
    req: int,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    i = int(obs["i"])
    t = float(obs["t"])
    truck_load = float(obs.get("truck_load", 0.0))
    req_type = int(env.request_type[req])
    req_weight = float(max(0.0, env.demand[req]))
    cap = max(1e-6, float(env.cfg.truck_capacity))

    is_loaded_delivery = req_type == REQUEST_DELIVERY and int(env.state["loaded"][req]) > 0
    if req_type == REQUEST_DELIVERY and not is_loaded_delivery:
        if i == 0:
            dist_est = float(env.dist_mat[0, req])
        else:
            dist_est = float(env.dist_mat[i, 0] + env.dist_mat[0, req])
    else:
        dist_est = float(env.dist_mat[i, req])

    travel_time_est = dist_est / max(1e-6, float(env.cfg.vT)) + float(env.cfg.sT)
    payload_ratio = float(np.clip(truck_load / cap, 0.0, 1.0))
    truck_energy_per_dist = float(env.cfg.truck_energy_per_dist) * (
        1.0 + float(env.cfg.payload_energy_factor) * payload_ratio
    )
    base_operating_cost = float(env.cfg.time_cost_weight) * travel_time_est + float(env.cfg.energy_cost_weight) * (
        truck_energy_per_dist * dist_est
    )

    due = float(env.due[req])
    lateness = 0.0 if not np.isfinite(due) else max(0.0, (t + travel_time_est) - due)
    lateness_cost = float(env.cfg.lateness_penalty) * lateness

    load_pressure_cost = 0.0
    if req_type == REQUEST_PICKUP:
        overflow = max(0.0, (truck_load + req_weight) - cap)
        load_pressure_cost = overflow / cap

    revenue = float(env.revenue[req]) * float(env.cfg.revenue_scale)
    reject_bias = float(env.cfg.reject_penalty)
    incremental_cost = float(alpha) * base_operating_cost + float(beta) * lateness_cost + float(gamma) * load_pressure_cost
    return revenue + reject_bias - incremental_cost


def _choose_pending_decision(
    env: TruckDroneRendezvousEnv,
    obs: dict,
    mode: str,
    *,
    profit_accept_alpha: float = 1.0,
    profit_accept_beta: float = 1.0,
    profit_accept_gamma: float = 1.0,
    profit_accept_margin: float = 0.0,
) -> int | None:
    req = int(obs.get("current_decision_request", -1))
    if req <= 0:
        return None
    if mode != "profit_accept":
        return req
    score = _estimate_profit_accept_score(
        env=env,
        obs=obs,
        req=req,
        alpha=float(profit_accept_alpha),
        beta=float(profit_accept_beta),
        gamma=float(profit_accept_gamma),
    )
    return int(req) if score >= float(profit_accept_margin) else 0


def _choose_j_nearest(env: TruckDroneRendezvousEnv, obs) -> int:
    """truck_only 基线：选最近可行点，优先非仓库节点。"""
    i = int(obs["i"])
    js = _feasible_js(env)
    if js.size == 0:
        return 0
    non_depot = js[js != 0]
    cand = non_depot if non_depot.size > 0 else js
    dist = env.dist_mat[i, cand]
    return int(cand[int(np.argmin(dist))])


def _choose_j_edd(env: TruckDroneRendezvousEnv, obs) -> int:
    """j 的启发式：最早截止期优先，距离作为平局打破。"""
    i = int(obs["i"])
    t = float(obs["t"])
    js = _feasible_js(env)
    if js.size == 0:
        return 0
    non_depot = js[js != 0]
    if non_depot.size == 0:
        if i in js:
            return int(i)
        return int(js[0])

    due = env.due[non_depot].astype(np.float64)
    finite = np.isfinite(due)
    if finite.any():
        cand = non_depot[finite]
        due_c = due[finite]
        min_due = due_c.min()
        tie = cand[np.isclose(due_c, min_due)]
        if tie.size == 1:
            return int(tie[0])
        dist = env.dist_mat[i, tie]
        return int(tie[int(np.argmin(dist))])

    # no deadlines -> nearest
    dist = env.dist_mat[i, non_depot]
    return int(non_depot[int(np.argmin(dist))])


def _choose_k_random(env: TruckDroneRendezvousEnv, j: int) -> int:
    """若可行则 70% 概率派无人机，否则 no-drone。"""
    dmask = env.get_masks(j=j)["drone_mask"]
    ks = np.where(dmask > 0)[0]
    if len(ks) > 0 and np.random.rand() < 0.7:
        return int(np.random.choice(ks))
    return env.K_NONE


def _choose_k_heuristic(env: TruckDroneRendezvousEnv, obs, j: int) -> int:
    """k 的启发式：综合截止裕量(slack)与飞行时长。"""
    t = float(obs["t"])
    i = int(obs["i"])
    dmask = env.get_masks(j=j)["drone_mask"]
    ks = np.where(dmask > 0)[0]
    if ks.size == 0:
        return env.K_NONE

    due = env.due[ks].astype(np.float64)
    slack = due - t
    slack[~np.isfinite(slack)] = 1e9

    drone_time = np.array(
        [env._tau_drone(i, int(k), j) + float(env.cfg.sD) for k in ks],
        dtype=np.float64,
    )
    score = slack + 0.2 * drone_time
    return int(ks[int(np.argmin(score))])


def baseline_rollout(
    env: TruckDroneRendezvousEnv,
    K: int = 8,
    max_steps: int = 256,
    mode: str = "random",  # random | truck_only | heuristic | local_search_truck | profit_accept
    return_stats: bool = False,
    profit_accept_alpha: float = 1.0,
    profit_accept_beta: float = 1.0,
    profit_accept_gamma: float = 1.0,
    profit_accept_margin: float = 0.0,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """
    Baseline rollout:
      - random: random feasible j and random feasible k/none
      - truck_only: nearest feasible j, always k=none
      - heuristic: EDD-like j + urgency-aware k
      - local_search_truck: online truck-only replan with 2-opt improvement
      - profit_accept: dynamic decision uses revenue-minus-cost threshold, routing uses heuristic
    Return total objective cost for each of K trials.
    """
    costs = []
    stats_list = [] if return_stats else None
    for _ in range(K):
        # 同一实例上，基线的每次试验都用独立环境副本。
        e = env.copy()
        obs = e.reset()
        episode_stats = _init_episode_stats(e) if return_stats else None
        total_cost = 0.0
        done = False
        for _ in range(max_steps):
            decision_action = _choose_pending_decision(
                e,
                obs,
                mode=mode,
                profit_accept_alpha=profit_accept_alpha,
                profit_accept_beta=profit_accept_beta,
                profit_accept_gamma=profit_accept_gamma,
                profit_accept_margin=profit_accept_margin,
            )
            if decision_action is not None:
                j = int(decision_action)
                k = e.K_NONE
            elif mode == "random":
                js = _feasible_js(e)
                j = int(np.random.choice(js)) if js.size > 0 else 0
                k = _choose_k_random(e, j)
            elif mode == "truck_only":
                j = _choose_j_nearest(e, obs)
                k = e.K_NONE
            elif mode == "heuristic":
                j = _choose_j_edd(e, obs)
                k = _choose_k_heuristic(e, obs, j)
            elif mode == "local_search_truck":
                j = choose_truck_next_local_search(e, obs)
                k = e.K_NONE
            elif mode == "profit_accept":
                j = _choose_j_edd(e, obs)
                k = _choose_k_heuristic(e, obs, j)
            else:
                raise ValueError(f"Unknown baseline mode: {mode}")

            obs2, r, done, info = e.step((k, j))
            total_cost += float(-r)
            if return_stats and episode_stats is not None:
                _update_episode_stats(episode_stats, e, obs, obs2, (k, j), info)
            obs = obs2
            if done:
                break

        if not done:
            total_cost += 1000.0  # timeout penalty
        if return_stats and episode_stats is not None and stats_list is not None:
            _finalize_episode_stats(episode_stats, e, obs, done=done)
            stats_list.append(episode_stats)
        costs.append(total_cost)
    costs_arr = np.array(costs, dtype=np.float32)
    if return_stats and stats_list is not None:
        return costs_arr, stats_list
    return costs_arr


def summary(x_list):
    """返回一组标量的 (mean, std, min, max)。"""
    x = np.array(x_list, dtype=np.float32)
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def _make_summary_dict(values):
    s_best = summary(values["best"])
    s_mean = summary(values["mean"])
    s_worst = summary(values["worst"])
    return {
        "best": {"mean": s_best[0], "std": s_best[1], "min": s_best[2], "max": s_best[3]},
        "mean": {"mean": s_mean[0], "std": s_mean[1], "min": s_mean[2], "max": s_mean[3]},
        "worst": {"mean": s_worst[0], "std": s_worst[1], "min": s_worst[2], "max": s_worst[3]},
    }


def _print_summary_line(prefix: str, ss: dict) -> None:
    b = ss["best"]
    m = ss["mean"]
    w = ss["worst"]
    print(f"{prefix} best(K): mean={b['mean']:.3f}, std={b['std']:.3f}, best(min)={b['min']:.3f}, worst(max)={b['max']:.3f}")
    print(f"{prefix} mean(K): mean={m['mean']:.3f}, std={m['std']:.3f}, best(min)={m['min']:.3f}, worst(max)={m['max']:.3f}")
    print(f"{prefix} worst(K): mean={w['mean']:.3f}, std={w['std']:.3f}, best(min)={w['min']:.3f}, worst(max)={w['max']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HGAT-POMO policy.")
    parser.add_argument("--model-path", type=str, default="policy.pt")
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k-nn-orders", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--edge-mode", type=str, default="static", choices=["static", "road"])
    parser.add_argument("--time-dependent", action="store_true")
    parser.add_argument("--peak-after-served-ratio", type=float, default=0.5)

    # data generation
    parser.add_argument("--coord-scale", type=float, default=10.0)
    parser.add_argument("--release-mode", type=str, default="batches", choices=["batches", "uniform", "poisson"])
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--max-release", type=float, default=10.0)
    parser.add_argument("--poisson-rate", type=float, default=1.0)
    parser.add_argument("--tw-mode", type=str, default="relative", choices=["relative", "mixed", "none"])
    parser.add_argument("--tw-slack-low", type=float, default=4.0)
    parser.add_argument("--tw-slack-high", type=float, default=14.0)
    parser.add_argument("--tw-active-prob", type=float, default=0.8)
    parser.add_argument("--scheduled-ratio", type=float, default=0.5)
    parser.add_argument("--dynamic-pickup-ratio", type=float, default=1.0)
    parser.add_argument("--response-slack-low", type=float, default=0.25)
    parser.add_argument("--response-slack-high", type=float, default=1.0)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Path to open-source CVRPLIB .vrp file or a directory of .vrp files.",
    )
    parser.add_argument("--dataset-format", type=str, default="cvrplib", choices=["cvrplib"])
    parser.add_argument(
        "--dataset-split-file",
        type=str,
        default="",
        help="Optional txt list of instance names for eval split (one name per line).",
    )
    parser.add_argument("--dataset-demand-scale", type=float, default=1.0)
    parser.add_argument("--dataset-no-normalize-coords", action="store_true")
    parser.add_argument("--eval-seed", type=int, default=0, help="Controls deterministic eval instance generation.")

    # heterogeneity + dynamicity
    parser.add_argument("--vT", type=float, default=1.0)
    parser.add_argument("--vD", type=float, default=1.5)
    parser.add_argument("--QD", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=6.0)
    parser.add_argument("--truck-capacity", type=float, default=3.0)
    parser.add_argument("--truck-service-time", type=float, default=0.05)
    parser.add_argument("--drone-service-time", type=float, default=0.03)
    parser.add_argument("--depot-service-time", type=float, default=0.10)
    parser.add_argument("--traffic-sigma", type=float, default=0.15)
    parser.add_argument("--lateness-penalty", type=float, default=0.5)
    parser.add_argument("--reject-penalty", type=float, default=0.5)
    parser.add_argument("--overtime-penalty", type=float, default=1.0)
    parser.add_argument("--time-cost-weight", type=float, default=1.0)
    parser.add_argument("--energy-cost-weight", type=float, default=0.2)
    parser.add_argument("--soc-init", type=float, default=1.0)
    parser.add_argument("--soc-reserve", type=float, default=0.10)
    parser.add_argument("--energy-per-dist", type=float, default=0.08)
    parser.add_argument("--truck-energy-per-dist", type=float, default=0.04)
    parser.add_argument("--payload-energy-factor", type=float, default=0.4)
    parser.add_argument("--recharge-rate", type=float, default=0.25)
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

    parser.add_argument("--metrics-json", type=str, default="")
    parser.add_argument("--traj-out", type=str, default="")
    parser.add_argument("--no-store-traj", action="store_true")
    parser.add_argument(
        "--extra-baselines",
        action="store_true",
        help="Evaluate truck_only / heuristic / local_search_truck / profit_accept baselines.",
    )
    parser.add_argument("--profit-accept-alpha", type=float, default=1.0)
    parser.add_argument("--profit-accept-beta", type=float, default=1.0)
    parser.add_argument("--profit-accept-gamma", type=float, default=1.0)
    parser.add_argument("--profit-accept-margin", type=float, default=0.0)
    parser.add_argument("--ablate-no-accept-reject", action="store_true")
    parser.add_argument("--ablate-no-pickup-capacity", action="store_true")
    parser.add_argument("--ablate-no-time-traffic", action="store_true")

    args = parser.parse_args()

    device = torch.device("cpu")
    print("Using device:", device)
    print(f"Edge mode: {args.edge_mode} | time_dependent={args.time_dependent}")

    max_steps = 8 * (args.N + 1)
    store_traj = not args.no_store_traj

    cfg = EnvConfig(
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
    if args.ablate_no_pickup_capacity:
        cfg.truck_capacity = 1e6
        cfg.payload_energy_factor = 0.0
    if args.ablate_no_time_traffic:
        cfg.time_dependent = False
    print(
        f"Ablation flags: no_accept_reject={args.ablate_no_accept_reject} "
        f"no_pickup_capacity={args.ablate_no_pickup_capacity} "
        f"no_time_traffic={args.ablate_no_time_traffic}"
    )
    print(
        f"Effective env: edge_mode={cfg.edge_mode} time_dependent={cfg.time_dependent} "
        f"truck_capacity={cfg.truck_capacity:.1f}"
    )

    open_instances = None
    use_open_dataset = len(args.dataset_path.strip()) > 0
    if use_open_dataset:
        if args.dataset_format != "cvrplib":
            raise ValueError(f"Unsupported dataset_format={args.dataset_format}")
        include_names = None
        if args.dataset_split_file.strip():
            include_names = read_instance_name_list(args.dataset_split_file.strip())
        open_instances = load_cvrplib_instances_filtered(
            args.dataset_path.strip(),
            include_names=include_names,
        )
        open_instances = [x for x in open_instances if x.n_customers >= int(args.N)]
        if len(open_instances) == 0:
            raise ValueError(
                "No CVRPLIB instance has enough customers for current --N. "
                "Please reduce --N or use larger instance files."
            )
        min_n = min(x.n_customers for x in open_instances)
        max_n = max(x.n_customers for x in open_instances)
        split_msg = args.dataset_split_file.strip() if args.dataset_split_file.strip() else "<all>"
        print(
            f"Using open dataset: {len(open_instances)} instances "
            f"(customers range: {min_n}-{max_n}), demand_scale={args.dataset_demand_scale}, "
            f"normalize_coords={not args.dataset_no_normalize_coords}, split={split_msg}"
        )

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
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint. This branch expects updated HGAT inputs "
            "(order feature dim 12 with revenue feature). Please retrain or "
            "use a checkpoint from the new feature schema."
        ) from exc
    policy.eval()

    model_values = {"best": [], "mean": [], "worst": []}
    model_ops_values = _init_ops_values()
    baseline_values = {"random": {"best": [], "mean": [], "worst": []}}
    baseline_ops_values = {"random": _init_ops_values()}
    if args.extra_baselines:
        baseline_values["truck_only"] = {"best": [], "mean": [], "worst": []}
        baseline_values["heuristic"] = {"best": [], "mean": [], "worst": []}
        baseline_values["local_search_truck"] = {"best": [], "mean": [], "worst": []}
        baseline_values["profit_accept"] = {"best": [], "mean": [], "worst": []}
        baseline_ops_values["truck_only"] = _init_ops_values()
        baseline_ops_values["heuristic"] = _init_ops_values()
        baseline_ops_values["local_search_truck"] = _init_ops_values()
        baseline_ops_values["profit_accept"] = _init_ops_values()

    best_trajs_all = []

    t0 = time.time()
    with torch.no_grad():
        for idx in range(1, args.n_instances + 1):
            inst_seed = int(args.eval_seed) * 100000 + idx
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
            if args.ablate_no_accept_reject:
                release = release.copy()
                release[1:] = 0.0
                meta["is_dynamic"] = meta["is_dynamic"].copy()
                meta["is_dynamic"][1:] = 0
                meta["decision_deadline"] = meta["decision_deadline"].copy()
                meta["decision_deadline"][1:] = 0.0
            if args.ablate_no_pickup_capacity:
                req_type = meta["request_type"].copy()
                req_type[req_type == REQUEST_PICKUP] = REQUEST_DELIVERY
                meta["request_type"] = req_type
            env = TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=cfg, seed=inst_seed, **meta)

            returns, _, trajs, _, rollout_stats = pomo_rollout(
                policy,
                env,
                K=args.K,
                max_steps=max_steps,
                store_traj=store_traj,
                collect_stats=True,
            )
            # returns 是奖励（负成本），这里转成目标成本（越小越好）。
            costs = (-returns).numpy()  # (K,), smaller is better
            model_values["best"].append(float(costs.min()))
            model_values["mean"].append(float(costs.mean()))
            model_values["worst"].append(float(costs.max()))
            best_id = int(costs.argmin())
            _append_ops_values(model_ops_values, rollout_stats[best_id])

            if store_traj:
                assert trajs is not None
                best_trajs_all.append((idx, float(costs[best_id]), trajs[best_id]))

            rcost, rstats = baseline_rollout(
                env,
                K=args.K,
                max_steps=max_steps,
                mode="random",
                return_stats=True,
            )
            baseline_values["random"]["best"].append(float(rcost.min()))
            baseline_values["random"]["mean"].append(float(rcost.mean()))
            baseline_values["random"]["worst"].append(float(rcost.max()))
            _append_ops_values(baseline_ops_values["random"], rstats[int(rcost.argmin())])

            if args.extra_baselines:
                tcost, tstats = baseline_rollout(
                    env,
                    K=args.K,
                    max_steps=max_steps,
                    mode="truck_only",
                    return_stats=True,
                )
                hcost, hstats = baseline_rollout(
                    env,
                    K=args.K,
                    max_steps=max_steps,
                    mode="heuristic",
                    return_stats=True,
                )
                lcost, lstats = baseline_rollout(
                    env,
                    K=args.K,
                    max_steps=max_steps,
                    mode="local_search_truck",
                    return_stats=True,
                )
                pcost, pstats = baseline_rollout(
                    env,
                    K=args.K,
                    max_steps=max_steps,
                    mode="profit_accept",
                    return_stats=True,
                    profit_accept_alpha=args.profit_accept_alpha,
                    profit_accept_beta=args.profit_accept_beta,
                    profit_accept_gamma=args.profit_accept_gamma,
                    profit_accept_margin=args.profit_accept_margin,
                )
                baseline_values["truck_only"]["best"].append(float(tcost.min()))
                baseline_values["truck_only"]["mean"].append(float(tcost.mean()))
                baseline_values["truck_only"]["worst"].append(float(tcost.max()))
                baseline_values["heuristic"]["best"].append(float(hcost.min()))
                baseline_values["heuristic"]["mean"].append(float(hcost.mean()))
                baseline_values["heuristic"]["worst"].append(float(hcost.max()))
                baseline_values["local_search_truck"]["best"].append(float(lcost.min()))
                baseline_values["local_search_truck"]["mean"].append(float(lcost.mean()))
                baseline_values["local_search_truck"]["worst"].append(float(lcost.max()))
                baseline_values["profit_accept"]["best"].append(float(pcost.min()))
                baseline_values["profit_accept"]["mean"].append(float(pcost.mean()))
                baseline_values["profit_accept"]["worst"].append(float(pcost.max()))
                _append_ops_values(baseline_ops_values["truck_only"], tstats[int(tcost.argmin())])
                _append_ops_values(baseline_ops_values["heuristic"], hstats[int(hcost.argmin())])
                _append_ops_values(baseline_ops_values["local_search_truck"], lstats[int(lcost.argmin())])
                _append_ops_values(baseline_ops_values["profit_accept"], pstats[int(pcost.argmin())])

            if idx % 10 == 0:
                msg = (
                    f"[{idx:03d}/{args.n_instances}] "
                    f"model_best={model_values['best'][-1]:.2f} model_mean={model_values['mean'][-1]:.2f} | "
                    f"rand_best={baseline_values['random']['best'][-1]:.2f} "
                    f"rand_mean={baseline_values['random']['mean'][-1]:.2f}"
                )
                if args.extra_baselines:
                    msg += (
                        f" | truck_best={baseline_values['truck_only']['best'][-1]:.2f} "
                        f"heur_best={baseline_values['heuristic']['best'][-1]:.2f} "
                        f"ls_best={baseline_values['local_search_truck']['best'][-1]:.2f} "
                        f"profit_best={baseline_values['profit_accept']['best'][-1]:.2f}"
                    )
                print(msg)

    print("\n=== Evaluation Summary (Objective Cost, smaller is better) ===")
    model_summary = _make_summary_dict(model_values)
    _print_summary_line("MODEL", model_summary)

    baseline_summary = {}
    for name, values in baseline_values.items():
        ss = _make_summary_dict(values)
        baseline_summary[name] = ss
        _print_summary_line(name.upper(), ss)

    print("\n=== Operations Summary (best-of-K, higher on_time is better) ===")
    model_ops_summary = _make_ops_summary(model_ops_values)
    _print_ops_summary_line("MODEL", model_ops_summary)

    baseline_ops_summary = {}
    for name, values in baseline_ops_values.items():
        ss = _make_ops_summary(values)
        baseline_ops_summary[name] = ss
        _print_ops_summary_line(name.upper(), ss)

    traj_path = args.traj_out.strip() if args.traj_out else f"eval_trajs_N{args.N}.txt"
    if store_traj:
        with open(traj_path, "w", encoding="utf-8") as f:
            for inst_id, best_cost, traj in best_trajs_all:
                f.write(f"instance={inst_id} best_cost={best_cost:.6f}\n")
                for step_id, item in enumerate(traj):
                    action = item["action"]
                    info = item["info"]
                    if action == ("TIMEOUT",):
                        f.write(f"  step={step_id:03d} TIMEOUT\n")
                        continue
                    k, j = action
                    f.write(
                        f"  step={step_id:03d} i={info.get('i')} j={j} k={k} "
                        f"dt={info.get('dt', 0.0):.6f} truck={info.get('truck_time', 0.0):.6f} "
                        f"drone={info.get('drone_time', 0.0):.6f} late={info.get('lateness', 0.0):.6f} "
                        f"soc={info.get('soc_prev', 0.0):.3f}->{info.get('soc_next', 0.0):.3f}\n"
                    )
                f.write("\n")
        print(f"\nSaved best trajectories to {traj_path}")

    elapsed = float(time.time() - t0)
    metrics = {
        "model_path": args.model_path,
        "N": int(args.N),
        "K": int(args.K),
        "n_instances": int(args.n_instances),
        "eval_seed": int(args.eval_seed),
        "elapsed_sec": elapsed,
        "model": model_summary,
        "model_ops": model_ops_summary,
        "baselines": baseline_summary,
        "baseline_ops": baseline_ops_summary,
        "profit_accept_cfg": {
            "alpha": float(args.profit_accept_alpha),
            "beta": float(args.profit_accept_beta),
            "gamma": float(args.profit_accept_gamma),
            "margin": float(args.profit_accept_margin),
        },
        "ablations": {
            "no_accept_reject": bool(args.ablate_no_accept_reject),
            "no_pickup_capacity": bool(args.ablate_no_pickup_capacity),
            "no_time_traffic": bool(args.ablate_no_time_traffic),
        },
        "dataset": {
            "path": args.dataset_path,
            "format": args.dataset_format,
            "split_file": args.dataset_split_file,
            "demand_scale": float(args.dataset_demand_scale),
            "normalize_coords": bool(not args.dataset_no_normalize_coords),
        },
    }
    if "random" in baseline_summary:
        # Backward compatibility for existing scripts
        metrics["random"] = baseline_summary["random"]
    if "random" in baseline_ops_summary:
        metrics["random_ops"] = baseline_ops_summary["random"]

    if args.metrics_json:
        metrics_dir = os.path.dirname(args.metrics_json)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics json to {args.metrics_json}")

    print(f"Total eval time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
