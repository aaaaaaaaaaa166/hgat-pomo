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

from src.env.instance_gen import make_random_instance
from src.env.td_env import TruckDroneRendezvousEnv, EnvConfig
from src.models.policy import HGATPolicy
from src.rl.pomo_rollout import pomo_rollout
from src.baselines.local_search import choose_truck_next_local_search


def _feasible_js(env: TruckDroneRendezvousEnv) -> np.ndarray:
    """返回当前状态下卡车可选的下一跳节点。"""
    mask = env.get_masks()["truck_mask"]
    return np.where(mask > 0)[0]


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
    mode: str = "random",  # random | truck_only | heuristic | local_search_truck
) -> np.ndarray:
    """
    Baseline rollout:
      - random: random feasible j and random feasible k/none
      - truck_only: nearest feasible j, always k=none
      - heuristic: EDD-like j + urgency-aware k
      - local_search_truck: online truck-only replan with 2-opt improvement
    Return total objective cost for each of K trials.
    """
    costs = []
    for _ in range(K):
        # 同一实例上，基线的每次试验都用独立环境副本。
        e = env.copy()
        obs = e.reset()
        total_cost = 0.0
        done = False
        for _ in range(max_steps):
            if mode == "random":
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
            else:
                raise ValueError(f"Unknown baseline mode: {mode}")

            obs, r, done, _ = e.step((k, j))
            total_cost += float(-r)
            if done:
                break

        if not done:
            total_cost += 1000.0  # timeout penalty
        costs.append(total_cost)
    return np.array(costs, dtype=np.float32)


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

    # heterogeneity + dynamicity
    parser.add_argument("--vT", type=float, default=1.0)
    parser.add_argument("--vD", type=float, default=1.5)
    parser.add_argument("--QD", type=float, default=1.0)
    parser.add_argument("--B", type=float, default=6.0)
    parser.add_argument("--traffic-sigma", type=float, default=0.15)
    parser.add_argument("--lateness-penalty", type=float, default=0.5)
    parser.add_argument("--soc-init", type=float, default=1.0)
    parser.add_argument("--soc-reserve", type=float, default=0.10)
    parser.add_argument("--energy-per-dist", type=float, default=0.08)
    parser.add_argument("--recharge-rate", type=float, default=0.25)
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
    parser.add_argument("--extra-baselines", action="store_true", help="Evaluate truck_only and heuristic baselines.")

    args = parser.parse_args()

    device = torch.device("cpu")
    print("Using device:", device)
    print(f"Edge mode: {args.edge_mode} | time_dependent={args.time_dependent}")

    max_steps = 5 * (args.N + 1)
    store_traj = not args.no_store_traj

    cfg = EnvConfig(
        vT=args.vT,
        vD=args.vD,
        QD=args.QD,
        B=args.B,
        sT=0.0,
        sD=0.0,
        allow_wait=True,
        idle_to_next_release=True,
        traffic_sigma=args.traffic_sigma,
        lateness_penalty=args.lateness_penalty,
        soc_init=args.soc_init,
        soc_min_reserve=args.soc_reserve,
        energy_per_dist=args.energy_per_dist,
        recharge_rate=args.recharge_rate,
        edge_mode=args.edge_mode,
        time_dependent=args.time_dependent,
        peak_after_served_ratio=args.peak_after_served_ratio,
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

    policy = HGATPolicy(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        k_nn_orders=args.k_nn_orders,
        num_encoder_layers=args.encoder_layers,
        tanh_clipping=args.tanh_clipping,
        temperature=args.temperature,
    ).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    policy.eval()

    model_values = {"best": [], "mean": [], "worst": []}
    baseline_values = {"random": {"best": [], "mean": [], "worst": []}}
    if args.extra_baselines:
        baseline_values["truck_only"] = {"best": [], "mean": [], "worst": []}
        baseline_values["heuristic"] = {"best": [], "mean": [], "worst": []}
        baseline_values["local_search_truck"] = {"best": [], "mean": [], "worst": []}

    best_trajs_all = []

    t0 = time.time()
    with torch.no_grad():
        for idx in range(1, args.n_instances + 1):
            coord, release, demand, due = make_random_instance(
                N=args.N,
                seed=idx,
                coord_scale=args.coord_scale,
                release_mode=args.release_mode,
                n_batches=args.n_batches,
                max_release=args.max_release,
                poisson_rate=args.poisson_rate,
                tw_mode=args.tw_mode,
                tw_slack_low=args.tw_slack_low,
                tw_slack_high=args.tw_slack_high,
                tw_active_prob=args.tw_active_prob,
                return_due=True,
            )
            env = TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=cfg, seed=idx)

            returns, _, trajs, _ = pomo_rollout(
                policy,
                env,
                K=args.K,
                max_steps=max_steps,
                store_traj=store_traj,
            )
            # returns 是奖励（负成本），这里转成目标成本（越小越好）。
            costs = (-returns).numpy()  # (K,), smaller is better
            model_values["best"].append(float(costs.min()))
            model_values["mean"].append(float(costs.mean()))
            model_values["worst"].append(float(costs.max()))

            if store_traj:
                best_id = int(costs.argmin())
                assert trajs is not None
                best_trajs_all.append((idx, float(costs[best_id]), trajs[best_id]))

            rcost = baseline_rollout(env, K=args.K, max_steps=max_steps, mode="random")
            baseline_values["random"]["best"].append(float(rcost.min()))
            baseline_values["random"]["mean"].append(float(rcost.mean()))
            baseline_values["random"]["worst"].append(float(rcost.max()))

            if args.extra_baselines:
                tcost = baseline_rollout(env, K=args.K, max_steps=max_steps, mode="truck_only")
                hcost = baseline_rollout(env, K=args.K, max_steps=max_steps, mode="heuristic")
                lcost = baseline_rollout(env, K=args.K, max_steps=max_steps, mode="local_search_truck")
                baseline_values["truck_only"]["best"].append(float(tcost.min()))
                baseline_values["truck_only"]["mean"].append(float(tcost.mean()))
                baseline_values["truck_only"]["worst"].append(float(tcost.max()))
                baseline_values["heuristic"]["best"].append(float(hcost.min()))
                baseline_values["heuristic"]["mean"].append(float(hcost.mean()))
                baseline_values["heuristic"]["worst"].append(float(hcost.max()))
                baseline_values["local_search_truck"]["best"].append(float(lcost.min()))
                baseline_values["local_search_truck"]["mean"].append(float(lcost.mean()))
                baseline_values["local_search_truck"]["worst"].append(float(lcost.max()))

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
                        f"ls_best={baseline_values['local_search_truck']['best'][-1]:.2f}"
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
        "elapsed_sec": elapsed,
        "model": model_summary,
        "baselines": baseline_summary,
    }
    if "random" in baseline_summary:
        # Backward compatibility for existing scripts
        metrics["random"] = baseline_summary["random"]

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
