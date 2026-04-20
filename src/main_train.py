"""HGAT-POMO 训练入口（卡车-无人机会合配送）。

脚本采用在线数据生成（每个 epoch 采新实例），
并用 REINFORCE + POMO baseline 做策略优化。
"""

# src/main_train.py
from __future__ import annotations
import gc
import os
import sys
import random
import argparse
import numpy as np
import torch

# Allow both:
# 1) python -m src.main_train
# 2) python src/main_train.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.instance_gen import make_instance_from_coord_demand, make_random_instance
from src.env.open_data_loader import (
    load_cvrplib_instances_filtered,
    read_instance_name_list,
    sample_open_vrp_base,
)
from src.env.td_env import TruckDroneRendezvousEnv, EnvConfig
from src.models.policy import HGATPolicy
from src.rl.pomo_rollout import pomo_rollout


def set_seed(seed: int = 0):
    """统一随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """设备选择优先级：CUDA > DirectML(Windows AMD) > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    # AMD on Windows path: try DirectML, then fall back to CPU if unsupported.
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def smoke_test_policy_device(device: torch.device, cfg: EnvConfig) -> bool:
    """做一次最小前向，验证当前加速后端是否兼容。

    部分 PyG 算子对后端有要求；先做烟雾测试可避免训练中途失败。
    """
    try:
        coord, release, demand, due, meta = make_random_instance(
            N=8,
            seed=123,
            coord_scale=10.0,
            release_mode="batches",
            n_batches=2,
            max_release=3.0,
            tw_mode="relative",
            tw_slack_low=2.0,
            tw_slack_high=6.0,
            return_due=True,
        )
        env = TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=cfg, seed=123, **meta)
        policy = HGATPolicy(hidden_dim=64, heads=2, dropout=0.0, k_nn_orders=4).to(device)
        obs = env.reset()
        policy.forward_step(env, obs)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Train HGAT-POMO policy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k-nn-orders", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.0)
    parser.add_argument("--save-path", type=str, default="policy.pt")
    parser.add_argument("--init-model-path", type=str, default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
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
        help="Optional txt list of instance names for train split (one name per line).",
    )
    parser.add_argument("--dataset-demand-scale", type=float, default=1.0)
    parser.add_argument("--dataset-no-normalize-coords", action="store_true")

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

    # curriculum over problem size
    parser.add_argument("--use-curriculum", action="store_true")
    parser.add_argument("--curriculum-start-n", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    print("PyTorch version:", torch.__version__)
    print("Is CUDA available?", torch.cuda.is_available())

    device = choose_device()
    print("Using device:", device)
    print(f"Edge mode: {args.edge_mode} | time_dependent={args.time_dependent}")
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(0))

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

    if device.type != "cpu":
        ok = smoke_test_policy_device(device, cfg)
        if not ok:
            print("Selected accelerator is not compatible with current HGAT/PyG ops, fallback to CPU.")
            device = torch.device("cpu")
        else:
            print("Accelerator smoke test passed.")

    policy = HGATPolicy(
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        k_nn_orders=args.k_nn_orders,
        num_encoder_layers=args.encoder_layers,
        tanh_clipping=args.tanh_clipping,
        temperature=args.temperature,
    ).to(device)
    if args.init_model_path.strip():
        init_path = args.init_model_path.strip()
        state_dict = torch.load(init_path, map_location=device, weights_only=True)
        try:
            policy.load_state_dict(state_dict, strict=True)
            print(f"Loaded initial model from {init_path}")
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load init model: {init_path}. "
                "Please ensure architecture and feature schema match."
            ) from exc
    print("policy param device:", next(policy.parameters()).device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        policy.train()
        progress = 0.0 if args.epochs <= 1 else (ep - 1) / float(args.epochs - 1)
        # 在训练过程中线性退火探索相关参数。
        cur_temp = args.temperature + (args.temperature_end - args.temperature) * progress
        cur_entropy_coef = args.entropy_coef + (args.entropy_coef_end - args.entropy_coef) * progress
        policy.decoder.temperature = float(cur_temp)

        if args.use_curriculum:
            # 课程学习：从小规模实例逐步增长到目标 N。
            n0 = max(4, min(args.curriculum_start_n, args.N))
            cur_N = int(round(n0 + (args.N - n0) * progress))
        else:
            cur_N = int(args.N)
        max_steps = 8 * (cur_N + 1)

        returns_b, logps_b, ents_b = [], [], []
        for b_id in range(args.batch_size):
            inst_seed = ep * 10000 + b_id
            if open_instances is None:
                coord, release, demand, due, meta = make_random_instance(
                    N=cur_N,
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
                    N=cur_N,
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
            env = TruckDroneRendezvousEnv(coord, release, demand, due=due, cfg=cfg, seed=inst_seed, **meta)

            returns, logps, _, entropies = pomo_rollout(
                policy,
                env,
                K=args.K,
                max_steps=max_steps,
                store_traj=False,
                start_mode="pomo",
            )
            returns_b.append(returns)
            logps_b.append(logps)
            ents_b.append(entropies)

        returns_b = torch.stack(returns_b, dim=0)  # (B,K), CPU
        logps_b = torch.stack(logps_b, dim=0)      # (B,K), policy device
        ents_b = torch.stack(ents_b, dim=0)        # (B,K), policy device

        # POMO baseline：同一实例内，对 K 条轨迹做均值基线。
        b = returns_b.mean(dim=1, keepdim=True)
        adv = (returns_b - b).to(device)

        # REINFORCE + baseline + 熵正则。
        policy_loss = -(adv.detach() * logps_b).mean()
        entropy_loss = -cur_entropy_coef * ents_b.mean()
        loss = policy_loss + entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        cost_mean = float((-returns_b).mean())
        cost_best = float((-returns_b).min())
        ent_mean = float(ents_b.mean().detach().cpu())
        print(
            f"[ep={ep:04d}] loss={loss.item():.4f} "
            f"cost_mean={cost_mean:.3f} cost_best={cost_best:.3f} "
            f"entropy={ent_mean:.3f} temp={cur_temp:.3f} ent_coef={cur_entropy_coef:.4f} "
            f"N={cur_N} B={args.batch_size} K={args.K}"
        )

        if args.checkpoint_every > 0 and (ep % args.checkpoint_every == 0 or ep == args.epochs):
            ckpt_path = f"{args.save_path}.ep{ep:04d}.pt"
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        # Release per-epoch tensors aggressively to avoid long CPU runs accumulating memory.
        del returns_b, logps_b, ents_b, b, adv, policy_loss, entropy_loss, loss
        gc.collect()

    torch.save(policy.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
