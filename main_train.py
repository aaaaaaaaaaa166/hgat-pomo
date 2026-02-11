# src/main_train.py
from __future__ import annotations
import os
import random
import numpy as np
import torch

from src.env.instance_gen import make_random_instance
from src.env.td_env import TruckDroneRendezvousEnv, EnvConfig
from src.models.policy import HGATPolicy
from src.rl.pomo_rollout import pomo_rollout


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(0)

    # CPU 稳定性：线程别开太满（Windows 尤其）
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    print("PyTorch version:", torch.__version__)  # 输出 PyTorch 版本
    print("Is CUDA available?", torch.cuda.is_available())  # 检查 CUDA 是否可用

    device = torch.device("cpu")
    print("Using device:", device)

    # ----- hyperparams -----
    N = 30
    K = 8
    epochs = 200
    lr = 1e-4

    # rollout step upper bound: 动态 release 也不需要 2000，一般 5*(N+1) 很够
    max_steps = 5 * (N + 1)

    cfg = EnvConfig(
        vT=1.0, vD=1.5, QD=1.0, B=6.0,
        sT=0.0, sD=0.0,
        allow_wait=True,
        idle_to_next_release=True,
    )

    policy = HGATPolicy(hidden_dim=128, heads=4, dropout=0.0, k_nn_orders=8).to(device)
    print("policy param device:", next(policy.parameters()).device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        policy.train()

        # --- sample a fresh instance each epoch (better generalization) ---
        coord, release, demand = make_random_instance(
            N=N,
            seed=ep,  # important!
            coord_scale=10.0,
            release_mode="batches",
            n_batches=4,
            max_release=10.0,
        )
        env = TruckDroneRendezvousEnv(coord, release, demand, cfg=cfg, seed=ep)

        returns, logps, _ = pomo_rollout(
            policy, env, K=K, max_steps=max_steps, store_traj=False
        )
        # returns: (K,) on CPU, reward = -total_time

        # POMO baseline: mean over K
        b = returns.mean()
        adv = (returns - b).to(device)  # move to policy device

        # REINFORCE with baseline
        loss = -(adv.detach() * logps).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        time_mean = float((-returns).mean())
        time_best = float((-returns).min())
        print(f"[ep={ep:04d}] loss={loss.item():.4f}  time_mean={time_mean:.3f}  time_best={time_best:.3f}")

    torch.save(policy.state_dict(), "policy.pt")
    print("Saved to policy.pt")


if __name__ == "__main__":
    main()
