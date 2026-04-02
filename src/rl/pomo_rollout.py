import torch
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _select_start_nodes(env, K: int, start_mode: str) -> List[int]:
    """为每条 POMO 轨迹选择首步卡车节点。

    - `pomo` 模式：在可行点中分散起点，降低方差；
    - 其他模式：首步不固定，交给策略自行决定。
    """
    if start_mode != "pomo":
        return [None] * K

    mask = env.get_masks()["truck_mask"]
    feasible = np.where(mask > 0)[0]
    if feasible.size == 0:
        return [0] * K

    # Prefer non-depot starts when feasible orders exist at current time.
    feasible_orders = feasible[feasible != 0]
    pool = feasible_orders if feasible_orders.size > 0 else feasible

    replace = pool.size < K
    starts = np.random.choice(pool, size=K, replace=replace)
    return [int(x) for x in starts]

def pomo_rollout(
    policy,
    env,
    K: int = 8,
    max_steps: int = 256,
    store_traj: bool = False,
    timeout_penalty: float = 1000.0,
    start_mode: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[List[Dict[str, Any]]]], torch.Tensor]:
    """
    POMO rollout for 1 instance:
      - Run K trajectories on K env copies
      - returns[k] = sum of rewards (reward = -dt)  => negative total time
      - logps[k]   = sum of log-probs along the trajectory
      - trajs[k]   = list of per-step dict logs if store_traj=True
    """

    returns = torch.zeros((K,), dtype=torch.float32, device="cpu")
    device = next(policy.parameters()).device
    logps = torch.zeros((K,), dtype=torch.float32, device=device)
    entropies = torch.zeros((K,), dtype=torch.float32, device=device)

    trajs: Optional[List[List[Dict[str, Any]]]] = [[] for _ in range(K)] if store_traj else None

    # 有梯度时走训练路径 `forward_step`，否则走推理路径 `act`。
    use_forward = torch.is_grad_enabled()
    start_js = _select_start_nodes(env, K=K, start_mode=start_mode)

    for k_id in range(K):
        e = env.copy()
        obs = e.reset()

        sum_r = 0.0
        sum_logp = torch.zeros((), dtype=torch.float32, device=device)
        sum_ent = torch.zeros((), dtype=torch.float32, device=device)

        done = False
        for step in range(max_steps):
            if step == 0 and start_js[k_id] is not None:
                j0 = int(start_js[k_id])
                if use_forward:
                    # POMO 固定首步起点；该人工固定动作不计入 logp 损失项。
                    action, logp, ent = policy.forward_step(
                        e, obs, j_fixed=j0, skip_j_logp=True, return_entropy=True
                    )
                else:
                    action, logp, ent = policy.act(e, obs, j_fixed=j0, return_entropy=True)
                    # 评估路径保持接口一致，但去掉这一步的人为 logp 贡献。
                    logp = logp * 0.0
            else:
                if use_forward:
                    action, logp, ent = policy.forward_step(e, obs, return_entropy=True)
                else:
                    action, logp, ent = policy.act(e, obs, return_entropy=True)

            obs2, r, done, info = e.step(action)

            r_float = float(r)
            sum_r += r_float
            sum_logp += logp.squeeze()
            sum_ent += ent.squeeze()

            if store_traj:
                trajs[k_id].append({
                    "obs": obs,
                    "action": action,
                    "reward": r_float,
                    "info": info,
                })

            obs = obs2
            if done:
                break

        if not done:
            # 到达上限仍未完成时施加超时惩罚，避免对半截轨迹过于乐观。
            sum_r -= timeout_penalty
            if store_traj:
                trajs[k_id].append({
                    "obs": obs,
                    "action": ("TIMEOUT",),
                    "reward": -timeout_penalty,
                    "info": {"timeout": True, "max_steps": max_steps},
                })

        returns[k_id] = sum_r
        logps[k_id] = sum_logp
        entropies[k_id] = sum_ent

    return returns, logps, trajs, entropies
