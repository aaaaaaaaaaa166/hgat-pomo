import torch
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _init_episode_stats(env) -> Dict[str, float]:
    node_ids = np.arange(env.N + 1)
    customer_mask = node_ids > 0
    dynamic_mask = customer_mask & (env.is_dynamic > 0)
    delivery_mask = customer_mask & (env.request_type > 0)
    pickup_mask = customer_mask & (env.request_type < 0)
    return {
        "total_orders": float(customer_mask.sum()),
        "dynamic_total": float(dynamic_mask.sum()),
        "delivery_total": float(delivery_mask.sum()),
        "pickup_total": float(pickup_mask.sum()),
        "accepted_dynamic": 0.0,
        "rejected_dynamic": 0.0,
        "served_total": 0.0,
        "served_dynamic": 0.0,
        "served_delivery": 0.0,
        "served_pickup": 0.0,
        "on_time_count": 0.0,
        "late_count": 0.0,
        "total_lateness": 0.0,
        "revenue_total": 0.0,
        "truck_energy_total": 0.0,
        "drone_energy_total": 0.0,
        "energy_total": 0.0,
        "depot_return_count": 0.0,
        "drone_dispatch_count": 0.0,
        "route_step_count": 0.0,
        "decision_step_count": 0.0,
        "timeout_count": 0.0,
    }


def _update_episode_stats(
    stats: Dict[str, float],
    env,
    obs: Dict[str, Any],
    obs2: Dict[str, Any],
    action: Tuple[int, int],
    info: Dict[str, Any],
) -> None:
    phase = str(info.get("phase", ""))
    if phase == "decision":
        stats["decision_step_count"] += 1.0
    else:
        stats["route_step_count"] += 1.0

    k, j = int(action[0]), int(action[1])
    if phase != "decision":
        prev_i = int(info.get("i", obs.get("i", 0)))
        if j == 0 and prev_i != 0:
            stats["depot_return_count"] += 1.0
        if k != env.K_NONE:
            stats["drone_dispatch_count"] += 1.0

    truck_energy = float(info.get("truck_energy_use", 0.0))
    drone_energy = float(info.get("energy_use", 0.0))
    stats["truck_energy_total"] += truck_energy
    stats["drone_energy_total"] += drone_energy
    stats["energy_total"] += truck_energy + drone_energy
    stats["revenue_total"] += float(info.get("revenue_gained", 0.0))

    prev_served = np.asarray(obs.get("served", []), dtype=np.float32)
    next_served = np.asarray(obs2.get("served", []), dtype=np.float32)
    if prev_served.size == 0 or next_served.size == 0:
        return

    new_nodes = np.where((next_served > 0.5) & (prev_served <= 0.5))[0]
    if new_nodes.size == 0:
        return

    t_prev = float(obs.get("t", 0.0))
    finish_truck = t_prev + float(info.get("truck_time", 0.0))
    finish_drone = t_prev + float(info.get("drone_time", 0.0))
    fallback_finish = t_prev + float(info.get("dt", 0.0))

    for node in new_nodes:
        node = int(node)
        if node <= 0:
            continue

        stats["served_total"] += 1.0
        if int(env.is_dynamic[node]) > 0:
            stats["served_dynamic"] += 1.0
        req_type = int(env.request_type[node])
        if req_type > 0:
            stats["served_delivery"] += 1.0
        elif req_type < 0:
            stats["served_pickup"] += 1.0

        due_t = float(env.due[node])
        if not np.isfinite(due_t):
            stats["on_time_count"] += 1.0
            continue

        if node == j:
            finish_t = finish_truck
        elif node == k:
            finish_t = finish_drone
        else:
            finish_t = fallback_finish

        late = max(0.0, finish_t - due_t)
        stats["total_lateness"] += late
        if late <= 1e-9:
            stats["on_time_count"] += 1.0
        else:
            stats["late_count"] += 1.0


def _finalize_episode_stats(stats: Dict[str, float], env, obs: Dict[str, Any], done: bool) -> None:
    node_ids = np.arange(env.N + 1)
    dynamic_mask = (node_ids > 0) & (env.is_dynamic > 0)
    accepted = np.asarray(obs.get("accepted", []), dtype=np.float32)
    rejected = np.asarray(obs.get("rejected", []), dtype=np.float32)
    if accepted.size > 0 and rejected.size > 0:
        stats["accepted_dynamic"] = float(((accepted > 0.5) & dynamic_mask).sum())
        stats["rejected_dynamic"] = float(((rejected > 0.5) & dynamic_mask).sum())
    if not done:
        stats["timeout_count"] += 1.0


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
    collect_stats: bool = False,
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
    stats_all: Optional[List[Dict[str, float]]] = [] if collect_stats else None

    # 有梯度时走训练路径 `forward_step`，否则走推理路径 `act`。
    use_forward = torch.is_grad_enabled()
    start_js = _select_start_nodes(env, K=K, start_mode=start_mode)

    for k_id in range(K):
        e = env.copy()
        obs = e.reset()
        episode_stats = _init_episode_stats(e) if collect_stats else None

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
            if collect_stats and episode_stats is not None:
                _update_episode_stats(episode_stats, e, obs, obs2, action, info)

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
        if collect_stats and episode_stats is not None and stats_all is not None:
            _finalize_episode_stats(episode_stats, e, obs, done=done)
            stats_all.append(episode_stats)

    if collect_stats and stats_all is not None:
        return returns, logps, trajs, entropies, stats_all
    return returns, logps, trajs, entropies
