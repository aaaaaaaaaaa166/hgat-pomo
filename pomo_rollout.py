# src/rl/pomo_rollout.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch


def pomo_rollout(
    policy,
    env,
    K: int = 8,
    max_steps: int = 256,
    store_traj: bool = False,
    timeout_penalty: float = 1000.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[List[Dict[str, Any]]]]]:
    """
    POMO rollout for 1 instance:
      - Run K trajectories on K env copies
      - returns[k] = sum of rewards (reward = -dt)  => negative total time
      - logps[k]   = sum of log-probs along the trajectory
      - trajs[k]   = list of per-step dict logs if store_traj=True
    """

    # returns on CPU for easy stats
    returns = torch.zeros((K,), dtype=torch.float32, device="cpu")

    # logps on same device as policy params
    device = next(policy.parameters()).device
    logps = torch.zeros((K,), dtype=torch.float32, device=device)

    trajs: Optional[List[List[Dict[str, Any]]]] = [ [] for _ in range(K) ] if store_traj else None

    # Decide which policy method to use:
    # - if grad enabled => training => forward_step (keeps graph)
    # - else => evaluation => act (no_grad)
    use_forward = torch.is_grad_enabled()

    for k_id in range(K):
        e = env.copy()
        obs = e.reset()

        sum_r: float = 0.0
        sum_logp = torch.zeros((), dtype=torch.float32, device=device)

        done = False
        for step in range(max_steps):
            if use_forward:
                action, logp = policy.forward_step(e, obs)
            else:
                action, logp = policy.act(e, obs)

            obs2, r, done, info = e.step(action)

            r_float = float(r)
            sum_r += r_float
            sum_logp = sum_logp + logp.squeeze()

            if store_traj:
                assert trajs is not None
                trajs[k_id].append({
                    "obs": obs,
                    "action": action,     # (k,j)
                    "reward": r_float,    # -dt
                    "info": info,         # must include dt, i, j, k, ...
                })

            obs = obs2
            if done:
                break

        # If not finished, add penalty so training doesn't silently learn "stall forever"
        if not done:
            sum_r -= float(timeout_penalty)
            if store_traj:
                assert trajs is not None
                trajs[k_id].append({
                    "obs": obs,
                    "action": ("TIMEOUT",),
                    "reward": -float(timeout_penalty),
                    "info": {"timeout": True, "max_steps": max_steps},
                })

        returns[k_id] = sum_r
        logps[k_id] = sum_logp

    return returns, logps, trajs
