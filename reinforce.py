# src/rl/reinforce.py
from __future__ import annotations
import torch


def reinforce_pomo_loss(returns: torch.Tensor, logps: torch.Tensor) -> torch.Tensor:
    """
    POMO baseline: mean over K
    loss = - E[(R - mean(R)) * logp]
    """
    baseline = returns.mean()
    adv = returns - baseline
    loss = -(adv.detach() * logps).mean()
    return loss
