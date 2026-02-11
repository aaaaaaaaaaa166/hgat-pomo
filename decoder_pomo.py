# src/models/decoder_pomo.py
from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn as nn


class TwoStageDecoder(nn.Module):
    """
    Decode action (k, j):
      1) choose truck next stop j among orders+depot using truck_mask
      2) choose drone order k among orders using drone_mask(i,j) from env (+ allow K_NONE)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_j = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_j = nn.Linear(hidden_dim, hidden_dim)

        self.q_k = nn.Linear(hidden_dim * 3, hidden_dim)
        self.k_k = nn.Linear(hidden_dim, hidden_dim)

        self.k_none = nn.Parameter(torch.zeros(hidden_dim))

    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: 0/1 float tensor
        neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        return torch.where(mask > 0, logits, neg)

    @staticmethod
    def _logp_from_probs(probs: torch.Tensor, idx: int) -> torch.Tensor:
        return torch.log(probs[idx].clamp_min(1e-12))

    def select_j(
        self,
        z_truck: torch.Tensor,
        z_drone: torch.Tensor,
        z_order: torch.Tensor,
        truck_mask: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        z_truck: (1,d), z_drone: (1,d), z_order: (M,d) where M=N+1
        truck_mask: (M,)
        """
        ctx = torch.cat([z_truck.squeeze(0), z_drone.squeeze(0)], dim=0)  # (2d,)
        q = self.q_j(ctx).unsqueeze(0)                                    # (1,d)
        K = self.k_j(z_order)                                             # (M,d)

        logits = (q * K).sum(dim=-1).squeeze(0) / math.sqrt(self.hidden_dim)  # (M,)
        logits = self.masked_logits(logits, truck_mask)
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            j = int(torch.argmax(probs, dim=-1).item())
            logp = self._logp_from_probs(probs, j)
            return j, logp

        dist = torch.distributions.Categorical(probs=probs)
        j = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(j, device=probs.device))
        return j, logp

    def select_k(
        self,
        env,
        obs,
        j: int,
        z_truck: torch.Tensor,
        z_drone: torch.Tensor,
        z_order: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        returns k in [1..N] or env.K_NONE
        """
        dm = env.get_masks(j=j)["drone_mask"]  # numpy int8
        drone_mask = torch.as_tensor(dm, dtype=torch.float32, device=z_order.device)
        drone_mask = drone_mask.clone()
        drone_mask[0] = 0.0  # depot 不可作为无人机服务点

        ctx = torch.cat([z_truck.squeeze(0), z_drone.squeeze(0), z_order[j]], dim=0)  # (3d,)
        q = self.q_k(ctx)                                                             # (d,)
        K = self.k_k(z_order)                                                         # (M,d)

        logits_k = (K * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.hidden_dim)      # (M,)
        masked_k = self.masked_logits(logits_k, drone_mask)

        logit_none = (q * self.k_none).sum() / math.sqrt(self.hidden_dim)             # scalar
        all_logits = torch.cat([masked_k, logit_none.view(1)], dim=0)                 # (M+1,)
        probs = torch.softmax(all_logits, dim=0)

        if greedy:
            a = int(torch.argmax(probs, dim=0).item())
            logp = self._logp_from_probs(probs, a)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=probs.device))

        if a == (all_logits.shape[0] - 1):
            return env.K_NONE, logp
        else:
            return a, logp
