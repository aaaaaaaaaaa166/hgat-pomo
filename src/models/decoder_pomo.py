import torch
import torch.nn as nn
import math
from typing import Tuple

class TwoStageDecoder(nn.Module):
    """两阶段自回归解码：先选 j，再选 k 或 no-drone。"""

    def __init__(self, hidden_dim: int = 128, tanh_clipping: float = 10.0, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tanh_clipping = float(tanh_clipping)
        self.temperature = float(temperature)

        self.q_j = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_j = nn.Linear(hidden_dim, hidden_dim)

        self.q_k = nn.Linear(hidden_dim * 3, hidden_dim)
        self.k_k = nn.Linear(hidden_dim, hidden_dim)

        self.k_none = nn.Parameter(torch.zeros(hidden_dim))

    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        return torch.where(mask > 0, logits, neg)

    @staticmethod
    def _logp_from_probs(probs: torch.Tensor, idx: int) -> torch.Tensor:
        return torch.log(probs[idx].clamp_min(1e-12))

    @staticmethod
    def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        p = probs.clamp_min(1e-12)
        return -(p * torch.log(p)).sum()

    def _shape_logits(self, logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        if self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)
        if not greedy and self.temperature > 0 and abs(self.temperature - 1.0) > 1e-8:
            logits = logits / self.temperature
        return logits

    def select_j(
        self,
        z_truck: torch.Tensor,
        z_drone: torch.Tensor,
        z_order: torch.Tensor,
        truck_mask: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        ctx = torch.cat([z_truck.squeeze(0), z_drone.squeeze(0)], dim=0)
        q = self.q_j(ctx).unsqueeze(0)
        K = self.k_j(z_order)

        logits = (q * K).sum(dim=-1).squeeze(0) / math.sqrt(self.hidden_dim)
        logits = self._shape_logits(logits, greedy=greedy)
        logits = self.masked_logits(logits, truck_mask)
        probs = torch.softmax(logits, dim=-1)
        entropy = self._entropy_from_probs(probs)

        if greedy:
            j = int(torch.argmax(probs, dim=-1).item())
            logp = self._logp_from_probs(probs, j)
            return j, logp, entropy

        dist = torch.distributions.Categorical(probs=probs)
        j = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(j, device=probs.device))
        return j, logp, entropy

    def select_k(
        self,
        env,
        obs,
        j: int,
        z_truck: torch.Tensor,
        z_drone: torch.Tensor,
        z_order: torch.Tensor,
        greedy: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        # `drone_mask` 的索引是 [0..N]，其中 0(仓库)不是合法 k。
        dm = env.get_masks(j=j)["drone_mask"]
        drone_mask = torch.as_tensor(dm, dtype=torch.float32, device=z_order.device)
        drone_mask[0] = 0.0

        ctx = torch.cat([z_truck.squeeze(0), z_drone.squeeze(0), z_order[j]], dim=0)
        q = self.q_k(ctx)
        K = self.k_k(z_order)

        logits_k = (K * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        logits_k = self._shape_logits(logits_k, greedy=greedy)
        masked_k = self.masked_logits(logits_k, drone_mask)

        logit_none = (q * self.k_none).sum() / math.sqrt(self.hidden_dim)
        logit_none = self._shape_logits(logit_none, greedy=greedy)
        all_logits = torch.cat([masked_k, logit_none.view(1)], dim=0)
        probs = torch.softmax(all_logits, dim=0)
        entropy = self._entropy_from_probs(probs)

        if greedy:
            a = int(torch.argmax(probs, dim=0).item())
            logp = self._logp_from_probs(probs, a)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=probs.device))

        if a == (all_logits.shape[0] - 1):
            return env.K_NONE, logp, entropy
        else:
            return a, logp, entropy
