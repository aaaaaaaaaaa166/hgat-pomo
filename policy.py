# src/models/policy.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

from src.graph.build_graph_pyg import build_hgat_heterodata
from src.models.hgat_encoder import LiteHGATEncoder
from src.models.decoder_pomo import TwoStageDecoder


class HGATPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0, k_nn_orders: int = 8):
        super().__init__()
        self.encoder = LiteHGATEncoder(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
        self.decoder = TwoStageDecoder(hidden_dim=hidden_dim)
        self.k_nn_orders = int(k_nn_orders)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _move_extra_to_device(self, extra: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in extra.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = torch.as_tensor(v, device=device)
        return out

    def _encode(self, env, obs: Dict[str, Any]):
        data, extra = build_hgat_heterodata(env, obs, k_nn_orders=self.k_nn_orders)
        data = data.to(self.device)
        extra = self._move_extra_to_device(extra, self.device)
        z = self.encoder(data)
        truck_mask = extra["truck_mask"].float()
        return z, truck_mask

    # ===== eval =====
    @torch.no_grad()
    def act(
        self,
        env,
        obs: Dict[str, Any],
        greedy: bool = False,
        j_fixed: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        self.eval()
        z, truck_mask = self._encode(env, obs)

        if j_fixed is None:
            j, logp_j = self.decoder.select_j(z["truck"], z["drone"], z["order"], truck_mask, greedy=greedy)
        else:
            j = int(j_fixed)
            logp_j = torch.zeros((), device=self.device)

        k, logp_k = self.decoder.select_k(env, obs, j, z["truck"], z["drone"], z["order"], greedy=greedy)
        return (k, j), (logp_j + logp_k)

    # ===== train =====
    def forward_step(
        self,
        env,
        obs: Dict[str, Any],
        greedy: bool = False,
        j_fixed: Optional[int] = None,
        skip_j_logp: bool = False,
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        skip_j_logp=True 用于 POMO forced-first-step：j 固定时不计 logp_j，只计 logp_k
        """
        self.train()
        z, truck_mask = self._encode(env, obs)

        if j_fixed is None:
            j, logp_j = self.decoder.select_j(z["truck"], z["drone"], z["order"], truck_mask, greedy=greedy)
        else:
            j = int(j_fixed)
            logp_j = torch.zeros((), device=self.device)

        k, logp_k = self.decoder.select_k(env, obs, j, z["truck"], z["drone"], z["order"], greedy=greedy)

        if skip_j_logp:
            return (k, j), logp_k
        return (k, j), (logp_j + logp_k)
