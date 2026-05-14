import torch
import torch.nn as nn
from src.graph.build_graph_pyg import build_hgat_heterodata
from src.models.hgat_encoder import LiteHGATEncoder
from src.models.decoder_pomo import TwoStageDecoder
from typing import Any, Dict, Optional

class HGATPolicy(nn.Module):
    """HGAT 编码器 + 两阶段解码器策略网络。

    动作分解为两步：
    1) 先选卡车目的地 j；
    2) 在 j 条件下再选无人机订单 k（或 no-drone）。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        k_nn_orders: int = 8,
        num_encoder_layers: int = 2,
        tanh_clipping: float = 10.0,
        temperature: float = 1.0,
        order_feature_dim: int = 12,
        truck_feature_dim: int = 6,
        drone_feature_dim: int = 6,
    ):
        super().__init__()
        self.encoder = LiteHGATEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_encoder_layers,
            order_feature_dim=order_feature_dim,
            truck_feature_dim=truck_feature_dim,
            drone_feature_dim=drone_feature_dim,
        )
        self.decoder = TwoStageDecoder(
            hidden_dim=hidden_dim,
            tanh_clipping=tanh_clipping,
            temperature=temperature,
        )
        self.k_nn_orders = int(k_nn_orders)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _move_extra_to_device(self, extra: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in extra.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = torch.as_tensor(v, device=device)
        return out

    def _encode(self, env, obs: Dict[str, Any]):
        """把 env/obs 转为异构图并送入编码器。"""
        data, extra = build_hgat_heterodata(env, obs, k_nn_orders=self.k_nn_orders)
        data = data.to(self.device)
        extra = self._move_extra_to_device(extra, self.device)
        z = self.encoder(data)
        truck_mask = extra["truck_mask"].float()
        return z, truck_mask

    @torch.no_grad()
    def act(
        self,
        env,
        obs: Dict[str, Any],
        greedy: bool = False,
        j_fixed: Optional[int] = None,
        return_entropy: bool = False,
    ):
        """推理路径（无梯度），可选贪心/采样。"""
        self.eval()
        z, truck_mask = self._encode(env, obs)

        if j_fixed is None:
            j, logp_j, ent_j = self.decoder.select_j(
                z["truck"], z["drone"], z["order"], truck_mask, greedy=greedy
            )
        else:
            j = int(j_fixed)
            logp_j = torch.zeros((), device=self.device)
            ent_j = torch.zeros((), device=self.device)

        k, logp_k, ent_k = self.decoder.select_k(
            env, obs, j, z["truck"], z["drone"], z["order"], greedy=greedy
        )
        if return_entropy:
            return (k, j), (logp_j + logp_k), (ent_j + ent_k)
        return (k, j), (logp_j + logp_k)

    def forward_step(
        self,
        env,
        obs: Dict[str, Any],
        greedy: bool = False,
        j_fixed: Optional[int] = None,
        skip_j_logp: bool = False,
        return_entropy: bool = False,
    ):
        """训练路径（保留梯度）。

        与 `act` 的关键区别是：这里保留 log-prob 的梯度，
        供 rollout 累积策略梯度目标。
        """
        self.train()
        z, truck_mask = self._encode(env, obs)

        if j_fixed is None:
            j, logp_j, ent_j = self.decoder.select_j(
                z["truck"], z["drone"], z["order"], truck_mask, greedy=greedy
            )
        else:
            j = int(j_fixed)
            logp_j = torch.zeros((), device=self.device)
            ent_j = torch.zeros((), device=self.device)

        k, logp_k, ent_k = self.decoder.select_k(
            env, obs, j, z["truck"], z["drone"], z["order"], greedy=greedy
        )

        if skip_j_logp:
            if return_entropy:
                return (k, j), logp_k, ent_k
            return (k, j), logp_k
        if return_entropy:
            return (k, j), (logp_j + logp_k), (ent_j + ent_k)
        return (k, j), (logp_j + logp_k)
