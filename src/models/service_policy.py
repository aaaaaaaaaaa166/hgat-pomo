from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.graph.build_graph_pyg import build_hgat_heterodata
from src.models.hgat_encoder import LiteHGATEncoder


class ServicePolicy(nn.Module):
    """Service V2 policy for explicit accept/reject and routing priorities.

    The original HGATPolicy is intentionally left untouched so frozen baseline
    weights remain loadable. This policy is designed for imitation warm start
    before any RL fine-tuning.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        k_nn_orders: int = 8,
        num_encoder_layers: int = 2,
        order_feature_dim: int = 22,
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
        self.acceptance_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.routing_priority_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.drone_assignment_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.no_drone_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.lateness_risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.insertion_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.k_nn_orders = int(k_nn_orders)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _move_extra_to_device(self, extra: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in extra.items():
            out[key] = value.to(device) if torch.is_tensor(value) else torch.as_tensor(value, device=device)
        return out

    def _order_context(self, z: Dict[str, torch.Tensor]) -> torch.Tensor:
        order = z["order"]
        truck = z["truck"][0].expand(order.size(0), -1)
        drone = z["drone"][0].expand(order.size(0), -1)
        return torch.cat([order, truck, drone], dim=-1)

    def forward_data(self, data: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        data = data.to(self.device)
        z = self.encoder(data)
        ctx = self._order_context(z)
        truck_drone = torch.cat([z["truck"][0], z["drone"][0]], dim=-1)
        no_drone_logit = self.no_drone_head(truck_drone).view(1)
        drone_logits = self.drone_assignment_head(ctx).squeeze(-1)
        return {
            "accept_logits": self.acceptance_head(ctx),
            "route_priority_logits": self.routing_priority_head(ctx).squeeze(-1),
            "drone_assignment_logits": drone_logits,
            "no_drone_logit": no_drone_logit,
            "lateness_risk": self.lateness_risk_head(ctx).squeeze(-1),
            "insertion_score": self.insertion_score_head(ctx).squeeze(-1),
            "z": z,
            "extra": extra or {},
        }

    def forward_env(self, env: Any, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        data, extra = build_hgat_heterodata(env, obs, k_nn_orders=self.k_nn_orders)
        extra = self._move_extra_to_device(extra, self.device)
        return self.forward_data(data, extra)

    @torch.no_grad()
    def act(self, env: Any, obs: Dict[str, Any], greedy: bool = True) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        self.eval()
        out = self.forward_env(env, obs)
        extra = out.get("extra", {})
        truck_mask = extra.get("truck_mask")
        current = int(obs.get("current_decision_request", -1))
        if current > 0:
            logits = out["accept_logits"][current]
            decision = int(torch.argmax(logits).item()) if greedy else int(torch.distributions.Categorical(logits=logits).sample().item())
            if decision == 1:
                return (env.K_NONE, current), {"phase": "acceptance", "decision": "accept", "accept_logits": logits.detach().cpu().tolist()}
            return (env.K_NONE, 0), {"phase": "acceptance", "decision": "reject", "accept_logits": logits.detach().cpu().tolist()}

        scores = out["route_priority_logits"].clone()
        if truck_mask is not None:
            mask = truck_mask.to(scores.device).bool()
            scores = scores.masked_fill(~mask, -1e9)
        j = int(torch.argmax(scores).item()) if greedy else int(torch.distributions.Categorical(logits=scores).sample().item())
        if j <= 0:
            return (env.K_NONE, j), {"phase": "route", "j": j, "k": env.K_NONE}
        drone_scores = torch.cat([out["no_drone_logit"], out["drone_assignment_logits"]], dim=0)
        k_choice = int(torch.argmax(drone_scores).item()) if greedy else int(torch.distributions.Categorical(logits=drone_scores).sample().item())
        k = env.K_NONE if k_choice == 0 else k_choice - 1
        return (int(k), int(j)), {"phase": "route", "j": int(j), "k": int(k)}

