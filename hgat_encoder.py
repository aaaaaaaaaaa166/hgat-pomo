from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv


class LiteHGATEncoder(nn.Module):
    """
    Minimal heterogeneous encoder for HeteroData:
      - per-relation TransformerConv (uses edge_attr)
      - concat=False so output dim = hidden_dim (avoid hidden_dim*heads)
    Outputs dict of node embeddings for each type.
    """

    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj = nn.ModuleDict({
            "order": nn.Linear(8, hidden_dim),
            "truck": nn.Linear(4, hidden_dim),
            "drone": nn.Linear(4, hidden_dim),
        })

        convs = {
            ("truck", "t2o", "order"): TransformerConv((-1, -1), hidden_dim, heads=heads, edge_dim=2, dropout=dropout, concat=False),
            ("order", "o2t", "truck"): TransformerConv((-1, -1), hidden_dim, heads=heads, edge_dim=2, dropout=dropout, concat=False),
            ("drone", "d2o", "order"): TransformerConv((-1, -1), hidden_dim, heads=heads, edge_dim=2, dropout=dropout, concat=False),
            ("order", "o2d", "drone"): TransformerConv((-1, -1), hidden_dim, heads=heads, edge_dim=2, dropout=dropout, concat=False),
            ("order", "o2o", "order"): TransformerConv((-1, -1), hidden_dim, heads=heads, edge_dim=3, dropout=dropout, concat=False),
        }
        self.hetero_conv = HeteroConv(convs, aggr="sum")

        self.post = nn.ModuleDict({
            "order": nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            "truck": nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            "drone": nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
        })

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x_dict = {
            "order": self.proj["order"](data["order"].x),
            "truck": self.proj["truck"](data["truck"].x),
            "drone": self.proj["drone"](data["drone"].x),
        }

        out_dict = self.hetero_conv(
            x_dict,
            {k: data[k].edge_index for k in data.edge_types},
            {k: data[k].edge_attr for k in data.edge_types},
        )

        z = {}
        for ntype in x_dict.keys():
            h = out_dict.get(ntype, torch.zeros_like(x_dict[ntype]))
            z[ntype] = self.post[ntype](x_dict[ntype] + h)
        return z
