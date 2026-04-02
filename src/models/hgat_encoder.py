import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv
from typing import Dict


class _HeteroAttnBlock(nn.Module):
    """一个异构注意力残差块（不同关系用不同 TransformerConv）。"""

    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        convs = {
            ("truck", "t2o", "order"): TransformerConv(
                (-1, -1), hidden_dim, heads=heads, edge_dim=8, dropout=dropout, concat=False
            ),
            ("order", "o2t", "truck"): TransformerConv(
                (-1, -1), hidden_dim, heads=heads, edge_dim=8, dropout=dropout, concat=False
            ),
            ("drone", "d2o", "order"): TransformerConv(
                (-1, -1), hidden_dim, heads=heads, edge_dim=2, dropout=dropout, concat=False
            ),
            ("order", "o2d", "drone"): TransformerConv(
                (-1, -1), hidden_dim, heads=heads, edge_dim=2, dropout=dropout, concat=False
            ),
            ("order", "o2o", "order"): TransformerConv(
                (-1, -1), hidden_dim, heads=heads, edge_dim=8, dropout=dropout, concat=False
            ),
        }
        self.hetero_conv = HeteroConv(convs, aggr="sum")

        self.norm1 = nn.ModuleDict({
            "order": nn.LayerNorm(hidden_dim),
            "truck": nn.LayerNorm(hidden_dim),
            "drone": nn.LayerNorm(hidden_dim),
        })
        self.norm2 = nn.ModuleDict({
            "order": nn.LayerNorm(hidden_dim),
            "truck": nn.LayerNorm(hidden_dim),
            "drone": nn.LayerNorm(hidden_dim),
        })

        self.ffn = nn.ModuleDict({
            "order": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ),
            "truck": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ),
            "drone": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ),
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)

        y_dict = {}
        for ntype, x in x_dict.items():
            h = out_dict.get(ntype, torch.zeros_like(x))
            y = self.norm1[ntype](x + h)
            y = self.norm2[ntype](y + self.ffn[ntype](y))
            y_dict[ntype] = y
        return y_dict


class LiteHGATEncoder(nn.Module):
    """先投影不同类型节点特征，再堆叠异构注意力块。"""

    def __init__(self, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0, num_layers: int = 2):
        super().__init__()

        self.proj = nn.ModuleDict({
            "order": nn.Linear(11, hidden_dim),
            "truck": nn.Linear(6, hidden_dim),
            "drone": nn.Linear(6, hidden_dim),
        })
        self.blocks = nn.ModuleList([
            _HeteroAttnBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
            for _ in range(max(1, int(num_layers)))
        ])

    def forward(self, data) -> Dict[str, torch.Tensor]:
        x_dict = {
            "order": self.proj["order"](data["order"].x),
            "truck": self.proj["truck"](data["truck"].x),
            "drone": self.proj["drone"](data["drone"].x),
        }

        edge_index_dict = {k: data[k].edge_index for k in data.edge_types}
        edge_attr_dict = {k: data[k].edge_attr for k in data.edge_types}
        for block in self.blocks:
            x_dict = block(x_dict, edge_index_dict, edge_attr_dict)
        return x_dict
