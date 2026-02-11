from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
from torch_geometric.data import HeteroData


def build_hgat_heterodata(env, obs: Dict[str, Any], k_nn_orders: int = 8) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
    """
    Node types:
      - truck: 1 node
      - drone: 1 node
      - order: N+1 nodes (0 depot + N orders)

    Edges:
      truck <-> order: edge_attr [dist(i,o), timeT(i,o)]
      drone <-> order: edge_attr [dist(i,o), timeD(i,o)]
      order -> order (o2o kNN): edge_attr [dist(a,b), timeT(a,b), timeD(a,b)]
    """
    t = float(obs["t"])
    i = int(obs["i"])
    served_np = obs["served"]  # numpy (N+1,)

    N = env.N
    M = N + 1

    # ---- tensors (CPU) ----
    coord = torch.from_numpy(env.coord)          # (M,2) float32
    release = torch.from_numpy(env.release)      # (M,) float32
    demand = torch.from_numpy(env.demand)        # (M,) float32
    served = torch.from_numpy(served_np).float() # (M,)

    # normalize coords
    xy_min = coord.min(dim=0).values
    xy_max = coord.max(dim=0).values
    xy_rng = torch.clamp(xy_max - xy_min, min=1e-6)
    coord_n = (coord - xy_min) / xy_rng

    released = (t >= release).float()
    wait_time = torch.clamp(torch.tensor(t, dtype=torch.float32) - release, min=0.0)

    is_depot = torch.zeros(M, dtype=torch.float32); is_depot[0] = 1.0
    is_current = torch.zeros(M, dtype=torch.float32); is_current[i] = 1.0

    x_order = torch.stack([
        coord_n[:, 0],
        coord_n[:, 1],
        demand,
        released,
        served,
        wait_time,
        is_depot,
        is_current
    ], dim=1)  # (M, 8)

    # truck node features
    t_den = float(max(1e-6, float(env.release.max())))
    t_norm = torch.tensor([t / t_den], dtype=torch.float32)
    unserved_ratio = torch.tensor([(N - int(served_np[1:].sum())) / max(1, N)], dtype=torch.float32)
    x_truck = torch.cat([coord_n[i], t_norm, unserved_ratio]).view(1, -1)  # (1,4)

    # drone node features
    vT = float(env.cfg.vT); vD = float(env.cfg.vD)
    vD_norm = torch.tensor([vD / max(1e-6, (vT + vD))], dtype=torch.float32)
    x_drone = torch.tensor([[float(env.cfg.QD), float(env.cfg.B), float(vD_norm), 1.0]], dtype=torch.float32)

    data = HeteroData()
    data["order"].x = x_order
    data["truck"].x = x_truck
    data["drone"].x = x_drone

    # ---- dynamic star edges: distances from current i (use cached dist_mat) ----
    dist_i = torch.from_numpy(env.dist_mat[i].copy()).float()  # (M,)
    timeT_i = dist_i / float(env.cfg.vT)
    timeD_i = dist_i / float(env.cfg.vD)

    # edge_index from env cache (numpy -> torch once per step, small)
    data["truck", "t2o", "order"].edge_index = torch.from_numpy(env.edge_index_t2o).long()
    data["truck", "t2o", "order"].edge_attr = torch.stack([dist_i, timeT_i], dim=1)  # (M,2)

    data["order", "o2t", "truck"].edge_index = torch.from_numpy(env.edge_index_o2t).long()
    data["order", "o2t", "truck"].edge_attr = torch.stack([dist_i, timeT_i], dim=1)

    data["drone", "d2o", "order"].edge_index = torch.from_numpy(env.edge_index_d2o).long()
    data["drone", "d2o", "order"].edge_attr = torch.stack([dist_i, timeD_i], dim=1)  # (M,2)

    data["order", "o2d", "drone"].edge_index = torch.from_numpy(env.edge_index_o2d).long()
    data["order", "o2d", "drone"].edge_attr = torch.stack([dist_i, timeD_i], dim=1)

    # ---- static o2o knn edges (cached in env) ----
    o2o_ei_np, o2o_ea_np = env.get_o2o_edges(k_nn=k_nn_orders)
    data["order", "o2o", "order"].edge_index = torch.from_numpy(o2o_ei_np).long()
    data["order", "o2o", "order"].edge_attr = torch.from_numpy(o2o_ea_np).float()  # (E,3)

    # ---- masks ----
    truck_mask = torch.from_numpy(env.get_masks()["truck_mask"]).float()  # (M,)

    extra = {
        "truck_mask": truck_mask,
        "served": served,
        "released": released,
        "cur_i": torch.tensor([i], dtype=torch.long),
        "t": torch.tensor([t], dtype=torch.float32),
    }
    return data, extra
