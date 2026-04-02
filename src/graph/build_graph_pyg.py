from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch_geometric.data import HeteroData


def build_hgat_heterodata(env, obs: Dict[str, Any], k_nn_orders: int = 8) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
    """
    Node types:
      - truck: 1 node
      - drone: 1 node
      - order: N+1 nodes (0 depot + N orders)

    Edges:
      truck <-> order: edge_attr [road/static pairwise edge features]
      drone <-> order: edge_attr [dist(i,o), timeD(i,o)]
      order -> order (o2o kNN): edge_attr [road/static pairwise edge features]
    """
    t = float(obs["t"])
    i = int(obs["i"])
    served_np = obs["served"]  # numpy (N+1,)
    soc = float(obs.get("soc", float(env.cfg.soc_init)))
    is_peak = float(obs.get("is_peak", 0.0))

    N = env.N
    M = N + 1

    # ---- 张量构建（先在 CPU）----
    # 统一在这里建图，随后在 policy._encode 中搬到目标设备。
    coord = torch.from_numpy(env.coord)          # (M,2) float32
    release = torch.from_numpy(env.release)      # (M,) float32
    demand = torch.from_numpy(env.demand)        # (M,) float32
    due = torch.from_numpy(env.due)              # (M,) float32, inf means no deadline
    served = torch.from_numpy(served_np).float() # (M,)

    # 按实例做坐标归一化，降低不同地图尺度带来的分布漂移。
    xy_min = coord.min(dim=0).values
    xy_max = coord.max(dim=0).values
    xy_rng = torch.clamp(xy_max - xy_min, min=1e-6)
    coord_n = (coord - xy_min) / xy_rng

    released = (t >= release).float()
    wait_time = torch.clamp(torch.tensor(t, dtype=torch.float32) - release, min=0.0)

    finite_due = torch.isfinite(due)
    finite_vals = due[finite_due]
    release_max = float(env.release.max()) if env.release.size > 0 else 0.0
    due_ref = float(finite_vals.max().item()) if finite_vals.numel() > 0 else release_max
    t_den = max(1e-6, due_ref if due_ref > 0 else (release_max + 1.0))

    # 对无截止时间节点，填充有限伪值，保证特征有限且可归一化。
    due_safe = torch.where(finite_due, due, torch.full_like(due, 2.0 * t_den))
    has_deadline = finite_due.float()
    due_norm = torch.clamp(due_safe / t_den, min=0.0, max=2.0)
    slack = due_safe - torch.tensor(t, dtype=torch.float32)
    slack_norm = torch.clamp(slack / t_den, min=-1.0, max=2.0)
    slack_norm = torch.where(has_deadline > 0.5, slack_norm, torch.full_like(slack_norm, 2.0))

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
        is_current,
        has_deadline,
        due_norm,
        slack_norm,
    ], dim=1)  # (M, 11)

    # truck 节点：当前位置 + 全局进度特征
    t_norm = torch.tensor([t / t_den], dtype=torch.float32)
    unserved_ratio = torch.tensor([(N - int(served_np[1:].sum())) / max(1, N)], dtype=torch.float32)
    unserved_ids = np.where(served_np[1:] == 0)[0] + 1
    if unserved_ids.size > 0:
        future_rel = env.release[unserved_ids]
        future_rel = future_rel[future_rel > t]
        next_gap = float(future_rel.min() - t) if future_rel.size > 0 else 0.0
    else:
        next_gap = 0.0
    next_gap_norm = torch.tensor([max(0.0, next_gap) / t_den], dtype=torch.float32)
    x_truck = torch.cat(
        [coord_n[i], t_norm, unserved_ratio, next_gap_norm, torch.tensor([is_peak], dtype=torch.float32)]
    ).view(1, -1)  # (1,6)

    # drone 节点：能力参数 + 当前电量状态
    vT = float(env.cfg.vT); vD = float(env.cfg.vD)
    vD_norm = vD / max(1e-6, (vT + vD))
    x_drone = torch.tensor(
        [[
            float(env.cfg.QD),
            float(env.cfg.B),
            float(vD_norm),
            float(soc),
            float(env.cfg.soc_min_reserve),
            float(env.cfg.energy_per_dist),
        ]],
        dtype=torch.float32,
    )  # (1,6)

    data = HeteroData()
    data["order"].x = x_order
    data["truck"].x = x_truck
    data["drone"].x = x_drone

    # ---- 动态星型边：从当前 i 指向所有节点 ----
    # 这部分依赖当前位置，每一步都要更新。
    dist_i = torch.from_numpy(env.dist_mat[i].copy()).float()  # (M,)
    timeD_i = dist_i / float(env.cfg.vD)
    truck_edge_attr_i = torch.from_numpy(env.get_dense_edge_attr()[i].copy()).float()  # (M,8)

    # edge_index from env cache (numpy -> torch once per step, small)
    data["truck", "t2o", "order"].edge_index = torch.from_numpy(env.edge_index_t2o).long()
    data["truck", "t2o", "order"].edge_attr = truck_edge_attr_i  # (M,8)

    data["order", "o2t", "truck"].edge_index = torch.from_numpy(env.edge_index_o2t).long()
    data["order", "o2t", "truck"].edge_attr = truck_edge_attr_i

    data["drone", "d2o", "order"].edge_index = torch.from_numpy(env.edge_index_d2o).long()
    data["drone", "d2o", "order"].edge_attr = torch.stack([dist_i, timeD_i], dim=1)  # (M,2)

    data["order", "o2d", "drone"].edge_index = torch.from_numpy(env.edge_index_o2d).long()
    data["order", "o2d", "drone"].edge_attr = torch.stack([dist_i, timeD_i], dim=1)

    # ---- 静态 o2o kNN 边（环境内缓存）----
    # 仅由几何关系决定，可跨 step 复用。
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
        "soc": torch.tensor([soc], dtype=torch.float32),
    }
    return data, extra
