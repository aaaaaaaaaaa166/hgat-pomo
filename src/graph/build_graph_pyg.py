from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.env.instance_gen import REQUEST_PICKUP
from src.training.sequence_time_window_features import (
    compute_drone_sequence_tw_features,
    compute_order_sequence_tw_features,
    compute_truck_sequence_tw_features,
)


def build_hgat_heterodata(env, obs: Dict[str, Any], k_nn_orders: int = 8) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
    t = float(obs["t"])
    i = int(obs["i"])
    served_np = np.asarray(obs["served"], dtype=np.float32)
    accepted_np = np.asarray(obs.get("accepted", np.zeros_like(served_np)), dtype=np.float32)
    rejected_np = np.asarray(obs.get("rejected", np.zeros_like(served_np)), dtype=np.float32)
    expired_np = np.asarray(obs.get("expired", np.zeros_like(served_np)), dtype=np.float32)
    known_np = np.asarray(obs.get("known", np.zeros_like(served_np)), dtype=np.float32)
    loaded_np = np.asarray(obs.get("loaded", np.zeros_like(served_np)), dtype=np.float32)
    active_deadlines_np = np.asarray(obs.get("active_deadlines", env.due), dtype=np.float32)
    soc = float(obs.get("soc", float(env.cfg.soc_init)))
    is_peak = float(obs.get("is_peak", 0.0))
    truck_load_ratio = float(obs.get("truck_load_ratio", 0.0))
    pending_count = float(obs.get("pending_count", 0))

    n = env.N
    m = n + 1

    coord = torch.from_numpy(env.coord)
    served = torch.from_numpy(served_np).float()
    accepted = torch.from_numpy(accepted_np).float()
    rejected = torch.from_numpy(rejected_np).float()
    expired = torch.from_numpy(expired_np).float()
    known = torch.from_numpy(known_np).float()
    loaded = torch.from_numpy(loaded_np).float()
    active_deadlines = torch.from_numpy(active_deadlines_np)
    known_mask = known > 0.5
    known_mask[0] = True

    visible_coord = coord[known_mask]
    if visible_coord.numel() == 0:
        visible_coord = coord[:1]
    xy_min = visible_coord.min(dim=0).values
    xy_max = visible_coord.max(dim=0).values
    xy_rng = torch.clamp(xy_max - xy_min, min=1e-6)
    coord_n = (coord - xy_min) / xy_rng
    coord_n = torch.where(known_mask.unsqueeze(-1), coord_n, torch.zeros_like(coord_n))

    request_sign = np.where(env.request_type == REQUEST_PICKUP, -1.0, 1.0).astype(np.float32)
    request_sign[0] = 0.0
    signed_demand = torch.from_numpy(env.demand * request_sign)
    signed_demand = torch.where(known_mask, signed_demand, torch.zeros_like(signed_demand))
    revenue = torch.from_numpy(env.revenue.copy()).float()
    visible_revenue = revenue[known_mask]
    revenue_ref = float(visible_revenue.max().item()) if visible_revenue.numel() > 0 else 0.0
    revenue_den = max(1e-6, revenue_ref)
    revenue_norm = torch.clamp(revenue / revenue_den, min=0.0, max=5.0)
    revenue_norm = torch.where(known_mask, revenue_norm, torch.zeros_like(revenue_norm))
    accepted = torch.where(known_mask, accepted, torch.zeros_like(accepted))
    rejected = torch.where(known_mask, rejected, torch.zeros_like(rejected))
    expired = torch.where(known_mask, expired, torch.zeros_like(expired))
    served = torch.where(known_mask, served, torch.zeros_like(served))
    loaded = torch.where(known_mask, loaded, torch.zeros_like(loaded))

    finite_deadline = torch.isfinite(active_deadlines)
    finite_vals = active_deadlines[finite_deadline]
    release_max = float(env.release.max()) if env.release.size > 0 else 0.0
    deadline_ref = float(finite_vals.max().item()) if finite_vals.numel() > 0 else release_max
    t_den = max(1e-6, deadline_ref if deadline_ref > 0 else (release_max + 1.0))

    deadline_safe = torch.where(finite_deadline, active_deadlines, torch.full_like(active_deadlines, 2.0 * t_den))
    deadline_norm = torch.clamp(deadline_safe / t_den, min=0.0, max=2.0)
    slack = deadline_safe - torch.tensor(t, dtype=torch.float32)
    slack_norm = torch.clamp(slack / t_den, min=-1.0, max=2.0)
    slack_norm = torch.where(finite_deadline, slack_norm, torch.full_like(slack_norm, 2.0))
    hidden_deadline = torch.full_like(deadline_norm, 2.0)
    deadline_norm = torch.where(known_mask, deadline_norm, hidden_deadline)
    slack_norm = torch.where(known_mask, slack_norm, hidden_deadline)

    is_depot = torch.zeros(m, dtype=torch.float32)
    is_depot[0] = 1.0
    is_current = torch.zeros(m, dtype=torch.float32)
    is_current[i] = 1.0

    x_order = torch.stack(
        [
            coord_n[:, 0],
            coord_n[:, 1],
            signed_demand,
            revenue_norm,
            known,
            accepted,
            served,
            loaded,
            is_depot,
            is_current,
            deadline_norm,
            slack_norm,
        ],
        dim=1,
    )
    if bool(getattr(env.cfg, "enable_sequence_time_window_features", False)):
        x_order = torch.cat([x_order, compute_order_sequence_tw_features(env, obs)], dim=1)
    if str(getattr(env.cfg, "feature_mode", "legacy")) == "service_v2":
        response_deadline = torch.from_numpy(env.decision_deadline.copy()).float()
        response_safe = torch.where(
            torch.isfinite(response_deadline),
            response_deadline,
            torch.full_like(response_deadline, 2.0 * t_den),
        )
        response_deadline_norm = torch.clamp(response_safe / t_den, min=0.0, max=2.0)
        response_slack_norm = torch.clamp((response_safe - torch.tensor(t, dtype=torch.float32)) / t_den, min=-1.0, max=2.0)
        estimated_arrival = []
        arrival_slack = []
        predicted_late = []
        urgency = []
        tight_threshold = max(1.0, 0.25 * float(getattr(env.cfg, "B", 6.0)) + float(env.cfg.sT))
        for node in range(m):
            if node == 0:
                eta = t
            else:
                try:
                    eta = t + float(env._tau_truck(i, node, apply_traffic=False, bucket=str(obs.get("time_bucket", env.get_time_bucket())))) + float(env.cfg.sT)
                except Exception:
                    eta = t + float(env.dist_mat[i, node]) / max(1e-9, float(env.cfg.vT)) + float(env.cfg.sT)
            due = float(env.due[node])
            slack_after = (due - eta) if np.isfinite(due) else 2.0 * t_den
            late = max(0.0, eta - due) if np.isfinite(due) else 0.0
            estimated_arrival.append(eta)
            arrival_slack.append(slack_after)
            predicted_late.append(late)
            urgency.append(0.0 if node == 0 else min(3.0, 1.0 / (1.0 + max(0.0, slack_after)) + late / t_den))
        estimated_arrival_t = torch.tensor(estimated_arrival, dtype=torch.float32)
        arrival_slack_t = torch.tensor(arrival_slack, dtype=torch.float32)
        predicted_late_t = torch.tensor(predicted_late, dtype=torch.float32)
        urgency_t = torch.tensor(urgency, dtype=torch.float32)
        service_v2_extra = torch.stack(
            [
                torch.where(known_mask, response_deadline_norm, torch.zeros_like(response_deadline_norm)),
                torch.where(known_mask, response_slack_norm, torch.zeros_like(response_slack_norm)),
                torch.where(known_mask, torch.clamp(estimated_arrival_t / t_den, 0.0, 3.0), torch.zeros_like(estimated_arrival_t)),
                torch.where(known_mask, torch.clamp(arrival_slack_t / t_den, -2.0, 2.0), torch.zeros_like(arrival_slack_t)),
                torch.where(known_mask, torch.clamp(predicted_late_t / t_den, 0.0, 3.0), torch.zeros_like(predicted_late_t)),
                ((response_safe - torch.tensor(t, dtype=torch.float32)) < 0.0).float() * known.float(),
                rejected,
                expired,
                ((arrival_slack_t <= tight_threshold).float() * known.float()),
                torch.where(known_mask, torch.clamp(urgency_t, 0.0, 3.0), torch.zeros_like(urgency_t)),
            ],
            dim=1,
        )
        x_order = torch.cat([x_order, service_v2_extra], dim=1)

    t_norm = torch.tensor([t / t_den], dtype=torch.float32)
    pending_norm = torch.tensor([pending_count / max(1.0, float(n))], dtype=torch.float32)
    x_truck = torch.cat(
        [
            coord_n[i],
            t_norm,
            torch.tensor([truck_load_ratio], dtype=torch.float32),
            pending_norm,
            torch.tensor([is_peak], dtype=torch.float32),
        ]
    ).view(1, -1)
    if bool(getattr(env.cfg, "enable_sequence_time_window_features", False)):
        x_truck = torch.cat([x_truck, compute_truck_sequence_tw_features(env, obs)], dim=1)

    v_t = float(env.cfg.vT)
    v_d = float(env.cfg.vD)
    v_d_norm = v_d / max(1e-6, v_t + v_d)
    x_drone = torch.tensor(
        [[
            float(env.cfg.QD),
            float(env.cfg.B),
            float(v_d_norm),
            float(soc),
            float(env.cfg.soc_min_reserve),
            float(env.cfg.payload_energy_factor),
        ]],
        dtype=torch.float32,
    )
    if bool(getattr(env.cfg, "enable_sequence_time_window_features", False)):
        x_drone = torch.cat([x_drone, compute_drone_sequence_tw_features(env, obs)], dim=1)

    data = HeteroData()
    data["order"].x = x_order
    data["truck"].x = x_truck
    data["drone"].x = x_drone

    known_idx = torch.where(known_mask)[0].long()
    dist_i = torch.from_numpy(env.dist_mat[i].copy()).float()
    time_d_i = dist_i / float(env.cfg.vD)
    truck_edge_attr_i = torch.from_numpy(env.get_dense_edge_attr()[i].copy()).float()

    data["truck", "t2o", "order"].edge_index = torch.stack(
        [torch.zeros_like(known_idx), known_idx],
        dim=0,
    )
    data["truck", "t2o", "order"].edge_attr = truck_edge_attr_i[known_idx]
    data["order", "o2t", "truck"].edge_index = torch.stack(
        [known_idx, torch.zeros_like(known_idx)],
        dim=0,
    )
    data["order", "o2t", "truck"].edge_attr = truck_edge_attr_i[known_idx]

    drone_edge_attr_i = torch.stack([dist_i, time_d_i], dim=1)
    data["drone", "d2o", "order"].edge_index = torch.stack(
        [torch.zeros_like(known_idx), known_idx],
        dim=0,
    )
    data["drone", "d2o", "order"].edge_attr = drone_edge_attr_i[known_idx]
    data["order", "o2d", "drone"].edge_index = torch.stack(
        [known_idx, torch.zeros_like(known_idx)],
        dim=0,
    )
    data["order", "o2d", "drone"].edge_attr = drone_edge_attr_i[known_idx]

    o2o_ei_np, o2o_ea_np = env.get_o2o_edges(k_nn=k_nn_orders)
    known_np_mask = known_mask.cpu().numpy()
    o2o_valid = known_np_mask[o2o_ei_np[0]] & known_np_mask[o2o_ei_np[1]]
    data["order", "o2o", "order"].edge_index = torch.from_numpy(o2o_ei_np[:, o2o_valid]).long()
    data["order", "o2o", "order"].edge_attr = torch.from_numpy(o2o_ea_np[o2o_valid]).float()

    truck_mask = torch.from_numpy(env.get_masks()["truck_mask"]).float()
    extra = {
        "truck_mask": truck_mask,
        "served": served,
        "known": known,
        "accepted": accepted,
        "rejected": rejected,
        "expired": expired,
        "cur_i": torch.tensor([i], dtype=torch.long),
        "t": torch.tensor([t], dtype=torch.float32),
        "soc": torch.tensor([soc], dtype=torch.float32),
    }
    return data, extra
