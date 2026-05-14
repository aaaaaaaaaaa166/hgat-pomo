from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.env.instance_gen import REQUEST_DELIVERY, REQUEST_PICKUP
from src.graph.road_aware_features import (
    IntersectionPenaltyConfig,
    RoadAwareMatrices,
    TimeBucketConfig,
    build_proxy_road_aware_matrices,
)
from src.training.sequence_time_window_reward import (
    sequence_tw_pressure,
    sequence_tw_reward_components,
)
from src.training.sequence_time_window_features import compute_global_sequence_tw_stats


@dataclass
class EnvConfig:
    vT: float = 1.0
    vD: float = 1.5
    QD: float = 0.35
    B: float = 6.0
    truck_capacity: float = 3.0
    sT: float = 0.05
    sD: float = 0.03
    depot_service_time: float = 0.10
    allow_wait: bool = True
    idle_to_next_release: bool = True
    traffic_sigma: float = 0.0
    lateness_penalty: float = 0.0
    reject_penalty: float = 0.5
    accept_reward: float = 0.0
    on_time_reward: float = 0.0
    late_count_penalty: float = 0.0
    severe_lateness_penalty: float = 0.0
    unserved_penalty: float = 2.0
    overtime_penalty: float = 1.0
    time_cost_weight: float = 1.0
    energy_cost_weight: float = 0.2
    revenue_scale: float = 1.0
    soc_init: float = 1.0
    soc_min_reserve: float = 0.1
    energy_per_dist: float = 0.08
    truck_energy_per_dist: float = 0.04
    payload_energy_factor: float = 0.4
    drone_takeoff_landing_energy: float = 0.0
    drone_idle_energy_per_time: float = 0.0
    recharge_rate: float = 0.25
    response_window: float = 0.0
    decision_mode: str = "legacy"
    reject_feasible_penalty: float = 0.0
    reject_infeasible_penalty: float = 0.0
    expired_order_penalty: float = 0.0
    edge_mode: str = "static"
    time_dependent: bool = False
    peak_after_served_ratio: float = 0.5
    workday_start: float = 8.0
    workday_end: float = 20.0
    morning_peak_start: float = 8.0
    morning_peak_end: float = 10.0
    evening_peak_start: float = 17.0
    evening_peak_end: float = 19.0
    road_detour_factor: float = 1.18
    road_signal_density: float = 0.006
    road_turn_density: float = 0.010
    road_one_way_ratio: float = 0.10
    road_peak_factor: float = 1.25
    signal_penalty: float = 0.05
    turn_penalty: float = 0.12
    left_turn_penalty: float = 0.08
    u_turn_penalty: float = 0.30
    enable_sequence_time_window_features: bool = False
    enable_sequence_time_window_reward: bool = False
    late_order_penalty: float = 0.0
    lateness_duration_penalty: float = 0.0
    severe_lateness_threshold: float = 10.0
    max_lateness_penalty: float = 0.0
    future_lateness_risk_penalty: float = 0.0
    tight_order_delay_penalty: float = 0.0
    slack_preservation_reward: float = 0.0
    distance_cost_weight: float = 0.0
    workload_balance_weight: float = 0.0
    hard_constraint_violation_penalty: float = 1000000.0
    feature_mode: str = "legacy"


class TruckDroneRendezvousEnv:
    K_NONE = -1

    def __init__(
        self,
        coord: np.ndarray,
        release: np.ndarray,
        demand: np.ndarray,
        due: Optional[np.ndarray] = None,
        cfg: Optional[EnvConfig] = None,
        seed: int = 0,
        road_matrices: Optional[RoadAwareMatrices] = None,
        request_type: Optional[np.ndarray] = None,
        is_dynamic: Optional[np.ndarray] = None,
        revenue: Optional[np.ndarray] = None,
        decision_deadline: Optional[np.ndarray] = None,
        drone_eligible: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg or EnvConfig()
        self.coord = np.asarray(coord, dtype=np.float32)
        self.release = np.asarray(release, dtype=np.float32)
        self.demand = np.asarray(demand, dtype=np.float32)
        self.due = (
            np.full((self.coord.shape[0],), np.inf, dtype=np.float32)
            if due is None
            else np.asarray(due, dtype=np.float32)
        )
        self.request_type = (
            np.where(np.arange(self.coord.shape[0]) == 0, 0, REQUEST_DELIVERY).astype(np.int8)
            if request_type is None
            else np.asarray(request_type, dtype=np.int8)
        )
        default_dynamic = np.zeros((self.coord.shape[0],), dtype=np.int8)
        default_dynamic[1:] = (self.release[1:] > 0).astype(np.int8)
        self.is_dynamic = default_dynamic if is_dynamic is None else np.asarray(is_dynamic, dtype=np.int8)
        self.revenue = (
            np.maximum(self.demand, 0.0).astype(np.float32)
            if revenue is None
            else np.asarray(revenue, dtype=np.float32)
        )
        self.decision_deadline = (
            self.release.copy().astype(np.float32)
            if decision_deadline is None
            else np.asarray(decision_deadline, dtype=np.float32)
        )
        if float(getattr(self.cfg, "response_window", 0.0)) > 0.0:
            dynamic_mask = self.is_dynamic > 0
            self.decision_deadline = self.decision_deadline.copy()
            self.decision_deadline[dynamic_mask] = (
                self.release[dynamic_mask] + float(self.cfg.response_window)
            )
        self.drone_eligible = (
            np.where(np.arange(self.coord.shape[0]) == 0, 0, 1).astype(np.int8)
            if drone_eligible is None
            else np.asarray(drone_eligible, dtype=np.int8)
        )

        assert self.coord.ndim == 2 and self.coord.shape[1] == 2
        assert self.release.shape[0] == self.coord.shape[0]
        assert self.demand.shape[0] == self.coord.shape[0]
        assert self.due.shape[0] == self.coord.shape[0]
        assert self.request_type.shape[0] == self.coord.shape[0]
        assert self.is_dynamic.shape[0] == self.coord.shape[0]
        assert self.revenue.shape[0] == self.coord.shape[0]
        assert self.decision_deadline.shape[0] == self.coord.shape[0]
        assert self.drone_eligible.shape[0] == self.coord.shape[0]

        self.N = self.coord.shape[0] - 1
        if self.cfg.edge_mode not in {"static", "road"}:
            raise ValueError("edge_mode must be 'static' or 'road'")
        self.max_work_time = max(1e-6, float(self.cfg.workday_end) - float(self.cfg.workday_start))
        self.rng = np.random.default_rng(seed)

        diff = self.coord[:, None, :] - self.coord[None, :, :]
        self.dist_mat = np.sqrt((diff * diff).sum(axis=-1) + 1e-12).astype(np.float32)
        self.road_matrices = self._init_road_matrices(road_matrices)
        self._dense_edge_attr: Optional[np.ndarray] = None

        m = self.N + 1
        o_ids = np.arange(m, dtype=np.int64)
        t_ids = np.zeros(m, dtype=np.int64)
        d_ids = np.zeros(m, dtype=np.int64)
        self.edge_index_t2o = np.stack([t_ids, o_ids], axis=0)
        self.edge_index_o2t = np.stack([o_ids, t_ids], axis=0)
        self.edge_index_d2o = np.stack([d_ids, o_ids], axis=0)
        self.edge_index_o2d = np.stack([o_ids, d_ids], axis=0)
        self._o2o_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self.state: Dict[str, Any] = {}
        self.reset()

    def _init_road_matrices(self, road_matrices: Optional[RoadAwareMatrices]) -> Optional[RoadAwareMatrices]:
        if road_matrices is not None:
            return road_matrices
        if self.cfg.edge_mode != "road":
            return None
        penalty = IntersectionPenaltyConfig(
            signal_penalty_sec=float(self.cfg.signal_penalty),
            turn_penalty_sec=float(self.cfg.turn_penalty),
            left_turn_penalty_sec=float(self.cfg.left_turn_penalty),
            u_turn_penalty_sec=float(self.cfg.u_turn_penalty),
        )
        bucket = TimeBucketConfig(
            offpeak_factor=1.0,
            peak_factor=float(self.cfg.road_peak_factor),
        )
        return build_proxy_road_aware_matrices(
            coords=self.coord,
            avg_speed_kmph=max(1e-6, float(self.cfg.vT) * 3.6),
            road_detour_factor=float(self.cfg.road_detour_factor),
            signal_density=float(self.cfg.road_signal_density),
            turn_density=float(self.cfg.road_turn_density),
            one_way_ratio=float(self.cfg.road_one_way_ratio),
            time_bucket=bucket,
            penalty=penalty,
        )

    def copy(self) -> "TruckDroneRendezvousEnv":
        new_env = TruckDroneRendezvousEnv(
            coord=self.coord,
            release=self.release,
            demand=self.demand,
            due=self.due,
            cfg=copy.deepcopy(self.cfg),
            seed=int(self.rng.integers(0, 10**9)),
            road_matrices=self.road_matrices,
            request_type=self.request_type,
            is_dynamic=self.is_dynamic,
            revenue=self.revenue,
            decision_deadline=self.decision_deadline,
            drone_eligible=self.drone_eligible,
        )
        new_env.dist_mat = self.dist_mat
        new_env._dense_edge_attr = self._dense_edge_attr
        new_env.edge_index_t2o = self.edge_index_t2o
        new_env.edge_index_o2t = self.edge_index_o2t
        new_env.edge_index_d2o = self.edge_index_d2o
        new_env.edge_index_o2d = self.edge_index_o2d
        new_env._o2o_cache = self._o2o_cache
        new_env.state = {
            "t": float(self.state["t"]),
            "i": int(self.state["i"]),
            "served": self.state["served"].copy(),
            "accepted": self.state["accepted"].copy(),
            "rejected": self.state["rejected"].copy(),
            "expired": self.state.get("expired", np.zeros_like(self.state["served"])).copy(),
            "known": self.state["known"].copy(),
            "loaded": self.state["loaded"].copy(),
            "accept_time": self.state.get(
                "accept_time", np.full((self.N + 1,), np.nan, dtype=np.float32)
            ).copy(),
            "finish_time": self.state.get(
                "finish_time", np.full((self.N + 1,), np.nan, dtype=np.float32)
            ).copy(),
            "reject_reason": self.state.get(
                "reject_reason", np.asarray([""] * (self.N + 1), dtype=object)
            ).copy(),
            "soc": float(self.state["soc"]),
            "truck_pickup_load": float(self.state["truck_pickup_load"]),
            "pending_queue": list(self.state["pending_queue"]),
        }
        return new_env

    def get_clock_time(self, t_elapsed: Optional[float] = None) -> float:
        elapsed = float(self.state["t"]) if t_elapsed is None else float(t_elapsed)
        return float(self.cfg.workday_start) + elapsed

    def _is_peak_clock(self, clock_t: float) -> bool:
        morning = float(self.cfg.morning_peak_start) <= clock_t < float(self.cfg.morning_peak_end)
        evening = float(self.cfg.evening_peak_start) <= clock_t < float(self.cfg.evening_peak_end)
        return bool(morning or evening)

    def get_time_bucket(self, t_elapsed: Optional[float] = None) -> str:
        if self.cfg.edge_mode != "road" or not self.cfg.time_dependent:
            return "offpeak"
        return "peak" if self._is_peak_clock(self.get_clock_time(t_elapsed=t_elapsed)) else "offpeak"

    def get_is_peak(self, t_elapsed: Optional[float] = None) -> float:
        return 1.0 if self.get_time_bucket(t_elapsed=t_elapsed) == "peak" else 0.0

    def get_dense_edge_attr(self) -> np.ndarray:
        if self._dense_edge_attr is not None:
            return self._dense_edge_attr
        if self.road_matrices is not None:
            self._dense_edge_attr = self.road_matrices.edge_attr().astype(np.float32)
            return self._dense_edge_attr

        dist = self.dist_mat.astype(np.float32)
        time_t = dist / float(self.cfg.vT)
        zeros = np.zeros_like(dist, dtype=np.float32)
        ones = np.ones_like(dist, dtype=np.float32)
        self._dense_edge_attr = np.stack(
            [dist, time_t, time_t, zeros, zeros, zeros, zeros, ones],
            axis=-1,
        ).astype(np.float32)
        return self._dense_edge_attr

    def _truck_time_matrix(self, bucket: Optional[str] = None) -> np.ndarray:
        dense = self.get_dense_edge_attr()
        if self.cfg.edge_mode == "road":
            selected_bucket = bucket or self.get_time_bucket()
            cost_idx = 2 if selected_bucket == "peak" else 1
            return dense[..., cost_idx]
        return dense[..., 1]

    def get_o2o_edges(self, k_nn: int) -> Tuple[np.ndarray, np.ndarray]:
        k_nn = int(k_nn)
        if k_nn <= 0:
            raise ValueError("k_nn must be >= 1")
        if k_nn in self._o2o_cache:
            return self._o2o_cache[k_nn]

        m = self.N + 1
        k = min(k_nn, m - 1)
        if k <= 0:
            raise ValueError("No valid neighbors (M too small)")

        dist = self.dist_mat.copy()
        np.fill_diagonal(dist, 1e9)
        nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        row = np.arange(m)[:, None]
        order = np.argsort(dist[row, nn], axis=1)
        nn = np.take_along_axis(nn, order, axis=1)

        src = np.repeat(np.arange(m, dtype=np.int64), k)
        dst = nn.reshape(-1).astype(np.int64)
        edge_index = np.stack([src, dst], axis=0)
        edge_attr = self.get_dense_edge_attr()[src, dst].astype(np.float32)
        self._o2o_cache[k_nn] = (edge_index, edge_attr)
        return edge_index, edge_attr

    def _traffic_factor(self, i: int, j: int) -> float:
        if i == j or float(self.cfg.traffic_sigma) <= 0.0:
            return 1.0
        factor = self.rng.normal(loc=1.0, scale=float(self.cfg.traffic_sigma))
        return float(np.clip(factor, 0.5, 2.0))

    def _tau_truck(self, i: int, j: int, apply_traffic: bool = False, bucket: Optional[str] = None) -> float:
        if i == j:
            return 0.0
        base = float(self._truck_time_matrix(bucket=bucket)[i, j])
        if apply_traffic:
            base *= self._traffic_factor(i, j)
        return base

    def _tau_drone(self, i: int, k: int, j: int) -> float:
        return (float(self.dist_mat[i, k]) + float(self.dist_mat[k, j])) / float(self.cfg.vD)

    def _request_weight(self, node: int) -> float:
        return float(max(0.0, self.demand[node]))

    def _delivery_load(self, accepted: np.ndarray, served: np.ndarray, loaded: np.ndarray) -> float:
        mask = (
            (accepted > 0)
            & (served == 0)
            & (loaded > 0)
            & (self.request_type == REQUEST_DELIVERY)
        )
        return float(self.demand[mask].sum())

    def _truck_total_load(
        self,
        accepted: np.ndarray,
        served: np.ndarray,
        loaded: np.ndarray,
        truck_pickup_load: float,
    ) -> float:
        return self._delivery_load(accepted=accepted, served=served, loaded=loaded) + float(truck_pickup_load)

    def _pending_delivery_backlog(
        self,
        accepted: np.ndarray,
        served: np.ndarray,
        rejected: np.ndarray,
        loaded: np.ndarray,
    ) -> bool:
        mask = (
            (accepted > 0)
            & (served == 0)
            & (rejected == 0)
            & (self.request_type == REQUEST_DELIVERY)
            & (loaded == 0)
        )
        return bool(np.any(mask))

    def _truck_energy(self, i: int, j: int, travel_load: float) -> float:
        dense = self.get_dense_edge_attr()
        dist = float(dense[i, j, 0])
        coef = float(self.cfg.truck_energy_per_dist)
        payload_factor = float(self.cfg.payload_energy_factor)
        return coef * dist * (1.0 + payload_factor * max(0.0, float(travel_load)))

    def _is_released(self, node: int, t: float) -> bool:
        return True if node == 0 else (t >= float(self.release[node]))

    def _next_release_time(
        self,
        t: float,
        known: np.ndarray,
        served: np.ndarray,
        rejected: np.ndarray,
    ) -> Optional[float]:
        unresolved = np.where(
            (np.arange(self.N + 1) > 0)
            & (known == 0)
            & (served == 0)
            & (rejected == 0)
            & (self.is_dynamic > 0)
        )[0]
        if unresolved.size == 0:
            return None
        fut = self.release[unresolved]
        fut = fut[fut > t]
        if fut.size == 0:
            return None
        return float(fut.min())

    def _lateness(self, node: int, finish_t: float) -> float:
        if node == 0:
            return 0.0
        due_t = float(self.due[node])
        if not np.isfinite(due_t):
            return 0.0
        return max(0.0, finish_t - due_t)

    def _request_feasible_for_truck(
        self,
        node: int,
        accepted: np.ndarray,
        served: np.ndarray,
        loaded: np.ndarray,
        truck_pickup_load: float,
        t: float,
    ) -> bool:
        if node <= 0 or served[node] == 1 or accepted[node] == 0 or not self._is_released(node, t):
            return False

        weight = self._request_weight(node)
        delivery_load = self._delivery_load(accepted=accepted, served=served, loaded=loaded)
        truck_load = delivery_load + float(truck_pickup_load)
        req_type = int(self.request_type[node])

        if req_type == REQUEST_DELIVERY:
            return bool(loaded[node] > 0 and weight <= delivery_load + 1e-6 and weight <= truck_load + 1e-6)
        if req_type == REQUEST_PICKUP:
            return bool(truck_load + weight <= float(self.cfg.truck_capacity) + 1e-6)
        return False

    def _drone_energy(self, i: int, k: int, j: int) -> float:
        weight = self._request_weight(k)
        leg_out = float(self.dist_mat[i, k])
        leg_back = float(self.dist_mat[k, j])
        req_type = int(self.request_type[k])

        out_payload = weight if req_type == REQUEST_DELIVERY else 0.0
        back_payload = weight if req_type == REQUEST_PICKUP else 0.0
        coef = float(self.cfg.energy_per_dist)
        payload_factor = float(self.cfg.payload_energy_factor)
        distance_energy = coef * (
            leg_out * (1.0 + payload_factor * out_payload)
            + leg_back * (1.0 + payload_factor * back_payload)
        )
        return float(distance_energy + float(self.cfg.drone_takeoff_landing_energy))

    def _combined_load_feasible(
        self,
        j: int,
        k: int,
        accepted: np.ndarray,
        served: np.ndarray,
        loaded: np.ndarray,
        truck_pickup_load: float,
    ) -> bool:
        delivery_load = self._delivery_load(accepted=accepted, served=served, loaded=loaded)
        delivery_need = 0.0
        pickup_gain = 0.0
        for node in (j, k):
            if node in {self.K_NONE, 0}:
                continue
            req_type = int(self.request_type[node])
            weight = self._request_weight(node)
            if req_type == REQUEST_DELIVERY:
                if loaded[node] == 0:
                    return False
                delivery_need += weight
            elif req_type == REQUEST_PICKUP:
                pickup_gain += weight

        if delivery_need > delivery_load + 1e-6:
            return False
        end_load = delivery_load - delivery_need + float(truck_pickup_load) + pickup_gain
        return bool(end_load <= float(self.cfg.truck_capacity) + 1e-6)

    def _drone_feasible(
        self,
        i: int,
        j: int,
        k: int,
        t: float,
        accepted: np.ndarray,
        served: np.ndarray,
        loaded: np.ndarray,
        soc: float,
        truck_pickup_load: float,
    ) -> bool:
        if k <= 0 or k == j or served[k] == 1 or accepted[k] == 0 or not self._is_released(k, t):
            return False
        if int(self.drone_eligible[k]) == 0:
            return False
        if float(self.demand[k]) > float(self.cfg.QD):
            return False
        if self._tau_drone(i, k, j) + float(self.cfg.sD) > float(self.cfg.B):
            return False
        if not self._combined_load_feasible(
            j=j,
            k=k,
            accepted=accepted,
            served=served,
            loaded=loaded,
            truck_pickup_load=truck_pickup_load,
        ):
            return False
        need = self._drone_energy(i, k, j)
        if need > max(0.0, soc - float(self.cfg.soc_min_reserve)):
            return False
        return True

    def _reload_from_depot(
        self,
        accepted: np.ndarray,
        served: np.ndarray,
        rejected: np.ndarray,
        loaded: np.ndarray,
        truck_pickup_load: float,
    ) -> np.ndarray:
        loaded = loaded.copy()
        loaded[(served > 0) | (rejected > 0)] = 0
        delivery_mask = (
            (accepted > 0)
            & (served == 0)
            & (rejected == 0)
            & (self.request_type == REQUEST_DELIVERY)
        )
        loaded[delivery_mask] = 0

        remaining_capacity = max(0.0, float(self.cfg.truck_capacity) - float(truck_pickup_load))
        delivery_candidates = np.where(delivery_mask)[0]
        if delivery_candidates.size == 0 or remaining_capacity <= 1e-6:
            return loaded

        due_vals = self.due[delivery_candidates].copy()
        due_vals[~np.isfinite(due_vals)] = 1e9
        release_vals = self.release[delivery_candidates]
        order = np.lexsort((release_vals, due_vals))
        for node in delivery_candidates[order]:
            weight = self._request_weight(int(node))
            if weight <= remaining_capacity + 1e-6:
                loaded[int(node)] = 1
                remaining_capacity -= weight
        return loaded

    def _refresh_pending_requests(
        self,
        t: float,
        known: np.ndarray,
        accepted: np.ndarray,
        rejected: np.ndarray,
        served: np.ndarray,
        pending_queue: List[int],
    ) -> Tuple[np.ndarray, List[int]]:
        known = known.copy()
        queue = [int(x) for x in pending_queue if served[int(x)] == 0 and accepted[int(x)] == 0 and rejected[int(x)] == 0]
        existing = set(queue)

        new_nodes = np.where(
            (np.arange(self.N + 1) > 0)
            & (self.is_dynamic > 0)
            & (known == 0)
            & (self.release <= t + 1e-9)
            & (served == 0)
            & (accepted == 0)
            & (rejected == 0)
        )[0]
        if new_nodes.size > 0:
            known[new_nodes] = 1
            due_vals = self.decision_deadline[new_nodes]
            order = np.argsort(due_vals)
            for node in new_nodes[order]:
                node = int(node)
                if node not in existing:
                    queue.append(node)
                    existing.add(node)
        return known, queue

    def _resolve_pending_queue(
        self,
        t: float,
        known: np.ndarray,
        accepted: np.ndarray,
        rejected: np.ndarray,
        served: np.ndarray,
        pending_queue: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, List[int], float, List[int]]:
        known, queue = self._refresh_pending_requests(
            t=t,
            known=known,
            accepted=accepted,
            rejected=rejected,
            served=served,
            pending_queue=pending_queue,
        )

        expired_nodes: List[int] = []
        kept_queue: List[int] = []
        reject_cost = 0.0
        for node in queue:
            node = int(node)
            if served[node] == 1 or accepted[node] == 1 or rejected[node] == 1:
                continue
            if t > float(self.decision_deadline[node]) + 1e-9:
                rejected[node] = 1
                expired_nodes.append(node)
                reject_cost += float(self.cfg.reject_penalty)
            else:
                kept_queue.append(node)
        return known, rejected, kept_queue, reject_cost, expired_nodes

    def _has_unresolved_requests(self, served: np.ndarray, rejected: np.ndarray) -> bool:
        unresolved = np.where((np.arange(self.N + 1) > 0) & (served == 0) & (rejected == 0))[0]
        return bool(unresolved.size > 0)

    def _current_decision_request(self, pending_queue: List[int]) -> Optional[int]:
        return int(pending_queue[0]) if len(pending_queue) > 0 else None

    def _active_deadlines(
        self,
        known: np.ndarray,
        accepted: np.ndarray,
        served: np.ndarray,
        rejected: np.ndarray,
    ) -> np.ndarray:
        active = self.due.copy()
        active[known == 0] = np.inf
        pending_mask = (
            (np.arange(self.N + 1) > 0)
            & (known > 0)
            & (self.is_dynamic > 0)
            & (accepted == 0)
            & (served == 0)
            & (rejected == 0)
        )
        active[pending_mask] = self.decision_deadline[pending_mask]
        return active.astype(np.float32)

    def reset(self) -> Dict[str, Any]:
        served = np.zeros((self.N + 1,), dtype=np.int8)
        accepted = np.zeros((self.N + 1,), dtype=np.int8)
        rejected = np.zeros((self.N + 1,), dtype=np.int8)
        known = np.zeros((self.N + 1,), dtype=np.int8)
        loaded = np.zeros((self.N + 1,), dtype=np.int8)
        expired = np.zeros((self.N + 1,), dtype=np.int8)
        accept_time = np.full((self.N + 1,), np.nan, dtype=np.float32)
        finish_time = np.full((self.N + 1,), np.nan, dtype=np.float32)
        reject_reason = np.asarray([""] * (self.N + 1), dtype=object)
        pending_queue: List[int] = []

        accepted[(np.arange(self.N + 1) > 0) & (self.is_dynamic == 0)] = 1
        accept_time[(np.arange(self.N + 1) > 0) & (self.is_dynamic == 0)] = 0.0
        known[(np.arange(self.N + 1) == 0) | (accepted > 0)] = 1

        truck_pickup_load = 0.0
        loaded = self._reload_from_depot(
            accepted=accepted,
            served=served,
            rejected=rejected,
            loaded=loaded,
            truck_pickup_load=truck_pickup_load,
        )
        known, rejected, pending_queue, _, expired_nodes = self._resolve_pending_queue(
            t=0.0,
            known=known,
            accepted=accepted,
            rejected=rejected,
            served=served,
            pending_queue=pending_queue,
        )
        for node in expired_nodes:
            expired[int(node)] = 1
            reject_reason[int(node)] = "expired_response_window"

        self.state = {
            "t": 0.0,
            "i": 0,
            "served": served,
            "accepted": accepted,
            "rejected": rejected,
            "expired": expired,
            "known": known,
            "loaded": loaded,
            "accept_time": accept_time,
            "finish_time": finish_time,
            "reject_reason": reject_reason,
            "soc": float(np.clip(float(self.cfg.soc_init), 0.0, 1.0)),
            "truck_pickup_load": truck_pickup_load,
            "pending_queue": pending_queue,
        }
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        t = float(self.state["t"])
        served = self.state["served"].copy()
        accepted = self.state["accepted"].copy()
        rejected = self.state["rejected"].copy()
        expired = self.state.get("expired", np.zeros_like(served)).copy()
        known = self.state["known"].copy()
        loaded = self.state["loaded"].copy()
        accept_time = self.state.get("accept_time", np.full((self.N + 1,), np.nan, dtype=np.float32)).copy()
        finish_time = self.state.get("finish_time", np.full((self.N + 1,), np.nan, dtype=np.float32)).copy()
        reject_reason = self.state.get("reject_reason", np.asarray([""] * (self.N + 1), dtype=object)).copy()
        truck_pickup_load = float(self.state["truck_pickup_load"])
        truck_load = self._truck_total_load(
            accepted=accepted,
            served=served,
            loaded=loaded,
            truck_pickup_load=truck_pickup_load,
        )
        pending_queue = list(self.state["pending_queue"])
        current_request = self._current_decision_request(pending_queue)
        obs = {
            "t": t,
            "clock_time": self.get_clock_time(t_elapsed=t),
            "i": int(self.state["i"]),
            "served": served,
            "accepted": accepted,
            "rejected": rejected,
            "expired": expired,
            "known": known,
            "loaded": loaded,
            "accept_time": accept_time,
            "finish_time": finish_time,
            "reject_reason": reject_reason,
            "soc": float(self.state["soc"]),
            "truck_pickup_load": truck_pickup_load,
            "truck_load": truck_load,
            "truck_load_ratio": truck_load / max(1e-6, float(self.cfg.truck_capacity)),
            "pending_queue": pending_queue,
            "pending_count": len(pending_queue),
            "current_decision_request": -1 if current_request is None else int(current_request),
            "active_deadlines": self._active_deadlines(
                known=known,
                accepted=accepted,
                served=served,
                rejected=rejected,
            ),
            "time_bucket": self.get_time_bucket(t_elapsed=t),
            "is_peak": self.get_is_peak(t_elapsed=t),
            "remaining_work_time": float(self.max_work_time - t),
        }
        if bool(self.cfg.enable_sequence_time_window_features):
            stats = compute_global_sequence_tw_stats(self, obs)
            obs.update(
                {
                    "sequence_tw_stats": stats,
                    "remaining_orders_count": stats["remaining_orders_count"],
                    "remaining_tight_orders_count": stats["remaining_tight_orders_count"],
                    "minimum_slack_among_remaining_orders": stats["minimum_slack_among_remaining_orders"],
                    "average_slack_among_remaining_orders": stats["average_slack_among_remaining_orders"],
                    "number_of_orders_predicted_late_if_delayed": stats[
                        "number_of_orders_predicted_late_if_delayed"
                    ],
                    "current_global_lateness_risk": stats["current_global_lateness_risk"],
                    "workload_balance_score": stats["workload_balance_score"],
                    "future_available_time": t,
                    "predicted_finish_time": t,
                    "utilization_so_far": 1.0 - float(self.state["soc"]),
                }
            )
        return obs

    def get_masks(self, j: Optional[int] = None) -> Dict[str, Any]:
        t = float(self.state["t"])
        i = int(self.state["i"])
        served = self.state["served"]
        accepted = self.state["accepted"]
        rejected = self.state["rejected"]
        loaded = self.state["loaded"]
        soc = float(self.state["soc"])
        truck_pickup_load = float(self.state["truck_pickup_load"])
        pending_queue = list(self.state["pending_queue"])

        current_request = self._current_decision_request(pending_queue)
        truck_mask = np.zeros((self.N + 1,), dtype=np.int8)

        if current_request is not None:
            truck_mask[0] = 1
            truck_mask[int(current_request)] = 1
            if j is None:
                return {"truck_mask": truck_mask}
            drone_mask = np.zeros((self.N + 1,), dtype=np.int8)
            return {"truck_mask": truck_mask, "drone_mask": drone_mask, "k_none_feasible": True}

        for node in range(1, self.N + 1):
            if self._request_feasible_for_truck(
                node=node,
                accepted=accepted,
                served=served,
                loaded=loaded,
                truck_pickup_load=truck_pickup_load,
                t=t,
            ):
                truck_mask[node] = 1

        need_depot = (
            i != 0
            or truck_pickup_load > 1e-6
            or self._pending_delivery_backlog(
                accepted=accepted,
                served=served,
                rejected=rejected,
                loaded=loaded,
            )
            or int(truck_mask[1:].sum()) == 0
        )
        if need_depot:
            truck_mask[0] = 1

        next_release = self._next_release_time(
            t=t,
            known=self.state["known"],
            served=served,
            rejected=rejected,
        )
        if int(truck_mask[1:].sum()) == 0 and self.cfg.allow_wait and next_release is not None:
            truck_mask[i] = 1

        if j is None:
            return {"truck_mask": truck_mask}

        if not (0 <= int(j) <= self.N):
            raise ValueError("j out of range")

        drone_mask = np.zeros((self.N + 1,), dtype=np.int8)
        if truck_mask[int(j)] == 0:
            return {"truck_mask": truck_mask, "drone_mask": drone_mask, "k_none_feasible": True}

        for k in range(1, self.N + 1):
            if self._drone_feasible(
                i=i,
                j=int(j),
                k=k,
                t=t,
                accepted=accepted,
                served=served,
                loaded=loaded,
                soc=soc,
                truck_pickup_load=truck_pickup_load,
            ):
                drone_mask[k] = 1

        return {"truck_mask": truck_mask, "drone_mask": drone_mask, "k_none_feasible": True}

    def _step_done(self, served: np.ndarray, rejected: np.ndarray, pending_queue: List[int]) -> bool:
        return (not self._has_unresolved_requests(served=served, rejected=rejected)) and (len(pending_queue) == 0)

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        k, j = int(action[0]), int(action[1])
        t = float(self.state["t"])
        i = int(self.state["i"])
        served = self.state["served"].copy()
        accepted = self.state["accepted"].copy()
        rejected = self.state["rejected"].copy()
        expired = self.state.get("expired", np.zeros_like(served)).copy()
        known = self.state["known"].copy()
        loaded = self.state["loaded"].copy()
        accept_time = self.state.get("accept_time", np.full((self.N + 1,), np.nan, dtype=np.float32)).copy()
        finish_time = self.state.get("finish_time", np.full((self.N + 1,), np.nan, dtype=np.float32)).copy()
        reject_reason = self.state.get("reject_reason", np.asarray([""] * (self.N + 1), dtype=object)).copy()
        soc = float(self.state["soc"])
        truck_pickup_load = float(self.state["truck_pickup_load"])
        pending_queue = list(self.state["pending_queue"])
        dense = self.get_dense_edge_attr()

        truck_load_prev = self._truck_total_load(
            accepted=accepted,
            served=served,
            loaded=loaded,
            truck_pickup_load=truck_pickup_load,
        )
        pre_sequence_tw_pressure = {}
        if bool(self.cfg.enable_sequence_time_window_reward):
            pre_sequence_tw_pressure = sequence_tw_pressure(
                self,
                t=t,
                i=i,
                accepted=accepted,
                served=served,
                rejected=rejected,
                loaded=loaded,
                truck_pickup_load=truck_pickup_load,
            )
        current_request = self._current_decision_request(pending_queue)

        if current_request is not None:
            masks = self.get_masks()
            if masks["truck_mask"][j] == 0:
                raise ValueError(f"Infeasible decision action j={j} for pending request={current_request}")
            if k != self.K_NONE:
                raise ValueError("Decision steps only support k = K_NONE")

            pending_queue.pop(0)
            reject_cost = 0.0
            decision = "reject" if j == 0 else "accept"
            if decision == "reject":
                rejected[current_request] = 1
                reject_reason[current_request] = "policy_reject"
                reject_cost = float(self.cfg.reject_penalty)
                accept_reward = 0.0
            else:
                accepted[current_request] = 1
                accept_time[current_request] = float(t)
                accept_reward = float(self.cfg.accept_reward)
                if i == 0:
                    loaded = self._reload_from_depot(
                        accepted=accepted,
                        served=served,
                        rejected=rejected,
                        loaded=loaded,
                        truck_pickup_load=truck_pickup_load,
                    )

            known, rejected, pending_queue, auto_reject_cost, expired_nodes = self._resolve_pending_queue(
                t=t,
                known=known,
                accepted=accepted,
                rejected=rejected,
                served=served,
                pending_queue=pending_queue,
            )
            for node in expired_nodes:
                expired[int(node)] = 1
                reject_reason[int(node)] = "expired_response_window"
            expired_order_cost = float(self.cfg.expired_order_penalty) * float(len(expired_nodes))
            step_cost = reject_cost + auto_reject_cost + expired_order_cost - accept_reward

            self.state = {
                "t": t,
                "i": i,
                "served": served,
                "accepted": accepted,
                "rejected": rejected,
                "expired": expired,
                "known": known,
                "loaded": loaded,
                "accept_time": accept_time,
                "finish_time": finish_time,
                "reject_reason": reject_reason,
                "soc": soc,
                "truck_pickup_load": truck_pickup_load,
                "pending_queue": pending_queue,
            }

            done = self._step_done(served=served, rejected=rejected, pending_queue=pending_queue)
            info = {
                "phase": "decision",
                "decision": decision,
                "decision_node": int(current_request),
                "dt": 0.0,
                "truck_time": 0.0,
                "drone_time": 0.0,
                "wait_time": 0.0,
                "lateness": 0.0,
                "time_cost": 0.0,
                "lateness_cost": 0.0,
                "energy_cost": 0.0,
                "overtime_cost": 0.0,
                "reject_cost": float(reject_cost),
                "auto_reject_cost": float(auto_reject_cost),
                "accept_reward": float(accept_reward),
                "on_time_reward": 0.0,
                "late_count_cost": 0.0,
                "severe_lateness_cost": 0.0,
                "expired_nodes": expired_nodes,
                "reject_reason": str(reject_reason[current_request]),
                "served_nodes": [],
                "service_finish_times": {},
                "service_lateness": {},
                "revenue_gained": 0.0,
                "step_cost": float(step_cost),
                "soc_prev": float(soc),
                "soc_next": float(soc),
                "truck_load_prev": float(truck_load_prev),
                "truck_load_next": float(
                    self._truck_total_load(
                        accepted=accepted,
                        served=served,
                        loaded=loaded,
                        truck_pickup_load=truck_pickup_load,
                    )
                ),
                "energy_use": 0.0,
                "drone_idle_energy_use": 0.0,
                "truck_energy_use": 0.0,
                "i": int(i),
                "j": int(j),
                "k": int(k),
                "time_bucket": self.get_time_bucket(t_elapsed=t),
                "edge_mode": self.cfg.edge_mode,
                "road_distance": 0.0,
                "signal_count": 0.0,
                "turn_count": 0.0,
                "left_turn_count": 0.0,
                "u_turn_count": 0.0,
                "one_way_factor": 1.0,
                "reward_components": {
                    "reject_cost": float(reject_cost),
                    "auto_reject_cost": float(auto_reject_cost),
                    "expired_order_cost": float(expired_order_cost),
                    "accept_reward": float(accept_reward),
                    "sequence_tw_total_cost": 0.0,
                },
            }
            return self.get_obs(), -float(step_cost), done, info

        masks = self.get_masks()
        if masks["truck_mask"][j] == 0:
            raise ValueError(f"Infeasible j={j} at t={t}, i={i}")

        if k != self.K_NONE:
            if not (1 <= k <= self.N):
                raise ValueError("k out of range")
            if self.get_masks(j=j)["drone_mask"][k] == 0:
                raise ValueError(f"Infeasible k={k} for (i={i}, j={j}, t={t})")

        time_bucket = self.get_time_bucket(t_elapsed=t)
        base_overtime = max(0.0, t - self.max_work_time)

        delivery_release = 0.0
        if k != self.K_NONE and int(self.request_type[k]) == REQUEST_DELIVERY:
            delivery_release = self._request_weight(k)
        truck_travel_load = max(0.0, truck_load_prev - delivery_release)

        truck_travel = self._tau_truck(i, j, apply_traffic=True, bucket=time_bucket)
        truck_service = 0.0
        if j == 0:
            if i != 0 or truck_pickup_load > 1e-6 or self._pending_delivery_backlog(accepted, served, rejected, loaded):
                truck_service = float(self.cfg.depot_service_time)
        elif self._request_feasible_for_truck(
            node=j,
            accepted=accepted,
            served=served,
            loaded=loaded,
            truck_pickup_load=truck_pickup_load,
            t=t,
        ):
            truck_service = float(self.cfg.sT)
        truck_time = float(truck_travel + truck_service)

        drone_time = 0.0
        drone_energy = 0.0
        if k != self.K_NONE:
            drone_time = float(self._tau_drone(i, k, j) + float(self.cfg.sD))
            drone_energy = float(self._drone_energy(i, k, j))

        truck_energy = float(self._truck_energy(i, j, travel_load=truck_travel_load))
        wait_time = 0.0
        dt = float(max(truck_time, drone_time))

        next_release = self._next_release_time(
            t=t,
            known=known,
            served=served,
            rejected=rejected,
        )
        if dt <= 1e-9 and self.cfg.idle_to_next_release and next_release is not None:
            wait_time = max(0.0, float(next_release) - t)
            dt = wait_time
            truck_time = wait_time

        drone_idle_energy = float(self.cfg.drone_idle_energy_per_time) * float(wait_time)
        drone_energy += drone_idle_energy

        t_next = t + dt
        finish_j = t + truck_time
        finish_k = t + drone_time
        i_next = int(j)

        lateness = 0.0
        revenue_gained = 0.0
        served_nodes: List[int] = []
        service_finish_times: Dict[str, float] = {}
        service_lateness: Dict[str, float] = {}
        if j != 0 and truck_service > 0.0:
            served[j] = 1
            finish_time[j] = float(finish_j)
            if int(self.request_type[j]) == REQUEST_DELIVERY:
                loaded[j] = 0
            elif int(self.request_type[j]) == REQUEST_PICKUP:
                truck_pickup_load += self._request_weight(j)
            node_late = self._lateness(j, finish_j)
            lateness += node_late
            served_nodes.append(int(j))
            service_finish_times[str(int(j))] = float(finish_j)
            service_lateness[str(int(j))] = float(node_late)
            revenue_gained += float(self.revenue[j]) * float(self.cfg.revenue_scale)

        if k != self.K_NONE:
            served[k] = 1
            finish_time[k] = float(finish_k)
            if int(self.request_type[k]) == REQUEST_DELIVERY:
                loaded[k] = 0
            elif int(self.request_type[k]) == REQUEST_PICKUP:
                truck_pickup_load += self._request_weight(k)
            node_late = self._lateness(k, finish_k)
            lateness += node_late
            served_nodes.append(int(k))
            service_finish_times[str(int(k))] = float(finish_k)
            service_lateness[str(int(k))] = float(node_late)
            revenue_gained += float(self.revenue[k]) * float(self.cfg.revenue_scale)

        if k == self.K_NONE:
            soc_after = max(0.0, soc - drone_energy)
            recharge_time = dt
        else:
            soc_after = max(0.0, soc - drone_energy)
            recharge_time = max(0.0, dt - drone_time)
        soc_next = min(1.0, soc_after + float(self.cfg.recharge_rate) * recharge_time)

        if i_next == 0:
            truck_pickup_load = 0.0
            loaded = self._reload_from_depot(
                accepted=accepted,
                served=served,
                rejected=rejected,
                loaded=loaded,
                truck_pickup_load=truck_pickup_load,
            )

        known, rejected, pending_queue, auto_reject_cost, expired_nodes = self._resolve_pending_queue(
            t=t_next,
            known=known,
            accepted=accepted,
            rejected=rejected,
            served=served,
            pending_queue=pending_queue,
        )
        for node in expired_nodes:
            expired[int(node)] = 1
            reject_reason[int(node)] = "expired_response_window"
        expired_order_cost = float(self.cfg.expired_order_penalty) * float(len(expired_nodes))

        overtime_inc = max(0.0, t_next - self.max_work_time) - base_overtime
        time_cost = float(self.cfg.time_cost_weight) * dt
        late_served_count = float(sum(1 for v in service_lateness.values() if float(v) > 1e-9))
        on_time_served_count = float(len(service_lateness) - int(late_served_count))
        max_step_lateness = float(max([0.0] + [float(v) for v in service_lateness.values()]))
        on_time_reward = float(self.cfg.on_time_reward) * on_time_served_count
        drone_distance = 0.0
        if k != self.K_NONE:
            drone_distance = float(self.dist_mat[i, k]) + float(self.dist_mat[k, j])
        truck_distance = float(dense[i, j, 0])
        truck_load_next_for_cost = self._truck_total_load(
            accepted=accepted,
            served=served,
            loaded=loaded,
            truck_pickup_load=truck_pickup_load,
        )
        reward_components: Dict[str, float] = {}
        if bool(self.cfg.enable_sequence_time_window_reward):
            post_pressure = sequence_tw_pressure(
                self,
                t=t_next,
                i=i_next,
                accepted=accepted,
                served=served,
                rejected=rejected,
                loaded=loaded,
                truck_pickup_load=truck_pickup_load,
            )
            hard_info = {
                "k": int(k),
                "energy_use": float(drone_energy),
                "soc_prev": float(soc),
                "drone_time": float(drone_time),
                "truck_load_next": float(truck_load_next_for_cost),
            }
            seq_components = sequence_tw_reward_components(
                self,
                pre_pressure=pre_sequence_tw_pressure,
                post_pressure=post_pressure,
                dt=dt,
                late_served_count=late_served_count,
                total_lateness=lateness,
                max_step_lateness=max_step_lateness,
                truck_distance=truck_distance,
                drone_distance=drone_distance,
                energy_use=truck_energy + drone_energy,
                info_for_hard=hard_info,
            )
            lateness_cost = float(seq_components["lateness_duration_cost"])
            late_count_cost = float(seq_components["late_order_cost"])
            severe_lateness_cost = float(seq_components["severe_lateness_cost"])
            energy_cost = float(seq_components["energy_cost"])
            distance_cost = float(seq_components["distance_cost"])
            max_lateness_cost = float(seq_components["max_lateness_cost"])
            future_lateness_risk_cost = float(seq_components["future_lateness_risk_cost"])
            tight_order_delay_cost = float(seq_components["tight_order_delay_cost"])
            slack_reward = float(seq_components["slack_preservation_reward"])
            hard_constraint_cost = float(seq_components["hard_constraint_cost"])
            severe_future_lateness_cost = float(seq_components["severe_future_lateness_cost"])
            max_lateness_proxy_cost = float(seq_components["max_lateness_proxy_cost"])
            workload_imbalance_cost = float(seq_components["workload_imbalance_cost"])
            reward_components.update(seq_components)
        else:
            lateness_cost = float(self.cfg.lateness_penalty) * lateness
            late_count_cost = float(self.cfg.late_count_penalty) * late_served_count
            severe_lateness_cost = float(self.cfg.severe_lateness_penalty) * max_step_lateness
            energy_cost = float(self.cfg.energy_cost_weight) * (truck_energy + drone_energy)
            distance_cost = 0.0
            max_lateness_cost = 0.0
            future_lateness_risk_cost = 0.0
            tight_order_delay_cost = 0.0
            slack_reward = 0.0
            hard_constraint_cost = 0.0
            severe_future_lateness_cost = 0.0
            max_lateness_proxy_cost = 0.0
            workload_imbalance_cost = 0.0
        overtime_cost = float(self.cfg.overtime_penalty) * overtime_inc
        step_cost = (
            time_cost
            + lateness_cost
            + late_count_cost
            + severe_lateness_cost
            + energy_cost
            + distance_cost
            + max_lateness_cost
            + future_lateness_risk_cost
            + tight_order_delay_cost
            + severe_future_lateness_cost
            + max_lateness_proxy_cost
            + workload_imbalance_cost
            + hard_constraint_cost
            + overtime_cost
            + auto_reject_cost
            + expired_order_cost
            - revenue_gained
            - on_time_reward
            - slack_reward
        )
        reward_components.update(
            {
                "time_cost": float(time_cost),
                "lateness_cost": float(lateness_cost),
                "late_count_cost": float(late_count_cost),
                "severe_lateness_cost": float(severe_lateness_cost),
                "max_lateness_cost": float(max_lateness_cost),
                "future_lateness_risk_cost": float(future_lateness_risk_cost),
                "tight_order_delay_cost": float(tight_order_delay_cost),
                "severe_future_lateness_cost": float(severe_future_lateness_cost),
                "max_lateness_proxy_cost": float(max_lateness_proxy_cost),
                "workload_imbalance_cost": float(workload_imbalance_cost),
                "slack_preservation_reward": float(slack_reward),
                "energy_cost": float(energy_cost),
                "distance_cost": float(distance_cost),
                "hard_constraint_cost": float(hard_constraint_cost),
                "overtime_cost": float(overtime_cost),
                "auto_reject_cost": float(auto_reject_cost),
                "expired_order_cost": float(expired_order_cost),
                "on_time_reward": float(on_time_reward),
                "revenue_gained": float(revenue_gained),
                "sequence_tw_enabled": 1.0 if bool(self.cfg.enable_sequence_time_window_reward) else 0.0,
            }
        )

        self.state = {
            "t": float(t_next),
            "i": int(i_next),
            "served": served,
            "accepted": accepted,
            "rejected": rejected,
            "expired": expired,
            "known": known,
            "loaded": loaded,
            "accept_time": accept_time,
            "finish_time": finish_time,
            "reject_reason": reject_reason,
            "soc": float(soc_next),
            "truck_pickup_load": float(truck_pickup_load),
            "pending_queue": pending_queue,
        }

        done = self._step_done(served=served, rejected=rejected, pending_queue=pending_queue)
        truck_load_next = truck_load_next_for_cost
        info = {
            "phase": "route" if wait_time <= 1e-9 else "wait",
            "decision": None,
            "decision_node": -1,
            "dt": float(dt),
            "truck_time": float(truck_time),
            "drone_time": float(drone_time),
            "wait_time": float(wait_time),
            "lateness": float(lateness),
            "time_cost": float(time_cost),
            "lateness_cost": float(lateness_cost),
            "energy_cost": float(energy_cost),
            "overtime_cost": float(overtime_cost),
            "reject_cost": 0.0,
            "auto_reject_cost": float(auto_reject_cost),
            "accept_reward": 0.0,
            "on_time_reward": float(on_time_reward),
            "late_count_cost": float(late_count_cost),
            "severe_lateness_cost": float(severe_lateness_cost),
            "expired_nodes": expired_nodes,
            "served_nodes": served_nodes,
            "service_finish_times": service_finish_times,
            "service_lateness": service_lateness,
            "revenue_gained": float(revenue_gained),
            "step_cost": float(step_cost),
            "soc_prev": float(soc),
            "soc_next": float(soc_next),
            "truck_load_prev": float(truck_load_prev),
            "truck_load_next": float(truck_load_next),
            "energy_use": float(drone_energy),
            "drone_idle_energy_use": float(drone_idle_energy),
            "truck_energy_use": float(truck_energy),
            "i": int(i),
            "j": int(j),
            "k": int(k),
            "time_bucket": time_bucket,
            "edge_mode": self.cfg.edge_mode,
            "road_distance": float(dense[i, j, 0]),
            "signal_count": float(dense[i, j, 3]),
            "turn_count": float(dense[i, j, 4]),
            "left_turn_count": float(dense[i, j, 5]),
            "u_turn_count": float(dense[i, j, 6]),
            "one_way_factor": float(dense[i, j, 7]),
            "reward_components": reward_components,
        }
        return self.get_obs(), -float(step_cost), done, info
