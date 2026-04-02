from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.graph.road_aware_features import (
    IntersectionPenaltyConfig,
    RoadAwareMatrices,
    TimeBucketConfig,
    build_proxy_road_aware_matrices,
)


@dataclass
class EnvConfig:
    vT: float = 1.0
    vD: float = 1.5
    QD: float = 1.0
    B: float = 6.0
    sT: float = 0.0
    sD: float = 0.0
    allow_wait: bool = True
    idle_to_next_release: bool = True
    traffic_sigma: float = 0.0
    lateness_penalty: float = 0.0
    soc_init: float = 1.0
    soc_min_reserve: float = 0.1
    energy_per_dist: float = 0.08
    recharge_rate: float = 0.25
    edge_mode: str = "static"  # static | road
    time_dependent: bool = False
    peak_after_served_ratio: float = 0.5
    road_detour_factor: float = 1.18
    road_signal_density: float = 0.006
    road_turn_density: float = 0.010
    road_one_way_ratio: float = 0.10
    road_peak_factor: float = 1.25
    signal_penalty: float = 0.05
    turn_penalty: float = 0.12
    left_turn_penalty: float = 0.08
    u_turn_penalty: float = 0.30


class TruckDroneRendezvousEnv:
    """
    1 Truck + 1 Drone, moving rendezvous:
      action = (k, j)
      - i is current stop (state['i'])
      - drone: i -> k -> j (recover at j)
      - truck: i -> j
    Nodes:
      0 = depot, 1..N = orders
    Dynamic orders: order c is available if t >= release[c]
    """

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
    ):
        self.coord = np.asarray(coord, dtype=np.float32)
        self.release = np.asarray(release, dtype=np.float32)
        self.demand = np.asarray(demand, dtype=np.float32)
        if due is None:
            self.due = np.full((self.coord.shape[0],), np.inf, dtype=np.float32)
        else:
            self.due = np.asarray(due, dtype=np.float32)
        assert self.coord.ndim == 2 and self.coord.shape[1] == 2
        assert self.release.shape[0] == self.coord.shape[0]
        assert self.demand.shape[0] == self.coord.shape[0]
        assert self.due.shape[0] == self.coord.shape[0]

        self.N = self.coord.shape[0] - 1
        self.cfg = cfg or EnvConfig()
        if self.cfg.edge_mode not in {"static", "road"}:
            raise ValueError("edge_mode must be 'static' or 'road'")
        self.rng = np.random.default_rng(seed)

        diff = self.coord[:, None, :] - self.coord[None, :, :]
        self.dist_mat = np.sqrt((diff * diff).sum(axis=-1) + 1e-12).astype(np.float32)
        self.road_matrices = self._init_road_matrices(road_matrices)
        self._dense_edge_attr: Optional[np.ndarray] = None

        M = self.N + 1
        o_ids = np.arange(M, dtype=np.int64)
        t_ids = np.zeros(M, dtype=np.int64)
        d_ids = np.zeros(M, dtype=np.int64)
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
            "soc": float(self.state["soc"]),
        }
        return new_env

    def get_time_bucket(self, served: Optional[np.ndarray] = None) -> str:
        if self.cfg.edge_mode != "road" or not self.cfg.time_dependent or self.N <= 0:
            return "offpeak"
        served_arr = self.state["served"] if served is None else served
        served_ratio = float(served_arr[1:].sum()) / float(max(1, self.N))
        return "peak" if served_ratio >= float(self.cfg.peak_after_served_ratio) else "offpeak"

    def get_is_peak(self, served: Optional[np.ndarray] = None) -> float:
        return 1.0 if self.get_time_bucket(served=served) == "peak" else 0.0

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
        """
        return (edge_index (2,E), edge_attr (E,8)) in numpy
        """
        k_nn = int(k_nn)
        if k_nn <= 0:
            raise ValueError("k_nn must be >= 1")

        if k_nn in self._o2o_cache:
            return self._o2o_cache[k_nn]

        M = self.N + 1
        k = min(k_nn, M - 1)
        if k <= 0:
            raise ValueError("No valid neighbors (M too small)")

        dist = self.dist_mat.copy()
        np.fill_diagonal(dist, 1e9)

        nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        row = np.arange(M)[:, None]
        order = np.argsort(dist[row, nn], axis=1)
        nn = np.take_along_axis(nn, order, axis=1)

        src = np.repeat(np.arange(M, dtype=np.int64), k)
        dst = nn.reshape(-1).astype(np.int64)
        edge_index = np.stack([src, dst], axis=0)
        dense = self.get_dense_edge_attr()
        edge_attr = dense[src, dst].astype(np.float32)

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

    def _drone_energy(self, i: int, k: int, j: int) -> float:
        dist = float(self.dist_mat[i, k]) + float(self.dist_mat[k, j])
        return dist * float(self.cfg.energy_per_dist)

    def _is_released(self, node: int, t: float) -> bool:
        return True if node == 0 else (t >= float(self.release[node]))

    def _next_release_time(self, t: float, served: np.ndarray) -> Optional[float]:
        unserved = np.where(served[1:] == 0)[0] + 1
        if unserved.size == 0:
            return None
        fut = self.release[unserved]
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

    def _drone_feasible(self, i: int, j: int, k: int, t: float, served: np.ndarray, soc: float) -> bool:
        if served[k] == 1:
            return False
        if not self._is_released(k, t):
            return False
        if k == j:
            return False
        if float(self.demand[k]) > float(self.cfg.QD):
            return False
        if self._tau_drone(i, k, j) + float(self.cfg.sD) > float(self.cfg.B):
            return False
        need = self._drone_energy(i, k, j)
        if need > max(0.0, soc - float(self.cfg.soc_min_reserve)):
            return False
        return True

    def reset(self) -> Dict[str, Any]:
        served = np.zeros((self.N + 1,), dtype=np.int8)
        soc0 = float(np.clip(float(self.cfg.soc_init), 0.0, 1.0))
        self.state = {"t": 0.0, "i": 0, "served": served, "soc": soc0}
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        return {
            "t": float(self.state["t"]),
            "i": int(self.state["i"]),
            "served": self.state["served"].copy(),
            "soc": float(self.state["soc"]),
            "time_bucket": self.get_time_bucket(),
            "is_peak": self.get_is_peak(),
        }

    def get_masks(self, j: Optional[int] = None) -> Dict[str, Any]:
        t = float(self.state["t"])
        i = int(self.state["i"])
        served = self.state["served"]
        soc = float(self.state["soc"])

        truck_mask = np.zeros((self.N + 1,), dtype=np.int8)

        feasible_orders = []
        for node in range(1, self.N + 1):
            if served[node] == 0 and self._is_released(node, t):
                feasible_orders.append(node)

        if len(feasible_orders) > 0:
            for node in feasible_orders:
                truck_mask[node] = 1
        else:
            truck_mask[0] = 1
            if self.cfg.allow_wait:
                truck_mask[i] = 1

        if j is None:
            return {"truck_mask": truck_mask}

        if not (0 <= j <= self.N):
            raise ValueError("j out of range")

        drone_mask = np.zeros((self.N + 1,), dtype=np.int8)
        for k in range(1, self.N + 1):
            if self._drone_feasible(i=i, j=j, k=k, t=t, served=served, soc=soc):
                drone_mask[k] = 1

        return {"truck_mask": truck_mask, "drone_mask": drone_mask, "k_none_feasible": True}

    def step(self, action: Tuple[int, int]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        k, j = action
        t = float(self.state["t"])
        i = int(self.state["i"])
        served = self.state["served"].copy()
        soc = float(self.state["soc"])
        time_bucket = self.get_time_bucket(served=served)

        truck_mask = self.get_masks()["truck_mask"]
        if truck_mask[j] == 0:
            raise ValueError(f"Infeasible j={j} at t={t}, i={i}")

        if k != self.K_NONE:
            if not (1 <= k <= self.N):
                raise ValueError("k out of range")
            if not self._drone_feasible(i=i, j=j, k=k, t=t, served=served, soc=soc):
                raise ValueError(f"Infeasible k={k} for (i={i}, j={j}, t={t})")

        travel_T = self._tau_truck(i, j, apply_traffic=True, bucket=time_bucket)
        service_T = float(self.cfg.sT) if (j != 0 and served[j] == 0 and self._is_released(j, t)) else 0.0
        truck_time = travel_T + service_T

        if k == self.K_NONE:
            drone_time = 0.0
            energy_use = 0.0
        else:
            drone_time = self._tau_drone(i, k, j) + float(self.cfg.sD)
            energy_use = self._drone_energy(i, k, j)

        dt = max(truck_time, drone_time)
        if dt == 0.0 and self.cfg.idle_to_next_release:
            nr = self._next_release_time(t, served)
            if nr is not None:
                dt = max(0.0, nr - t)

        t_next = t + dt
        i_next = j

        finish_j = t + truck_time
        finish_k = t + drone_time
        lateness = 0.0

        if j != 0 and served[j] == 0 and self._is_released(j, t):
            served[j] = 1
            lateness += self._lateness(j, finish_j)
        if k != self.K_NONE and served[k] == 0 and self._is_released(k, t):
            served[k] = 1
            lateness += self._lateness(k, finish_k)

        if k == self.K_NONE:
            soc_after = soc
            recharge_time = dt
        else:
            soc_after = max(0.0, soc - energy_use)
            recharge_time = max(0.0, dt - drone_time)
        soc_next = min(1.0, soc_after + float(self.cfg.recharge_rate) * recharge_time)

        penalty = float(self.cfg.lateness_penalty) * float(lateness)
        reward = -float(dt) - penalty
        done = bool(served[1:].sum() == self.N)

        self.state = {"t": t_next, "i": i_next, "served": served, "soc": soc_next}

        dense = self.get_dense_edge_attr()
        info = {
            "dt": float(dt),
            "truck_time": float(truck_time),
            "drone_time": float(drone_time),
            "lateness": float(lateness),
            "penalty": float(penalty),
            "soc_prev": float(soc),
            "soc_next": float(soc_next),
            "energy_use": float(energy_use),
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
        }
        return self.get_obs(), reward, done, info
