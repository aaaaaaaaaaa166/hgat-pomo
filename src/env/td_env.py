from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import copy
import numpy as np


@dataclass
class EnvConfig:
    vT: float = 1.0     # truck speed
    vD: float = 1.5     # drone speed
    QD: float = 1.0     # drone payload capacity
    B: float = 6.0      # max drone flight time for (i->k->j) + sD
    sT: float = 0.0     # truck service time at order node
    sD: float = 0.0     # drone service time at order node
    allow_wait: bool = True
    idle_to_next_release: bool = True
    traffic_sigma: float = 0.0        # multiplicative noise std on truck travel time
    lateness_penalty: float = 0.0     # cost per unit tardiness
    soc_init: float = 1.0             # initial drone state-of-charge in [0,1]
    soc_min_reserve: float = 0.1      # reserve SoC that must remain after sortie
    energy_per_dist: float = 0.08     # SoC cost per drone flight distance
    recharge_rate: float = 0.25       # SoC recovered per unit waiting time on truck


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

    K_NONE = -1  # no-drone action

    def __init__(
        self,
        coord: np.ndarray,    # (N+1,2)
        release: np.ndarray,  # (N+1,)
        demand: np.ndarray,   # (N+1,)
        due: Optional[np.ndarray] = None,  # (N+1,), np.inf means no deadline
        cfg: Optional[EnvConfig] = None,
        seed: int = 0,
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
        self.rng = np.random.default_rng(seed)

        # dist_mat[a,b] = euclidean distance
        diff = self.coord[:, None, :] - self.coord[None, :, :]
        self.dist_mat = np.sqrt((diff * diff).sum(axis=-1) + 1e-12).astype(np.float32)

        # cached star edges truck/drone <-> orders
        M = self.N + 1
        o_ids = np.arange(M, dtype=np.int64)
        t_ids = np.zeros(M, dtype=np.int64)
        d_ids = np.zeros(M, dtype=np.int64)
        self.edge_index_t2o = np.stack([t_ids, o_ids], axis=0)
        self.edge_index_o2t = np.stack([o_ids, t_ids], axis=0)
        self.edge_index_d2o = np.stack([d_ids, o_ids], axis=0)
        self.edge_index_o2d = np.stack([o_ids, d_ids], axis=0)

        # cached o2o kNN edges
        self._o2o_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self.state: Dict[str, Any] = {}
        self.reset()

    def copy(self) -> "TruckDroneRendezvousEnv":
        # share static instance data + caches, but independent state
        new_env = TruckDroneRendezvousEnv(
            coord=self.coord,
            release=self.release,
            demand=self.demand,
            due=self.due,
            cfg=copy.deepcopy(self.cfg),
            seed=int(self.rng.integers(0, 10**9)),
        )
        new_env.dist_mat = self.dist_mat
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

    # ---------- cached o2o edges ----------
    def get_o2o_edges(self, k_nn: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        return (edge_index (2,E), edge_attr (E,3)) in numpy (float32 for attr)
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

        d_ab = self.dist_mat[src, dst].astype(np.float32)
        edge_attr = np.stack(
            [
                d_ab,
                d_ab / float(self.cfg.vT),
                d_ab / float(self.cfg.vD),
            ],
            axis=1,
        ).astype(np.float32)

        self._o2o_cache[k_nn] = (edge_index, edge_attr)
        return edge_index, edge_attr

    # ---------- helpers ----------
    def _traffic_factor(self, i: int, j: int) -> float:
        if i == j or float(self.cfg.traffic_sigma) <= 0.0:
            return 1.0
        factor = self.rng.normal(loc=1.0, scale=float(self.cfg.traffic_sigma))
        return float(np.clip(factor, 0.5, 2.0))

    def _tau_truck(self, i: int, j: int, apply_traffic: bool = False) -> float:
        if i == j:
            return 0.0
        base = float(self.dist_mat[i, j]) / float(self.cfg.vT)
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
        """判断当前状态下 (k, j) 的无人机派送是否可行。

        约束包括：
        - k 必须是“未服务且已释放”的订单
        - 载重与航程时长约束
        - 执行后 SoC 仍需高于安全余量
        - 本实现不允许 k 与 j 是同一订单
        """
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

    # ---------- api ----------
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
        }

    def get_masks(self, j: Optional[int] = None) -> Dict[str, Any]:
        """返回动作 mask。

        - 不传 `j`：仅返回卡车可选目的地；
        - 传入 `j`：额外返回该 j 下无人机可选 k。
        """
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
            # 没有可服务订单时：允许回仓库，且可原地等待（若配置允许）。
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
        """
        action = (k, j)
          k in [1..N] or K_NONE(-1)
          j in [0..N] (order/depot only)
        """
        k, j = action
        t = float(self.state["t"])
        i = int(self.state["i"])
        served = self.state["served"].copy()
        soc = float(self.state["soc"])

        truck_mask = self.get_masks()["truck_mask"]
        if truck_mask[j] == 0:
            raise ValueError(f"Infeasible j={j} at t={t}, i={i}")

        if k != self.K_NONE:
            if not (1 <= k <= self.N):
                raise ValueError("k out of range")
            if not self._drone_feasible(i=i, j=j, k=k, t=t, served=served, soc=soc):
                raise ValueError(f"Infeasible k={k} for (i={i}, j={j}, t={t})")

        travel_T = self._tau_truck(i, j, apply_traffic=True)
        service_T = float(self.cfg.sT) if (j != 0 and served[j] == 0 and self._is_released(j, t)) else 0.0
        truck_time = travel_T + service_T

        if k == self.K_NONE:
            drone_time = 0.0
            energy_use = 0.0
        else:
            drone_time = self._tau_drone(i, k, j) + float(self.cfg.sD)
            energy_use = self._drone_energy(i, k, j)

        # 卡车与无人机并行执行，单步耗时由较慢者决定。
        dt = max(truck_time, drone_time)
        if dt == 0.0 and self.cfg.idle_to_next_release:
            # 事件驱动快进到下一次订单释放，避免零时长循环。
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
            # 无人机仅在会合后等待卡车的时间里充电。
            soc_after = max(0.0, soc - energy_use)
            recharge_time = max(0.0, dt - drone_time)
        soc_next = min(1.0, soc_after + float(self.cfg.recharge_rate) * recharge_time)

        # 奖励定义：时间成本取负，并叠加迟到惩罚。
        penalty = float(self.cfg.lateness_penalty) * float(lateness)
        reward = -float(dt) - penalty
        done = bool(served[1:].sum() == self.N)

        self.state = {"t": t_next, "i": i_next, "served": served, "soc": soc_next}

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
        }
        return self.get_obs(), reward, done, info
