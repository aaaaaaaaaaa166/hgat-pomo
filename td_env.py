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
        cfg: Optional[EnvConfig] = None,
        seed: int = 0,
    ):
        self.coord = np.asarray(coord, dtype=np.float32)
        self.release = np.asarray(release, dtype=np.float32)
        self.demand = np.asarray(demand, dtype=np.float32)
        assert self.coord.ndim == 2 and self.coord.shape[1] == 2
        assert self.release.shape[0] == self.coord.shape[0]
        assert self.demand.shape[0] == self.coord.shape[0]

        self.N = self.coord.shape[0] - 1
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(seed)

        # ===== 静态缓存：距离矩阵 =====
        # dist_mat[a,b] = euclidean distance
        diff = self.coord[:, None, :] - self.coord[None, :, :]  # (M,M,2)
        self.dist_mat = np.sqrt((diff * diff).sum(axis=-1) + 1e-12).astype(np.float32)  # (M,M)

        # ===== 静态缓存：常量 edge_index（truck/drone 与 order 的星型边）=====
        M = self.N + 1
        o_ids = np.arange(M, dtype=np.int64)
        t_ids = np.zeros(M, dtype=np.int64)
        d_ids = np.zeros(M, dtype=np.int64)
        self.edge_index_t2o = np.stack([t_ids, o_ids], axis=0)  # (2,M)
        self.edge_index_o2t = np.stack([o_ids, t_ids], axis=0)
        self.edge_index_d2o = np.stack([d_ids, o_ids], axis=0)
        self.edge_index_o2d = np.stack([o_ids, d_ids], axis=0)

        # ===== 静态缓存：不同 k 的 o2o kNN 边 =====
        self._o2o_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self.state: Dict[str, Any] = {}
        self.reset()

    def copy(self) -> "TruckDroneRendezvousEnv":
        # share static instance data + caches, but independent state
        new_env = TruckDroneRendezvousEnv(
            coord=self.coord,
            release=self.release,
            demand=self.demand,
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
        k = min(k_nn, M - 1)  # 最多只能连到除自己外的 M-1 个点
        if k <= 0:
            raise ValueError("No valid neighbors (M too small)")

        # dist[a,b]
        dist = self.dist_mat.copy()
        np.fill_diagonal(dist, 1e9)

        # 先用 argpartition 拿到每行的 k 个近邻（无序）
        nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]  # (M,k)

        # 再按距离对这 k 个近邻做行内排序（关键：用 take_along_axis）
        row = np.arange(M)[:, None]
        order = np.argsort(dist[row, nn], axis=1)  # (M,k)
        nn = np.take_along_axis(nn, order, axis=1)  # (M,k) 保持形状不变

        # 展平得到边
        src = np.repeat(np.arange(M, dtype=np.int64), k)  # (M*k,)
        dst = nn.reshape(-1).astype(np.int64)  # (M*k,)

        edge_index = np.stack([src, dst], axis=0)  # (2,E)

        d_ab = self.dist_mat[src, dst].astype(np.float32)
        edge_attr = np.stack([
            d_ab,
            d_ab / float(self.cfg.vT),
            d_ab / float(self.cfg.vD),
        ], axis=1).astype(np.float32)  # (E,3)

        self._o2o_cache[k_nn] = (edge_index, edge_attr)
        return edge_index, edge_attr

    # ---------- time helpers ----------
    def _tau_truck(self, i: int, j: int) -> float:
        return 0.0 if i == j else float(self.dist_mat[i, j]) / float(self.cfg.vT)

    def _tau_drone(self, i: int, k: int, j: int) -> float:
        return (float(self.dist_mat[i, k]) + float(self.dist_mat[k, j])) / float(self.cfg.vD)

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

    # ---------- api ----------
    def reset(self) -> Dict[str, Any]:
        served = np.zeros((self.N + 1,), dtype=np.int8)
        self.state = {"t": 0.0, "i": 0, "served": served}
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        return {
            "t": float(self.state["t"]),
            "i": int(self.state["i"]),
            "served": self.state["served"].copy(),
        }

    def get_masks(self, j: Optional[int] = None) -> Dict[str, Any]:
        t = float(self.state["t"])
        i = int(self.state["i"])
        served = self.state["served"]

        truck_mask = np.zeros((self.N + 1,), dtype=np.int8)

        feasible_orders = []
        for node in range(1, self.N + 1):
            if served[node] == 0 and self._is_released(node, t):
                feasible_orders.append(node)

        if len(feasible_orders) > 0:
            # 有可服务订单：必须推进（不允许等待、不允许去 depot）
            for node in feasible_orders:
                truck_mask[node] = 1
        else:
            # 没有可服务订单：允许等待 or 去 depot 触发“跳到下一批 release”
            truck_mask[0] = 1
            if self.cfg.allow_wait:
                truck_mask[i] = 1

        if j is None:
            return {"truck_mask": truck_mask}

        # ---- drone mask（保持你原来的）----
        if not (0 <= j <= self.N):
            raise ValueError("j out of range")

        drone_mask = np.zeros((self.N + 1,), dtype=np.int8)
        for k in range(1, self.N + 1):
            if served[k] == 1:
                continue
            if not self._is_released(k, t):
                continue
            if k == j:
                continue
            if float(self.demand[k]) > float(self.cfg.QD):
                continue
            if self._tau_drone(i, k, j) + float(self.cfg.sD) > float(self.cfg.B):
                continue
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

        truck_mask = self.get_masks()["truck_mask"]
        if truck_mask[j] == 0:
            raise ValueError(f"Infeasible j={j} at t={t}, i={i}")

        if k != self.K_NONE:
            if not (1 <= k <= self.N):
                raise ValueError("k out of range")
            dmask = self.get_masks(j=j)["drone_mask"]
            if dmask[k] == 0:
                raise ValueError(f"Infeasible k={k} for (i={i}, j={j}, t={t})")

        travel_T = self._tau_truck(i, j)
        service_T = float(self.cfg.sT) if (j != 0 and served[j] == 0 and self._is_released(j, t)) else 0.0
        truck_time = travel_T + service_T

        if k == self.K_NONE:
            drone_time = 0.0
        else:
            drone_time = self._tau_drone(i, k, j) + float(self.cfg.sD)

        dt = max(truck_time, drone_time)

        if dt == 0.0 and self.cfg.idle_to_next_release:
            nr = self._next_release_time(t, served)
            if nr is not None:
                dt = max(0.0, nr - t)

        t_next = t + dt
        i_next = j

        if j != 0 and served[j] == 0 and self._is_released(j, t):
            served[j] = 1
        if k != self.K_NONE and served[k] == 0 and self._is_released(k, t):
            served[k] = 1

        reward = -float(dt)
        done = bool(served[1:].sum() == self.N)

        self.state = {"t": t_next, "i": i_next, "served": served}

        info = {
            "dt": float(dt),
            "truck_time": float(truck_time),
            "drone_time": float(drone_time),
            "i": int(i), "j": int(j), "k": int(k),
        }
        return self.get_obs(), reward, done, info
