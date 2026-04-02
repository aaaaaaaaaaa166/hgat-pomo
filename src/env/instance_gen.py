from __future__ import annotations
from typing import Tuple, Union
import numpy as np


def make_random_instance(
    N: int,
    seed: int = 0,
    coord_scale: float = 10.0,
    release_mode: str = "batches",  # "batches" or "uniform" or "poisson"
    n_batches: int = 4,
    max_release: float = 10.0,
    poisson_rate: float = 1.0,
    demand_low: float = 0.1,
    demand_high: float = 1.0,
    tw_mode: str = "relative",  # "relative" or "none" or "mixed"
    tw_slack_low: float = 4.0,
    tw_slack_high: float = 14.0,
    tw_active_prob: float = 0.8,
    return_due: bool = False,
    depot_coord: Tuple[float, float] = (0.0, 0.0),
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """生成一个随机卡车-无人机配送实例。

    约定：
    - 0 号节点是仓库(depot)，1..N 是订单点。
    - `release_mode` 控制动态到达（订单释放时刻）。
    - `tw_mode` 控制截止时间；`inf` 表示无时间窗。
    - `return_due=False` 时保持旧接口兼容。
    """
    rng = np.random.default_rng(seed)

    coord = rng.uniform(0.0, coord_scale, size=(N + 1, 2)).astype(np.float32)
    coord[0] = np.array(depot_coord, dtype=np.float32)

    demand = np.zeros((N + 1,), dtype=np.float32)
    demand[1:] = rng.uniform(demand_low, demand_high, size=(N,)).astype(np.float32)

    release = np.zeros((N + 1,), dtype=np.float32)
    if release_mode == "uniform":
        release[1:] = rng.uniform(0.0, max_release, size=(N,)).astype(np.float32)
    elif release_mode == "batches":
        # 把订单分到若干“同步释放批次”。
        batch_times = np.linspace(0.0, max_release, num=max(2, n_batches)).astype(np.float32)
        batch_ids = rng.integers(0, len(batch_times), size=(N,))
        release[1:] = batch_times[batch_ids]
    elif release_mode == "poisson":
        # 先采样泊松到达间隔，再缩放到 [0, max_release]。
        lam = max(1e-6, float(poisson_rate))
        inter = rng.exponential(scale=1.0 / lam, size=(N,)).astype(np.float32)
        rel = np.cumsum(inter, dtype=np.float32)
        rel_max = float(rel.max()) if N > 0 else 1.0
        if rel_max > 1e-6:
            rel = rel / rel_max * float(max_release)
        release[1:] = rel
    else:
        raise ValueError("release_mode must be 'batches', 'uniform', or 'poisson'")

    due = np.full((N + 1,), np.inf, dtype=np.float32)
    due[0] = np.inf
    if tw_mode == "none":
        pass
    elif tw_mode == "relative":
        slack = rng.uniform(tw_slack_low, tw_slack_high, size=(N,)).astype(np.float32)
        due[1:] = release[1:] + slack
    elif tw_mode == "mixed":
        slack = rng.uniform(tw_slack_low, tw_slack_high, size=(N,)).astype(np.float32)
        active = (rng.uniform(0.0, 1.0, size=(N,)) < float(tw_active_prob)).astype(np.float32)
        due_values = release[1:] + slack
        due[1:] = np.where(active > 0.5, due_values, np.inf).astype(np.float32)
    else:
        raise ValueError("tw_mode must be 'relative', 'none', or 'mixed'")

    if return_due:
        return coord, release, demand, due
    return coord, release, demand
