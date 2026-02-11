from __future__ import annotations
from typing import Tuple
import numpy as np


def make_random_instance(
    N: int,
    seed: int = 0,
    coord_scale: float = 10.0,
    release_mode: str = "batches",  # "batches" or "uniform"
    n_batches: int = 4,
    max_release: float = 10.0,
    demand_low: float = 0.1,
    demand_high: float = 1.0,
    depot_coord: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    coord = rng.uniform(0.0, coord_scale, size=(N + 1, 2)).astype(np.float32)
    coord[0] = np.array(depot_coord, dtype=np.float32)

    demand = np.zeros((N + 1,), dtype=np.float32)
    demand[1:] = rng.uniform(demand_low, demand_high, size=(N,)).astype(np.float32)

    release = np.zeros((N + 1,), dtype=np.float32)
    if release_mode == "uniform":
        release[1:] = rng.uniform(0.0, max_release, size=(N,)).astype(np.float32)
    elif release_mode == "batches":
        batch_times = np.linspace(0.0, max_release, num=max(2, n_batches)).astype(np.float32)
        batch_ids = rng.integers(0, len(batch_times), size=(N,))
        release[1:] = batch_times[batch_ids]
    else:
        raise ValueError("release_mode must be 'batches' or 'uniform'")

    return coord, release, demand
