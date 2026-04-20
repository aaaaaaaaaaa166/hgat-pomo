from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np


REQUEST_DELIVERY = 1
REQUEST_PICKUP = -1


def _sample_release_times(
    rng: np.random.Generator,
    size: int,
    release_mode: str,
    n_batches: int,
    max_release: float,
    poisson_rate: float,
) -> np.ndarray:
    release = np.zeros((size,), dtype=np.float32)
    if size <= 0:
        return release

    if release_mode == "uniform":
        release[:] = rng.uniform(0.0, max_release, size=(size,)).astype(np.float32)
    elif release_mode == "batches":
        batch_times = np.linspace(0.0, max_release, num=max(2, n_batches)).astype(np.float32)
        batch_ids = rng.integers(0, len(batch_times), size=(size,))
        release[:] = batch_times[batch_ids]
    elif release_mode == "poisson":
        lam = max(1e-6, float(poisson_rate))
        inter = rng.exponential(scale=1.0 / lam, size=(size,)).astype(np.float32)
        rel = np.cumsum(inter, dtype=np.float32)
        rel_max = float(rel.max()) if size > 0 else 1.0
        if rel_max > 1e-6:
            rel = rel / rel_max * float(max_release)
        release[:] = rel
    else:
        raise ValueError("release_mode must be 'batches', 'uniform', or 'poisson'")
    return release


def make_instance_from_coord_demand(
    coord: np.ndarray,
    demand: np.ndarray,
    seed: int = 0,
    release_mode: str = "batches",
    n_batches: int = 4,
    max_release: float = 10.0,
    poisson_rate: float = 1.0,
    tw_mode: str = "relative",
    tw_slack_low: float = 4.0,
    tw_slack_high: float = 14.0,
    tw_active_prob: float = 0.8,
    return_due: bool = False,
    scheduled_ratio: float = 0.5,
    dynamic_pickup_ratio: float = 1.0,
    revenue_low: float = 4.0,
    revenue_high: float = 10.0,
    dynamic_revenue_bonus: float = 1.2,
    drone_eligible_prob: float = 0.75,
    response_slack_low: float = 0.25,
    response_slack_high: float = 1.00,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]],
]:
    coord = np.asarray(coord, dtype=np.float32)
    demand = np.asarray(demand, dtype=np.float32)
    if coord.ndim != 2 or coord.shape[1] != 2:
        raise ValueError("coord must be shape (N+1, 2)")
    if demand.ndim != 1 or demand.shape[0] != coord.shape[0]:
        raise ValueError("demand must be shape (N+1,)")
    if coord.shape[0] < 2:
        raise ValueError("coord must contain depot + at least 1 customer")

    N = int(coord.shape[0] - 1)
    demand = demand.copy()
    demand[0] = 0.0
    rng = np.random.default_rng(seed)

    release = np.zeros((N + 1,), dtype=np.float32)
    due = np.full((N + 1,), np.inf, dtype=np.float32)

    request_type = np.zeros((N + 1,), dtype=np.int8)
    is_dynamic = np.zeros((N + 1,), dtype=np.int8)
    revenue = np.zeros((N + 1,), dtype=np.float32)
    decision_deadline = np.zeros((N + 1,), dtype=np.float32)
    drone_eligible = np.zeros((N + 1,), dtype=np.int8)

    n_sched = int(round(float(np.clip(scheduled_ratio, 0.0, 1.0)) * N))
    n_sched = min(max(n_sched, 0), N)
    if N > 0 and n_sched == 0:
        n_sched = 1
    sched_ids = np.arange(1, n_sched + 1, dtype=np.int64)
    dyn_ids = np.arange(n_sched + 1, N + 1, dtype=np.int64)

    request_type[sched_ids] = REQUEST_DELIVERY
    is_dynamic[sched_ids] = 0
    release[sched_ids] = 0.0

    if dyn_ids.size > 0:
        dyn_release = _sample_release_times(
            rng=rng,
            size=int(dyn_ids.size),
            release_mode=release_mode,
            n_batches=n_batches,
            max_release=max_release,
            poisson_rate=poisson_rate,
        )
        dyn_release = np.maximum(dyn_release, 1e-3).astype(np.float32)
        release[dyn_ids] = dyn_release

        pickup_ratio = float(np.clip(dynamic_pickup_ratio, 0.0, 1.0))
        is_pickup = rng.uniform(0.0, 1.0, size=(dyn_ids.size,)) < pickup_ratio
        request_type[dyn_ids] = np.where(is_pickup, REQUEST_PICKUP, REQUEST_DELIVERY).astype(np.int8)
        is_dynamic[dyn_ids] = 1

        response_slack = rng.uniform(
            low=response_slack_low,
            high=max(response_slack_low + 1e-6, response_slack_high),
            size=(dyn_ids.size,),
        ).astype(np.float32)
        decision_deadline[dyn_ids] = dyn_release + response_slack

    if tw_mode == "none":
        pass
    elif tw_mode == "relative":
        if N > 0:
            slack = rng.uniform(tw_slack_low, tw_slack_high, size=(N,)).astype(np.float32)
            due[1:] = release[1:] + slack
    elif tw_mode == "mixed":
        if N > 0:
            slack = rng.uniform(tw_slack_low, tw_slack_high, size=(N,)).astype(np.float32)
            active = (rng.uniform(0.0, 1.0, size=(N,)) < float(tw_active_prob)).astype(np.float32)
            due_values = release[1:] + slack
            due[1:] = np.where(active > 0.5, due_values, np.inf).astype(np.float32)
    else:
        raise ValueError("tw_mode must be 'relative', 'mixed', or 'none'")

    if N > 0:
        revenue_scale = rng.uniform(revenue_low, revenue_high, size=(N,)).astype(np.float32)
        base_revenue = revenue_scale * np.maximum(demand[1:], 1e-3)
        type_bonus = np.where(request_type[1:] == REQUEST_PICKUP, dynamic_revenue_bonus, 1.0).astype(np.float32)
        revenue[1:] = base_revenue * type_bonus

        eligible = (rng.uniform(0.0, 1.0, size=(N,)) < float(np.clip(drone_eligible_prob, 0.0, 1.0)))
        drone_eligible[1:] = eligible.astype(np.int8)
        drone_eligible[sched_ids] = 1

    meta = {
        "request_type": request_type,
        "is_dynamic": is_dynamic,
        "revenue": revenue,
        "decision_deadline": decision_deadline,
        "drone_eligible": drone_eligible,
    }

    if return_due:
        return coord, release, demand, due, meta
    return coord, release, demand


def make_random_instance(
    N: int,
    seed: int = 0,
    coord_scale: float = 10.0,
    release_mode: str = "batches",
    n_batches: int = 4,
    max_release: float = 10.0,
    poisson_rate: float = 1.0,
    demand_low: float = 0.05,
    demand_high: float = 0.25,
    tw_mode: str = "relative",
    tw_slack_low: float = 4.0,
    tw_slack_high: float = 14.0,
    tw_active_prob: float = 0.8,
    return_due: bool = False,
    depot_coord: Tuple[float, float] = (0.0, 0.0),
    scheduled_ratio: float = 0.5,
    dynamic_pickup_ratio: float = 1.0,
    revenue_low: float = 4.0,
    revenue_high: float = 10.0,
    dynamic_revenue_bonus: float = 1.2,
    drone_eligible_prob: float = 0.75,
    response_slack_low: float = 0.25,
    response_slack_high: float = 1.00,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]],
]:
    """
    Generate one truck-drone instance with:
      - scheduled deliveries known at t=0,
      - dynamic requests released later,
      - request-level metadata for acceptance and service flow.

    Metadata keys:
      - request_type:  1 delivery, -1 pickup, 0 depot
      - is_dynamic:    1 dynamic request, 0 scheduled request
      - revenue:       service revenue per completed request
      - decision_deadline: response deadline for dynamic requests
      - drone_eligible: whether the request may be served by drone
    """

    rng = np.random.default_rng(seed)

    coord = rng.uniform(0.0, coord_scale, size=(N + 1, 2)).astype(np.float32)
    coord[0] = np.array(depot_coord, dtype=np.float32)

    demand = np.zeros((N + 1,), dtype=np.float32)
    if N > 0:
        demand[1:] = rng.uniform(demand_low, demand_high, size=(N,)).astype(np.float32)

    return make_instance_from_coord_demand(
        coord=coord,
        demand=demand,
        seed=seed,
        release_mode=release_mode,
        n_batches=n_batches,
        max_release=max_release,
        poisson_rate=poisson_rate,
        tw_mode=tw_mode,
        tw_slack_low=tw_slack_low,
        tw_slack_high=tw_slack_high,
        tw_active_prob=tw_active_prob,
        return_due=return_due,
        scheduled_ratio=scheduled_ratio,
        dynamic_pickup_ratio=dynamic_pickup_ratio,
        revenue_low=revenue_low,
        revenue_high=revenue_high,
        dynamic_revenue_bonus=dynamic_revenue_bonus,
        drone_eligible_prob=drone_eligible_prob,
        response_slack_low=response_slack_low,
        response_slack_high=response_slack_high,
    )
