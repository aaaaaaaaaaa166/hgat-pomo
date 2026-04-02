from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def _route_cost(
    env,
    start_i: int,
    start_t: float,
    route: Sequence[int],
    cost_fn: Callable[[int, int], float],
    service_time: float,
    lateness_penalty: float,
) -> float:
    total = 0.0
    cur_i = int(start_i)
    cur_t = float(start_t)
    for node in route:
        travel = float(cost_fn(cur_i, int(node)))
        arrive = cur_t + travel + service_time
        due_t = float(env.due[int(node)])
        late = 0.0 if not np.isfinite(due_t) else max(0.0, arrive - due_t)
        total += travel + service_time + lateness_penalty * late
        cur_i = int(node)
        cur_t = arrive
    return float(total)


def _nearest_edd_seed(
    env,
    start_i: int,
    start_t: float,
    candidates: Sequence[int],
    cost_fn: Callable[[int, int], float],
) -> List[int]:
    remaining = list(int(x) for x in candidates)
    route: List[int] = []
    cur_i = int(start_i)
    cur_t = float(start_t)
    service_time = float(env.cfg.sT)
    while remaining:
        def score(node: int) -> Tuple[float, float, float]:
            travel = float(cost_fn(cur_i, node))
            due_t = float(env.due[node])
            slack = 1e9 if not np.isfinite(due_t) else max(-1e9, due_t - (cur_t + travel + service_time))
            return (slack, travel, due_t if np.isfinite(due_t) else 1e9)

        nxt = min(remaining, key=score)
        route.append(nxt)
        remaining.remove(nxt)
        cur_t += float(cost_fn(cur_i, nxt)) + service_time
        cur_i = nxt
    return route


def _two_opt(
    env,
    start_i: int,
    start_t: float,
    route: Sequence[int],
    cost_fn: Callable[[int, int], float],
) -> List[int]:
    best = list(route)
    if len(best) < 4:
        return best

    best_cost = _route_cost(
        env=env,
        start_i=start_i,
        start_t=start_t,
        route=best,
        cost_fn=cost_fn,
        service_time=float(env.cfg.sT),
        lateness_penalty=float(env.cfg.lateness_penalty),
    )

    improved = True
    max_passes = 3
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for a in range(0, len(best) - 2):
            for b in range(a + 2, len(best)):
                cand = best[:a] + best[a:b][::-1] + best[b:]
                cand_cost = _route_cost(
                    env=env,
                    start_i=start_i,
                    start_t=start_t,
                    route=cand,
                    cost_fn=cost_fn,
                    service_time=float(env.cfg.sT),
                    lateness_penalty=float(env.cfg.lateness_penalty),
                )
                if cand_cost + 1e-6 < best_cost:
                    best = cand
                    best_cost = cand_cost
                    improved = True
    return best


def choose_truck_next_local_search(env, obs) -> int:
    """
    Online truck-only baseline:
    replan over currently released orders, initialize with nearest/EDD, then 2-opt.
    """
    i = int(obs["i"])
    t = float(obs["t"])
    truck_mask = env.get_masks()["truck_mask"]
    feasible = np.where(truck_mask > 0)[0]
    orders = [int(x) for x in feasible if int(x) != 0 and env.state["served"][int(x)] == 0]
    if not orders:
        if i in feasible:
            return int(i)
        return int(feasible[0]) if len(feasible) > 0 else 0

    bucket = str(obs.get("time_bucket", env.get_time_bucket()))

    def truck_cost(a: int, b: int) -> float:
        return float(env._tau_truck(int(a), int(b), apply_traffic=False, bucket=bucket))

    seed = _nearest_edd_seed(env, i, t, orders, truck_cost)
    improved = _two_opt(env, i, t, seed, truck_cost)
    return int(improved[0]) if improved else int(seed[0])
