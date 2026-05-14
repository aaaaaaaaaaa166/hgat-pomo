from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class JointBeamObjectiveConfig:
    accept_weight: float = 20.0
    on_time_weight: float = 30.0
    late_weight: float = 40.0
    lateness_weight: float = 3.0
    max_lateness_weight: float = 8.0
    energy_weight: float = 0.08
    distance_weight: float = 0.04
    hard_violation_weight: float = 1_000_000.0

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class JointBeamMetrics:
    accepted_orders: int = 0
    on_time_orders: int = 0
    late_orders: int = 0
    total_lateness: float = 0.0
    max_lateness: float = 0.0
    total_energy: float = 0.0
    total_distance: float = 0.0
    hard_constraint_violations: int = 0

    def copy(self) -> "JointBeamMetrics":
        return JointBeamMetrics(**asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["total_lateness"] = float(out["total_lateness"])
        out["max_lateness"] = float(out["max_lateness"])
        out["total_energy"] = float(out["total_energy"])
        out["total_distance"] = float(out["total_distance"])
        return out


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        value_f = float(value)
        return value_f if math.isfinite(value_f) else default
    except Exception:
        return default


def service_distance_energy(env: Any, info: Dict[str, Any]) -> Tuple[float, float]:
    """Extract comparable distance/energy from a real env transition."""
    distance = _num(info.get("road_distance"))
    energy = _num(info.get("truck_energy_use")) + _num(info.get("energy_use"))
    try:
        k = int(info.get("k", getattr(env, "K_NONE", -1)))
        i = int(info.get("i", 0))
        j = int(info.get("j", 0))
        if k != getattr(env, "K_NONE", -1):
            distance += float(env.dist_mat[i, k]) + float(env.dist_mat[k, j])
    except Exception:
        pass
    return float(distance), float(energy)


def transition_metrics(env: Any, info: Dict[str, Any]) -> JointBeamMetrics:
    out = JointBeamMetrics()
    if info.get("decision") == "accept":
        out.accepted_orders += 1
    hard = 0
    try:
        k = int(info.get("k", getattr(env, "K_NONE", -1)))
        if k != getattr(env, "K_NONE", -1):
            reserve = _num(getattr(env.cfg, "soc_min_reserve", 0.0))
            if _num(info.get("energy_use")) > max(0.0, _num(info.get("soc_prev")) - reserve) + 1e-9:
                hard += 1
            if _num(info.get("drone_time")) > _num(getattr(env.cfg, "B", 0.0)) + 1e-9:
                hard += 1
        if _num(info.get("truck_load_next")) > _num(getattr(env.cfg, "truck_capacity", 0.0)) + 1e-9:
            hard += 1
    except Exception:
        hard += 1
    out.hard_constraint_violations += int(hard)

    late_values: List[float] = []
    for value in (info.get("service_lateness", {}) or {}).values():
        late_values.append(max(0.0, _num(value)))
    out.on_time_orders += sum(1 for value in late_values if value <= 1e-9)
    out.late_orders += sum(1 for value in late_values if value > 1e-9)
    out.total_lateness += float(sum(late_values))
    out.max_lateness = max([0.0] + late_values)
    distance, energy = service_distance_energy(env, info)
    out.total_distance += float(distance)
    out.total_energy += float(energy)
    return out


def add_metrics(a: JointBeamMetrics, b: JointBeamMetrics) -> JointBeamMetrics:
    return JointBeamMetrics(
        accepted_orders=int(a.accepted_orders) + int(b.accepted_orders),
        on_time_orders=int(a.on_time_orders) + int(b.on_time_orders),
        late_orders=int(a.late_orders) + int(b.late_orders),
        total_lateness=float(a.total_lateness) + float(b.total_lateness),
        max_lateness=max(float(a.max_lateness), float(b.max_lateness)),
        total_energy=float(a.total_energy) + float(b.total_energy),
        total_distance=float(a.total_distance) + float(b.total_distance),
        hard_constraint_violations=int(a.hard_constraint_violations) + int(b.hard_constraint_violations),
    )


def merge_transition_metrics(env: Any, infos: Iterable[Dict[str, Any]]) -> JointBeamMetrics:
    merged = JointBeamMetrics()
    for info in infos:
        merged = add_metrics(merged, transition_metrics(env, info))
    return merged


def score_metrics(metrics: JointBeamMetrics, cfg: JointBeamObjectiveConfig) -> float:
    score = 0.0
    score -= float(cfg.accept_weight) * float(metrics.accepted_orders)
    score -= float(cfg.on_time_weight) * float(metrics.on_time_orders)
    score += float(cfg.late_weight) * float(metrics.late_orders)
    score += float(cfg.lateness_weight) * float(metrics.total_lateness)
    score += float(cfg.max_lateness_weight) * float(metrics.max_lateness)
    score += float(cfg.energy_weight) * float(metrics.total_energy)
    score += float(cfg.distance_weight) * float(metrics.total_distance)
    score += float(cfg.hard_violation_weight) * float(metrics.hard_constraint_violations)
    return float(score)


def pending_order_risk_score(env: Any, cfg: JointBeamObjectiveConfig) -> Tuple[float, Dict[str, float]]:
    """Proxy risk for accepted-but-unserved orders left by a partial beam.

    Without this proxy, an accept-only transition can collect acceptance reward
    while deferring all lateness cost beyond the lookahead horizon. The proxy is
    intentionally conservative and uses the same business weights as completed
    lateness terms.
    """
    state = getattr(env, "state", {}) or {}
    accepted = state.get("accepted")
    served = state.get("served")
    rejected = state.get("rejected")
    if accepted is None or served is None or rejected is None:
        return 0.0, {}
    t = _num(state.get("t"))
    i = int(state.get("i", 0))
    total_late = 0.0
    max_late = 0.0
    late_count = 0
    total_distance = 0.0
    due_pressure = 0.0
    for node in range(1, int(getattr(env, "N", 0)) + 1):
        if int(accepted[node]) <= 0 or int(served[node]) > 0 or int(rejected[node]) > 0:
            continue
        due = _num(env.due[node], float("inf"))
        if not math.isfinite(due):
            continue
        try:
            travel = float(env._tau_truck(i, node, apply_traffic=False, bucket=env.get_time_bucket(t_elapsed=t)))
        except Exception:
            travel = float(env.dist_mat[i, node]) / max(1e-9, float(getattr(env.cfg, "vT", 1.0)))
        service = float(getattr(env.cfg, "sT", 0.0))
        eta = max(t, _num(env.release[node])) + travel + service
        late = max(0.0, eta - due)
        slack = due - eta
        total_distance += float(env.dist_mat[i, node])
        if late > 1e-9:
            late_count += 1
            total_late += late
            max_late = max(max_late, late)
        elif slack <= 2.0:
            due_pressure += (2.0 - slack) / 2.0
    score = 0.0
    score += float(cfg.late_weight) * float(late_count)
    score += float(cfg.lateness_weight) * float(total_late)
    score += float(cfg.max_lateness_weight) * float(max_late)
    score += 0.25 * float(cfg.distance_weight) * float(total_distance)
    score += 0.5 * float(cfg.late_weight) * float(due_pressure)
    return float(score), {
        "pending_late_count": float(late_count),
        "pending_total_lateness": float(total_late),
        "pending_max_lateness": float(max_late),
        "pending_distance_proxy": float(total_distance),
        "pending_due_pressure": float(due_pressure),
        "pending_risk_score": float(score),
    }


def dominance_prune(states: List[Any]) -> List[Any]:
    """Remove states that are no better on every business metric."""
    kept: List[Any] = []
    for i, state in enumerate(states):
        a = state.metrics
        dominated = False
        for j, other in enumerate(states):
            if i == j:
                continue
            b = other.metrics
            no_better = (
                a.accepted_orders <= b.accepted_orders
                and a.on_time_orders <= b.on_time_orders
                and a.late_orders >= b.late_orders
                and a.max_lateness >= b.max_lateness
                and a.total_lateness >= b.total_lateness
                and a.total_distance >= b.total_distance
                and a.total_energy >= b.total_energy
                and a.hard_constraint_violations >= b.hard_constraint_violations
            )
            strictly_worse = (
                a.accepted_orders < b.accepted_orders
                or a.on_time_orders < b.on_time_orders
                or a.late_orders > b.late_orders
                or a.max_lateness > b.max_lateness
                or a.total_lateness > b.total_lateness
                or a.total_distance > b.total_distance
                or a.total_energy > b.total_energy
                or a.hard_constraint_violations > b.hard_constraint_violations
            )
            if no_better and strictly_worse:
                dominated = True
                break
        if not dominated:
            kept.append(state)
    return kept
