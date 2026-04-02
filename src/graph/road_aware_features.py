from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class IntersectionPenaltyConfig:
    """Lightweight delay model for signalized last-mile routing."""

    signal_penalty_sec: float = 5.0
    turn_penalty_sec: float = 12.0
    left_turn_penalty_sec: float = 8.0
    u_turn_penalty_sec: float = 30.0


@dataclass(frozen=True)
class TimeBucketConfig:
    """Two-bucket time dependence keeps the thesis scope manageable."""

    offpeak_factor: float = 1.00
    peak_factor: float = 1.25


@dataclass
class RoadAwareMatrices:
    road_distance: np.ndarray
    travel_time_offpeak: np.ndarray
    travel_time_peak: np.ndarray
    signal_count: np.ndarray
    turn_count: np.ndarray
    left_turn_count: np.ndarray
    u_turn_count: np.ndarray
    one_way_factor: np.ndarray

    def edge_attr(self) -> np.ndarray:
        """Returns dense pairwise edge features for a stop-level complete graph."""

        return np.stack(
            [
                self.road_distance,
                self.travel_time_offpeak,
                self.travel_time_peak,
                self.signal_count,
                self.turn_count,
                self.left_turn_count,
                self.u_turn_count,
                self.one_way_factor,
            ],
            axis=-1,
        )

    def select_cost_matrix(self, bucket: str = "offpeak") -> np.ndarray:
        if bucket == "peak":
            return self.travel_time_peak
        return self.travel_time_offpeak


def _ensure_square(name: str, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape={array.shape}.")
    return array


def pairwise_euclidean_distance(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance in meters for planar or projected coordinates."""

    pts = np.asarray(coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("coords must have shape [num_nodes, 2].")

    diff = pts[:, None, :] - pts[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_intersection_delay(
    signal_count: np.ndarray,
    turn_count: np.ndarray,
    left_turn_count: Optional[np.ndarray] = None,
    u_turn_count: Optional[np.ndarray] = None,
    penalty: IntersectionPenaltyConfig = IntersectionPenaltyConfig(),
) -> np.ndarray:
    signal_count = _ensure_square("signal_count", signal_count)
    turn_count = _ensure_square("turn_count", turn_count)
    left_turn_count = (
        np.zeros_like(signal_count)
        if left_turn_count is None
        else _ensure_square("left_turn_count", left_turn_count)
    )
    u_turn_count = (
        np.zeros_like(signal_count)
        if u_turn_count is None
        else _ensure_square("u_turn_count", u_turn_count)
    )

    delay = (
        signal_count * penalty.signal_penalty_sec
        + turn_count * penalty.turn_penalty_sec
        + left_turn_count * penalty.left_turn_penalty_sec
        + u_turn_count * penalty.u_turn_penalty_sec
    )
    np.fill_diagonal(delay, 0.0)
    return delay.astype(np.float32)


def build_time_dependent_travel_time(
    base_travel_time: np.ndarray,
    signal_count: np.ndarray,
    turn_count: np.ndarray,
    left_turn_count: Optional[np.ndarray] = None,
    u_turn_count: Optional[np.ndarray] = None,
    time_bucket: TimeBucketConfig = TimeBucketConfig(),
    penalty: IntersectionPenaltyConfig = IntersectionPenaltyConfig(),
) -> Dict[str, np.ndarray]:
    """
    Creates a lightweight time-dependent edge cost.

    This follows a thesis-friendly approximation:
    1. keep stop-level routing,
    2. use road-based travel time as the base cost,
    3. add deterministic intersection delay,
    4. scale only the moving part during peak periods.
    """

    base_travel_time = _ensure_square("base_travel_time", base_travel_time)
    delay = build_intersection_delay(
        signal_count=signal_count,
        turn_count=turn_count,
        left_turn_count=left_turn_count,
        u_turn_count=u_turn_count,
        penalty=penalty,
    )

    moving_time = np.maximum(base_travel_time, 0.0)
    offpeak = moving_time * time_bucket.offpeak_factor + delay
    peak = moving_time * time_bucket.peak_factor + delay
    np.fill_diagonal(offpeak, 0.0)
    np.fill_diagonal(peak, 0.0)
    return {
        "offpeak": offpeak.astype(np.float32),
        "peak": peak.astype(np.float32),
    }


def build_road_aware_matrices(
    road_distance: np.ndarray,
    base_travel_time: np.ndarray,
    signal_count: np.ndarray,
    turn_count: np.ndarray,
    left_turn_count: Optional[np.ndarray] = None,
    u_turn_count: Optional[np.ndarray] = None,
    one_way_factor: Optional[np.ndarray] = None,
    time_bucket: TimeBucketConfig = TimeBucketConfig(),
    penalty: IntersectionPenaltyConfig = IntersectionPenaltyConfig(),
) -> RoadAwareMatrices:
    road_distance = _ensure_square("road_distance", road_distance)
    base_travel_time = _ensure_square("base_travel_time", base_travel_time)
    signal_count = _ensure_square("signal_count", signal_count)
    turn_count = _ensure_square("turn_count", turn_count)
    left_turn_count = (
        np.zeros_like(signal_count)
        if left_turn_count is None
        else _ensure_square("left_turn_count", left_turn_count)
    )
    u_turn_count = (
        np.zeros_like(signal_count)
        if u_turn_count is None
        else _ensure_square("u_turn_count", u_turn_count)
    )
    one_way_factor = (
        np.ones_like(signal_count)
        if one_way_factor is None
        else _ensure_square("one_way_factor", one_way_factor)
    )

    td = build_time_dependent_travel_time(
        base_travel_time=base_travel_time,
        signal_count=signal_count,
        turn_count=turn_count,
        left_turn_count=left_turn_count,
        u_turn_count=u_turn_count,
        time_bucket=time_bucket,
        penalty=penalty,
    )

    return RoadAwareMatrices(
        road_distance=road_distance.astype(np.float32),
        travel_time_offpeak=td["offpeak"],
        travel_time_peak=td["peak"],
        signal_count=signal_count.astype(np.float32),
        turn_count=turn_count.astype(np.float32),
        left_turn_count=left_turn_count.astype(np.float32),
        u_turn_count=u_turn_count.astype(np.float32),
        one_way_factor=one_way_factor.astype(np.float32),
    )


def build_proxy_road_aware_matrices(
    coords: np.ndarray,
    avg_speed_kmph: float = 22.0,
    road_detour_factor: float = 1.18,
    signal_density: float = 0.006,
    turn_density: float = 0.010,
    one_way_ratio: float = 0.10,
    time_bucket: TimeBucketConfig = TimeBucketConfig(),
    penalty: IntersectionPenaltyConfig = IntersectionPenaltyConfig(),
) -> RoadAwareMatrices:
    """
    Fallback builder when no routing engine is available yet.

    It keeps the thesis runnable by approximating road features from coordinates.
    Replace this with OSMnx/OSRM statistics when the real route matrices are ready.
    """

    euclid = pairwise_euclidean_distance(coords)
    road_distance = euclid * road_detour_factor

    avg_speed_mps = max(avg_speed_kmph, 1e-6) * 1000.0 / 3600.0
    base_travel_time = road_distance / avg_speed_mps
    signal_count = np.rint(road_distance * signal_density).astype(np.float32)
    turn_count = np.maximum(1.0, np.rint(road_distance * turn_density)).astype(np.float32)
    np.fill_diagonal(signal_count, 0.0)
    np.fill_diagonal(turn_count, 0.0)

    left_turn_count = np.floor(turn_count * 0.35).astype(np.float32)
    u_turn_count = np.zeros_like(turn_count, dtype=np.float32)
    one_way_factor = np.ones_like(turn_count, dtype=np.float32)
    mask = road_distance > 0.0
    one_way_factor[mask] = 1.0 + one_way_ratio
    np.fill_diagonal(one_way_factor, 1.0)

    return build_road_aware_matrices(
        road_distance=road_distance,
        base_travel_time=base_travel_time,
        signal_count=signal_count,
        turn_count=turn_count,
        left_turn_count=left_turn_count,
        u_turn_count=u_turn_count,
        one_way_factor=one_way_factor,
        time_bucket=time_bucket,
        penalty=penalty,
    )


def edge_attr_to_dict(edge_attr: np.ndarray) -> Mapping[str, np.ndarray]:
    edge_attr = np.asarray(edge_attr, dtype=np.float32)
    if edge_attr.ndim != 3 or edge_attr.shape[-1] != 8:
        raise ValueError("edge_attr must have shape [N, N, 8].")
    return {
        "road_distance": edge_attr[..., 0],
        "travel_time_offpeak": edge_attr[..., 1],
        "travel_time_peak": edge_attr[..., 2],
        "signal_count": edge_attr[..., 3],
        "turn_count": edge_attr[..., 4],
        "left_turn_count": edge_attr[..., 5],
        "u_turn_count": edge_attr[..., 6],
        "one_way_factor": edge_attr[..., 7],
    }


def route_cost(
    route: Iterable[int],
    cost_matrix: np.ndarray,
) -> float:
    route = list(route)
    if len(route) < 2:
        return 0.0

    cost_matrix = _ensure_square("cost_matrix", cost_matrix)
    total = 0.0
    for src, dst in zip(route[:-1], route[1:]):
        total += float(cost_matrix[src, dst])
    return total
