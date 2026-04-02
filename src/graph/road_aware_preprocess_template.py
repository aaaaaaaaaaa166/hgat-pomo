from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph.road_aware_features import (
    IntersectionPenaltyConfig,
    RoadAwareMatrices,
    TimeBucketConfig,
    build_road_aware_matrices,
)


@dataclass(frozen=True)
class StopPoint:
    lat: float
    lon: float


def download_drive_graph(place_name: str) -> nx.MultiDiGraph:
    """
    Open-source reference stack:
    - OSMnx for map extraction
    - NetworkX shortest path for path-level statistics

    This is intentionally lightweight and thesis-friendly.
    """

    graph = ox.graph_from_place(place_name, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def _node_has_signal(graph: nx.MultiDiGraph, node_id: int) -> bool:
    data = graph.nodes[node_id]
    highway = data.get("highway")
    if isinstance(highway, list):
        return "traffic_signals" in highway
    return highway == "traffic_signals"


def _best_path(graph: nx.MultiDiGraph, src: int, dst: int) -> List[int]:
    return nx.shortest_path(graph, src, dst, weight="travel_time")


def _path_length(graph: nx.MultiDiGraph, path: Sequence[int]) -> float:
    return float(nx.path_weight(graph, path, weight="length"))


def _path_travel_time(graph: nx.MultiDiGraph, path: Sequence[int]) -> float:
    return float(nx.path_weight(graph, path, weight="travel_time"))


def _path_signal_count(graph: nx.MultiDiGraph, path: Sequence[int]) -> int:
    if len(path) <= 2:
        return 0
    return sum(1 for node_id in path[1:-1] if _node_has_signal(graph, node_id))


def _path_turn_count(path: Sequence[int]) -> int:
    return max(len(path) - 2, 0)


def build_pairwise_road_matrices(
    graph: nx.MultiDiGraph,
    stops: Iterable[StopPoint],
    time_bucket: TimeBucketConfig = TimeBucketConfig(),
    penalty: IntersectionPenaltyConfig = IntersectionPenaltyConfig(),
) -> RoadAwareMatrices:
    stops = list(stops)
    node_ids = [ox.distance.nearest_nodes(graph, stop.lon, stop.lat) for stop in stops]
    num_nodes = len(node_ids)

    road_distance = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    travel_time = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    signal_count = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    turn_count = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    left_turn_count = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    u_turn_count = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    one_way_factor = np.ones((num_nodes, num_nodes), dtype=np.float32)

    for i, src in enumerate(node_ids):
        for j, dst in enumerate(node_ids):
            if i == j:
                continue
            path = _best_path(graph, src, dst)
            road_distance[i, j] = _path_length(graph, path)
            travel_time[i, j] = _path_travel_time(graph, path)
            signal_count[i, j] = _path_signal_count(graph, path)
            turn_count[i, j] = _path_turn_count(path)
            left_turn_count[i, j] = np.floor(turn_count[i, j] * 0.35)

            reverse_exists = nx.has_path(graph, dst, src)
            one_way_factor[i, j] = 1.0 if reverse_exists else 1.15

    return build_road_aware_matrices(
        road_distance=road_distance,
        base_travel_time=travel_time,
        signal_count=signal_count,
        turn_count=turn_count,
        left_turn_count=left_turn_count,
        u_turn_count=u_turn_count,
        one_way_factor=one_way_factor,
        time_bucket=time_bucket,
        penalty=penalty,
    )


if __name__ == "__main__":
    example_stops = [
        StopPoint(lat=31.2304, lon=121.4737),
        StopPoint(lat=31.2282, lon=121.4823),
        StopPoint(lat=31.2216, lon=121.4901),
    ]
    graph = download_drive_graph("Huangpu District, Shanghai, China")
    matrices = build_pairwise_road_matrices(graph, example_stops)
    print("edge_attr shape:", matrices.edge_attr().shape)
