from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class OpenVRPInstance:
    name: str
    source_path: str
    coord: np.ndarray  # (N+1, 2), depot is index 0
    demand: np.ndarray  # (N+1,), depot demand is 0
    capacity: float

    @property
    def n_customers(self) -> int:
        return int(self.coord.shape[0] - 1)


def _parse_cvrplib_file(path: Path) -> OpenVRPInstance:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Prefer filename stem to avoid inconsistent NAME fields in some public files.
    name = path.stem
    dimension = None
    capacity = None
    coords = {}
    demands = {}
    depot_ids = []
    section = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        upper = line.upper()
        if upper == "NODE_COORD_SECTION":
            section = "coord"
            continue
        if upper == "DEMAND_SECTION":
            section = "demand"
            continue
        if upper == "DEPOT_SECTION":
            section = "depot"
            continue
        if upper == "EOF":
            break

        if section is None:
            if ":" in line:
                key, val = line.split(":", 1)
            else:
                toks = line.split()
                if len(toks) >= 2:
                    key, val = toks[0], " ".join(toks[1:])
                else:
                    continue
            key_u = key.strip().upper()
            val = val.strip()
            if key_u == "DIMENSION":
                dimension = int(float(val))
            elif key_u == "CAPACITY":
                capacity = float(val)
            continue

        toks = line.split()
        if section == "coord":
            if len(toks) < 3:
                continue
            node_id = int(float(toks[0]))
            x = float(toks[1])
            y = float(toks[2])
            coords[node_id] = (x, y)
        elif section == "demand":
            if len(toks) < 2:
                continue
            node_id = int(float(toks[0]))
            d = float(toks[1])
            demands[node_id] = d
        elif section == "depot":
            node_id = int(float(toks[0]))
            if node_id == -1:
                section = None
            elif node_id > 0:
                depot_ids.append(node_id)

    if dimension is None:
        if coords:
            dimension = len(coords)
        elif demands:
            dimension = len(demands)
        else:
            raise ValueError(f"Cannot infer DIMENSION from file: {path}")
    if capacity is None:
        raise ValueError(f"Missing CAPACITY in CVRPLIB file: {path}")
    if len(coords) == 0 or len(demands) == 0:
        raise ValueError(f"Missing NODE_COORD_SECTION or DEMAND_SECTION in file: {path}")

    if len(depot_ids) == 0:
        depot_id = 1
    else:
        depot_id = int(depot_ids[0])

    all_node_ids = sorted(coords.keys())
    if len(all_node_ids) != int(dimension):
        # Keep robust behavior for slightly non-standard files.
        dimension = len(all_node_ids)
    if depot_id not in coords:
        raise ValueError(f"Depot id {depot_id} not found in coordinates for file: {path}")

    customer_ids = [nid for nid in all_node_ids if nid != depot_id]
    ordered_ids = [depot_id] + customer_ids

    coord = np.zeros((dimension, 2), dtype=np.float32)
    demand = np.zeros((dimension,), dtype=np.float32)
    for idx, nid in enumerate(ordered_ids):
        if nid not in coords:
            raise ValueError(f"Node {nid} missing in NODE_COORD_SECTION for file: {path}")
        coord[idx, 0] = float(coords[nid][0])
        coord[idx, 1] = float(coords[nid][1])
        demand[idx] = float(demands.get(nid, 0.0))
    demand[0] = 0.0

    return OpenVRPInstance(
        name=str(name),
        source_path=str(path.resolve()),
        coord=coord,
        demand=demand,
        capacity=float(capacity),
    )


def load_cvrplib_instances(dataset_path: str) -> List[OpenVRPInstance]:
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset path not found: {root}")

    files: List[Path]
    if root.is_file():
        files = [root]
    else:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".vrp"]
    if len(files) == 0:
        raise ValueError(f"No .vrp files found under: {root}")

    instances = [_parse_cvrplib_file(p) for p in sorted(files)]
    if len(instances) == 0:
        raise ValueError(f"No valid CVRPLIB instances parsed from: {root}")
    return instances


def read_instance_name_list(list_path: str) -> List[str]:
    path = Path(list_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"instance list file not found: {path}")
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if len(out) == 0:
        raise ValueError(f"instance list file is empty: {path}")
    return out


def load_cvrplib_instances_filtered(
    dataset_path: str,
    include_names: Optional[List[str]] = None,
) -> List[OpenVRPInstance]:
    instances = load_cvrplib_instances(dataset_path)
    if include_names is None:
        return instances

    wanted: Set[str] = set()
    for name in include_names:
        n = str(name).strip()
        if not n:
            continue
        if n.lower().endswith(".vrp"):
            wanted.add(n[:-4])
        wanted.add(n)
    if len(wanted) == 0:
        raise ValueError("include_names must not be empty")

    filtered = [x for x in instances if x.name in wanted or f"{x.name}.vrp" in wanted]
    if len(filtered) == 0:
        raise ValueError(
            f"No instances matched include_names under dataset path. "
            f"dataset_path={dataset_path}, include_count={len(wanted)}"
        )
    return filtered


def _normalize_coord_scale(coord: np.ndarray, coord_scale: float) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float32)
    lo = coord.min(axis=0)
    hi = coord.max(axis=0)
    span = float(np.max(hi - lo))
    if span <= 1e-9:
        return coord.copy()
    return ((coord - lo) / span * float(coord_scale)).astype(np.float32)


def sample_open_vrp_base(
    instances: List[OpenVRPInstance],
    N: int,
    seed: int,
    coord_scale: float = 10.0,
    normalize_coords: bool = True,
    demand_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, OpenVRPInstance]:
    if len(instances) == 0:
        raise ValueError("instances must not be empty")
    if N <= 0:
        raise ValueError("N must be >= 1")
    if demand_scale <= 0:
        raise ValueError("demand_scale must be > 0")

    rng = np.random.default_rng(seed)
    chosen = instances[int(rng.integers(0, len(instances)))]
    if chosen.n_customers < N:
        raise ValueError(
            f"Instance '{chosen.name}' has only {chosen.n_customers} customers, "
            f"but N={N} is requested."
        )

    all_customers = np.arange(1, chosen.coord.shape[0], dtype=np.int64)
    picked_customers = rng.choice(all_customers, size=(N,), replace=False)
    node_ids = np.concatenate(([0], picked_customers)).astype(np.int64)

    coord = chosen.coord[node_ids].astype(np.float32)
    demand = chosen.demand[node_ids].astype(np.float32)
    demand[0] = 0.0

    cap_ref = max(1e-6, float(chosen.capacity))
    demand[1:] = demand[1:] / cap_ref * float(demand_scale)
    if normalize_coords:
        coord = _normalize_coord_scale(coord, coord_scale=coord_scale)

    return coord, demand, chosen
