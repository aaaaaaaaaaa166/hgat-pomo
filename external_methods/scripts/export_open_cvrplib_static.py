from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from common import dump_json, ensure_dir, repo_relative
from src.env.open_data_loader import (
    load_cvrplib_instances_filtered,
    read_instance_name_list,
    sample_open_vrp_base,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the repo's open CVRPLIB split protocol into static datasets for external methods.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/cvrplib",
        help="Directory containing CVRPLIB .vrp files.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="datasets/cvrplib/splits",
        help="Directory containing train.txt / val.txt / test.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_methods/data/open_cvrplib_n30",
        help="Output root for exported datasets.",
    )
    parser.add_argument("--problem-size", type=int, default=30, help="Number of customers per sampled instance.")
    parser.add_argument("--coord-scale", type=float, default=10.0, help="Coordinate normalization scale.")
    parser.add_argument("--demand-scale", type=float, default=1.0, help="Demand scaling after capacity normalization.")
    parser.add_argument("--train-size", type=int, default=10000, help="Number of train samples.")
    parser.add_argument("--val-size", type=int, default=1000, help="Number of validation samples.")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test samples.")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260420,
        help="Base seed for deterministic offline export.",
    )
    parser.add_argument(
        "--no-normalize-coords",
        action="store_true",
        help="Disable coordinate normalization before export.",
    )
    return parser.parse_args()


def _split_config(args: argparse.Namespace) -> List[tuple[str, int, int]]:
    return [
        ("train", int(args.train_size), 0),
        ("val", int(args.val_size), 1_000_000),
        ("test", int(args.test_size), 2_000_000),
    ]


def _family_name(instance_name: str) -> str:
    name = str(instance_name).strip()
    return name[:1].upper() if name else "?"


def _save_common_npz(
    split_dir: Path,
    coords: np.ndarray,
    demands: np.ndarray,
    source_names: List[str],
    seeds: List[int],
    source_paths: List[str],
) -> Dict[str, str]:
    common_dir = ensure_dir(split_dir / "common")
    npz_path = common_dir / "samples.npz"
    np.savez_compressed(
        npz_path,
        coord=coords.astype(np.float32),
        demand=demands.astype(np.float32),
    )

    manifest_csv = common_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "seed", "source_instance", "family", "source_path"],
        )
        writer.writeheader()
        for idx, (seed, name, path) in enumerate(zip(seeds, source_names, source_paths)):
            writer.writerow(
                {
                    "sample_idx": idx,
                    "seed": int(seed),
                    "source_instance": str(name),
                    "family": _family_name(name),
                    "source_path": str(path),
                }
            )
    return {
        "common_npz": repo_relative(npz_path),
        "common_manifest_csv": repo_relative(manifest_csv),
    }


def _save_attention_pickle(split_dir: Path, coords: np.ndarray, demands: np.ndarray) -> str:
    out_dir = ensure_dir(split_dir / "attention_learn_to_route")
    out_path = out_dir / "dataset.pkl"
    records = []
    for coord, demand in zip(coords, demands):
        depot = coord[0].tolist()
        locs = coord[1:].tolist()
        node_demand = demand[1:].tolist()
        records.append((depot, locs, node_demand, 1.0))
    with out_path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return repo_relative(out_path)


def _save_pomo_tensor(split_dir: Path, coords: np.ndarray, demands: np.ndarray) -> str:
    out_dir = ensure_dir(split_dir / "pomo")
    out_path = out_dir / "dataset.pt"
    payload = {
        "depot_xy": torch.tensor(coords[:, :1, :], dtype=torch.float32),
        "node_xy": torch.tensor(coords[:, 1:, :], dtype=torch.float32),
        "node_demand": torch.tensor(demands[:, 1:], dtype=torch.float32),
    }
    torch.save(payload, out_path)
    return repo_relative(out_path)


def _save_rl4co_npz(split_dir: Path, coords: np.ndarray, demands: np.ndarray) -> str:
    out_dir = ensure_dir(split_dir / "rl4co")
    out_path = out_dir / "dataset.npz"
    np.savez_compressed(
        out_path,
        depot=coords[:, 0, :].astype(np.float32),
        locs=coords[:, 1:, :].astype(np.float32),
        demand=demands[:, 1:].astype(np.float32),
        capacity=np.ones((coords.shape[0],), dtype=np.float32),
    )
    return repo_relative(out_path)


def export_split(
    dataset_path: str,
    split_file: Path,
    split_name: str,
    n_samples: int,
    problem_size: int,
    coord_scale: float,
    demand_scale: float,
    normalize_coords: bool,
    seed_base: int,
    split_seed_offset: int,
    output_root: Path,
) -> Dict[str, object]:
    include_names = read_instance_name_list(split_file)
    instances = load_cvrplib_instances_filtered(dataset_path, include_names=include_names)
    instances = [inst for inst in instances if inst.n_customers >= problem_size]
    if not instances:
        raise ValueError(f"No eligible instances for split={split_name} with N={problem_size}")

    coords = np.zeros((n_samples, problem_size + 1, 2), dtype=np.float32)
    demands = np.zeros((n_samples, problem_size + 1), dtype=np.float32)
    source_names: List[str] = []
    source_paths: List[str] = []
    seeds: List[int] = []

    split_dir = ensure_dir(output_root / split_name)

    for sample_idx in range(n_samples):
        sample_seed = int(seed_base + split_seed_offset + sample_idx)
        coord, demand, source = sample_open_vrp_base(
            instances=instances,
            N=problem_size,
            seed=sample_seed,
            coord_scale=coord_scale,
            normalize_coords=normalize_coords,
            demand_scale=demand_scale,
        )
        coords[sample_idx] = coord
        demands[sample_idx] = demand
        source_names.append(source.name)
        source_paths.append(source.source_path)
        seeds.append(sample_seed)

    source_counter = Counter(source_names)
    family_counter = Counter(_family_name(name) for name in source_names)
    file_map = {}
    file_map.update(_save_common_npz(split_dir, coords, demands, source_names, seeds, source_paths))
    file_map["attention_dataset_pkl"] = _save_attention_pickle(split_dir, coords, demands)
    file_map["pomo_dataset_pt"] = _save_pomo_tensor(split_dir, coords, demands)
    file_map["rl4co_dataset_npz"] = _save_rl4co_npz(split_dir, coords, demands)

    return {
        "split": split_name,
        "n_samples": int(n_samples),
        "eligible_instances": len(instances),
        "source_instance_counts": dict(sorted(source_counter.items())),
        "source_family_counts": dict(sorted(family_counter.items())),
        "files": file_map,
    }


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_dir)
    splits_dir = Path(args.splits_dir)
    normalize_coords = not args.no_normalize_coords

    split_summaries = {}
    for split_name, n_samples, split_seed_offset in _split_config(args):
        split_file = splits_dir / f"{split_name}.txt"
        split_summaries[split_name] = export_split(
            dataset_path=args.dataset_path,
            split_file=split_file,
            split_name=split_name,
            n_samples=n_samples,
            problem_size=args.problem_size,
            coord_scale=args.coord_scale,
            demand_scale=args.demand_scale,
            normalize_coords=normalize_coords,
            seed_base=args.seed_base,
            split_seed_offset=split_seed_offset,
            output_root=output_root,
        )

    protocol = {
        "source_dataset_path": repo_relative(Path(args.dataset_path)),
        "source_splits_dir": repo_relative(splits_dir),
        "output_dir": repo_relative(output_root),
        "problem_size": int(args.problem_size),
        "coord_scale": float(args.coord_scale),
        "normalize_coords": bool(normalize_coords),
        "demand_scale": float(args.demand_scale),
        "seed_base": int(args.seed_base),
        "note": (
            "These offline datasets use the same open CVRPLIB split protocol and the same "
            "sample_open_vrp_base sampler as HGAT-POMO open-data training, but freeze the stream "
            "into deterministic train/val/test files for external static CVRP methods."
        ),
        "splits": split_summaries,
    }
    dump_json(protocol, output_root / "protocol.json")
    print(f"Exported static open CVRPLIB datasets to: {output_root}")


if __name__ == "__main__":
    main()
