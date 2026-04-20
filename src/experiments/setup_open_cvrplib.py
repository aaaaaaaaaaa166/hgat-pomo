from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from http.client import IncompleteRead
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.env.open_data_loader import OpenVRPInstance, load_cvrplib_instances


DEFAULT_TARBALL_URL = "https://registry.npmjs.org/vrpinstances/-/vrpinstances-1.0.3.tgz"
DEFAULT_OFFICIAL_INDEX_URL = "https://vrp.galgos.inf.puc-rio.br/index.php/en/"
DEFAULT_OFFICIAL_DOWNLOAD_HOST = "https://galgos.inf.puc-rio.br"
NAME_RE = re.compile(r"^\s*NAME\s*[: ]\s*(?P<name>[A-Za-z0-9-]+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class OfficialInstanceMeta:
    name: str
    family: str
    n_customers: int
    trucks: int
    capacity: float
    download_url: str


def _download_with_retries(url: str, out_path: Path, retries: int = 8, timeout_sec: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return
        except IncompleteRead as exc:
            last_err = exc
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
        time.sleep(1.0 + 0.5 * attempt)
    raise RuntimeError(f"Failed to download after {retries} retries: {url}") from last_err


def _extract_vrp_files(tarball_path: Path, out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    with tarfile.open(tarball_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".vrp")]
        for m in members:
            target_name = Path(m.name).name
            if not target_name:
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            data = src.read()
            text_head = data.decode("utf-8", errors="ignore").splitlines()[:20]
            canonical_name = target_name
            for raw in text_head:
                match = NAME_RE.match(raw.strip())
                if match:
                    canonical_name = f"{match.group('name').strip()}.vrp"
                    break
            target_path = out_dir / canonical_name
            target_path.write_bytes(data)
            if canonical_name != target_name:
                alias_path = out_dir / target_name
                if alias_path.exists():
                    alias_path.unlink()
            names.append(canonical_name)
    return sorted(set(names))


def _find_curl_exe() -> str:
    for cand in ("curl.exe", "curl"):
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError("curl executable not found; required for official CVRPLIB downloads.")


def _curl_fetch_text(url: str, retries: int = 5, timeout_sec: int = 120) -> str:
    curl_exe = _find_curl_exe()
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                [
                    curl_exe,
                    "-fsSL",
                    "--max-time",
                    str(timeout_sec),
                    "-A",
                    "Mozilla/5.0",
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return proc.stdout
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            time.sleep(1.0 + 0.5 * attempt)
    raise RuntimeError(f"Failed to fetch text after {retries} retries: {url}") from last_err


def _curl_download(url: str, out_path: Path, retries: int = 5, timeout_sec: int = 120) -> None:
    curl_exe = _find_curl_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            subprocess.run(
                [
                    curl_exe,
                    "-fsSL",
                    "--max-time",
                    str(timeout_sec),
                    "-A",
                    "Mozilla/5.0",
                    url,
                    "-o",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            tmp_path.replace(out_path)
            return
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            time.sleep(1.0 + 0.5 * attempt)
    raise RuntimeError(f"Failed to download after {retries} retries: {url}") from last_err


def _parse_families(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in text.split(","):
        family = raw.strip().upper()
        if not family or family in seen:
            continue
        seen.add(family)
        out.append(family)
    return out


def _family_of_name(name: str) -> str:
    stem = Path(name).stem
    return stem.split("-", 1)[0].upper()


def _discover_official_instances(
    index_url: str,
    include_families: Sequence[str],
    min_customers: int,
    download_host: str,
) -> Dict[str, OfficialInstanceMeta]:
    html = _curl_fetch_text(index_url)
    pattern = re.compile(
        r'<a href="(?P<href>/cvrplib/index\.php/en/download/instance/\d+)" title="Instance File">\s*'
        r"(?P<name>[A-Za-z0-9-]+)\s*</a>\s*</td>\s*"
        r'<td class="text-end">\$(?P<n_customers>\d+)\$</td>\s*'
        r'<td class="text-end">\$(?P<trucks>\d+)\$</td>\s*'
        r'<td class="text-end">\$(?P<capacity>[0-9.]+)\$</td>',
        re.S,
    )

    fam_set = set(include_families)
    discovered: Dict[str, OfficialInstanceMeta] = {}
    for match in pattern.finditer(html):
        name = match.group("name").strip()
        family = _family_of_name(name)
        n_customers = int(match.group("n_customers"))
        if fam_set and family not in fam_set:
            continue
        if n_customers < int(min_customers):
            continue

        href = match.group("href").strip()
        if href.startswith("http://") or href.startswith("https://"):
            download_url = href
        else:
            download_url = download_host.rstrip("/") + href
        discovered[name] = OfficialInstanceMeta(
            name=name,
            family=family,
            n_customers=n_customers,
            trucks=int(match.group("trucks")),
            capacity=float(match.group("capacity")),
            download_url=download_url,
        )
    return discovered


def _sync_official_instances(
    instances: Dict[str, OfficialInstanceMeta],
    out_dir: Path,
    refresh_download: bool,
) -> Dict[str, int]:
    downloaded = 0
    reused = 0
    for meta in sorted(instances.values(), key=lambda x: (x.family, x.n_customers, x.name)):
        target = out_dir / f"{meta.name}.vrp"
        if refresh_download or not target.exists():
            _curl_download(meta.download_url, target)
            downloaded += 1
        else:
            reused += 1
    return {"downloaded": downloaded, "reused": reused}


def _split_names_by_family(
    names: Sequence[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    include_families: Sequence[str],
) -> Dict[str, List[str]]:
    fam_set = {x.strip().upper() for x in include_families if x.strip()}
    grouped: Dict[str, List[str]] = {}
    for name in names:
        stem = Path(name).stem
        family = _family_of_name(stem)
        if fam_set and family not in fam_set:
            continue
        grouped.setdefault(family, []).append(stem)

    missing = sorted(fam_set.difference(grouped.keys()))
    if missing:
        raise ValueError(f"Requested families have no eligible instances: {missing}")

    rng = np.random.default_rng(seed)
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    for family in sorted(grouped):
        arr = sorted(grouped[family])
        perm = rng.permutation(len(arr))
        arr = [arr[int(i)] for i in perm]
        n = len(arr)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        else:
            n_train = min(max(1, n_train), n)
            n_val = max(0, min(n_val, n - n_train))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
        train.extend(arr[:n_train])
        val.extend(arr[n_train : n_train + n_val])
        test.extend(arr[n_train + n_val : n_train + n_val + n_test])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def _write_list(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for n in names:
            f.write(f"{n}\n")


def _write_instance_manifest(
    path: Path,
    instances: Sequence[OpenVRPInstance],
    split_lookup: Dict[str, str],
    official_meta: Dict[str, OfficialInstanceMeta],
    tarball_stems: Sequence[str],
    tarball_url: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tarball_set = set(tarball_stems)
    headers = [
        "name",
        "family",
        "n_customers",
        "capacity",
        "split",
        "source_kind",
        "source_url",
        "local_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for inst in sorted(instances, key=lambda x: (_family_of_name(x.name), x.n_customers, x.name)):
            family = _family_of_name(inst.name)
            if inst.name in tarball_set:
                source_kind = "tarball"
                source_url = tarball_url
            elif inst.name in official_meta:
                source_kind = "official"
                source_url = official_meta[inst.name].download_url
            else:
                source_kind = "existing"
                source_url = ""
            writer.writerow(
                {
                    "name": inst.name,
                    "family": family,
                    "n_customers": int(inst.n_customers),
                    "capacity": float(inst.capacity),
                    "split": split_lookup.get(inst.name, ""),
                    "source_kind": source_kind,
                    "source_url": source_url,
                    "local_path": str(Path(inst.source_path).resolve()),
                }
            )


def _family_summary_rows(
    instances: Sequence[OpenVRPInstance],
    split_lookup: Dict[str, str],
    include_families: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[OpenVRPInstance]] = {}
    for inst in instances:
        family = _family_of_name(inst.name)
        grouped.setdefault(family, []).append(inst)

    families = list(include_families) if include_families else sorted(grouped.keys())
    rows: List[Dict[str, object]] = []
    for family in families:
        items = grouped.get(family, [])
        if not items:
            rows.append(
                {
                    "family": family,
                    "total_instances": 0,
                    "min_customers": 0,
                    "max_customers": 0,
                    "avg_customers": 0.0,
                    "train": 0,
                    "val": 0,
                    "test": 0,
                }
            )
            continue
        customer_counts = [int(x.n_customers) for x in items]
        rows.append(
            {
                "family": family,
                "total_instances": len(items),
                "min_customers": min(customer_counts),
                "max_customers": max(customer_counts),
                "avg_customers": float(sum(customer_counts) / len(customer_counts)),
                "train": sum(1 for x in items if split_lookup.get(x.name) == "train"),
                "val": sum(1 for x in items if split_lookup.get(x.name) == "val"),
                "test": sum(1 for x in items if split_lookup.get(x.name) == "test"),
            }
        )
    return rows


def _write_family_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["family", "total_instances", "min_customers", "max_customers", "avg_customers", "train", "val", "test"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare open CVRPLIB instances and train/val/test split files.")
    parser.add_argument("--dataset-dir", type=str, default="datasets/cvrplib")
    parser.add_argument("--tarball-url", type=str, default=DEFAULT_TARBALL_URL)
    parser.add_argument("--cache-tarball", type=str, default="datasets/vrpinstances-1.0.3.tgz")
    parser.add_argument("--refresh-download", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--include-families", type=str, default="A,B,P")
    parser.add_argument("--min-customers", type=int, default=30)
    parser.add_argument("--split-dir", type=str, default="")
    parser.add_argument("--official-index-url", type=str, default=DEFAULT_OFFICIAL_INDEX_URL)
    parser.add_argument("--official-download-host", type=str, default=DEFAULT_OFFICIAL_DOWNLOAD_HOST)
    parser.add_argument("--skip-tarball-source", action="store_true")
    parser.add_argument("--skip-official-source", action="store_true")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("Require 0 < train_ratio, val_ratio and train_ratio + val_ratio < 1")

    dataset_dir = Path(args.dataset_dir).resolve()
    tarball_path = Path(args.cache_tarball).resolve()
    split_dir = Path(args.split_dir).resolve() if args.split_dir else dataset_dir / "splits"
    fams = _parse_families(args.include_families)

    extracted: List[str] = []
    if not args.skip_tarball_source:
        if args.refresh_download or not tarball_path.exists():
            print(f"Downloading CVRPLIB tarball: {args.tarball_url}")
            _download_with_retries(args.tarball_url, tarball_path)
        else:
            print(f"Using cached tarball: {tarball_path}")

        extracted = _extract_vrp_files(tarball_path, dataset_dir)
        print(f"Extracted {len(extracted)} .vrp files from tarball into {dataset_dir}")
    else:
        print("Skipping tarball source.")

    official_meta: Dict[str, OfficialInstanceMeta] = {}
    official_sync = {"downloaded": 0, "reused": 0}
    if not args.skip_official_source:
        official_meta = _discover_official_instances(
            index_url=args.official_index_url,
            include_families=fams,
            min_customers=int(args.min_customers),
            download_host=args.official_download_host,
        )
        official_sync = _sync_official_instances(
            instances=official_meta,
            out_dir=dataset_dir,
            refresh_download=bool(args.refresh_download),
        )
        print(
            "Official CVRPLIB sync: "
            f"{len(official_meta)} candidates, downloaded={official_sync['downloaded']}, reused={official_sync['reused']}"
        )
    else:
        print("Skipping official CVRPLIB source.")

    if args.skip_tarball_source and args.skip_official_source:
        raise ValueError("At least one source must be enabled.")

    all_instances = load_cvrplib_instances(str(dataset_dir))
    filtered = [
        x
        for x in all_instances
        if x.n_customers >= int(args.min_customers) and (not fams or _family_of_name(x.name) in set(fams))
    ]
    if len(filtered) == 0:
        raise ValueError(f"No instances with n_customers >= {args.min_customers} in {dataset_dir}")

    eligible_names = [f"{x.name}.vrp" for x in filtered]
    split = _split_names_by_family(
        eligible_names,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        include_families=fams,
    )
    if len(split["train"]) == 0 or len(split["val"]) == 0 or len(split["test"]) == 0:
        raise ValueError("Split failed: one of train/val/test is empty. Adjust ratios or dataset.")

    _write_list(split_dir / "train.txt", split["train"])
    _write_list(split_dir / "val.txt", split["val"])
    _write_list(split_dir / "test.txt", split["test"])

    split_lookup: Dict[str, str] = {}
    for split_name, names in split.items():
        for name in names:
            split_lookup[name] = split_name

    manifest_path = split_dir / "instance_manifest.csv"
    family_summary_path = split_dir / "family_summary.csv"
    _write_instance_manifest(
        manifest_path,
        instances=filtered,
        split_lookup=split_lookup,
        official_meta=official_meta,
        tarball_stems=[Path(x).stem for x in extracted],
        tarball_url=args.tarball_url,
    )
    family_rows = _family_summary_rows(filtered, split_lookup=split_lookup, include_families=fams)
    _write_family_summary(family_summary_path, family_rows)

    summary = {
        "dataset_dir": str(dataset_dir),
        "tarball_path": str(tarball_path),
        "tarball_url": args.tarball_url,
        "official_index_url": args.official_index_url,
        "official_download_host": args.official_download_host,
        "requested_families": fams,
        "total_extracted_files": len(extracted),
        "official_candidates": len(official_meta),
        "official_downloaded_files": official_sync["downloaded"],
        "official_reused_files": official_sync["reused"],
        "min_customers": int(args.min_customers),
        "eligible_instances": len(filtered),
        "families": fams,
        "family_breakdown": {
            str(row["family"]): {
                "total_instances": int(row["total_instances"]),
                "min_customers": int(row["min_customers"]),
                "max_customers": int(row["max_customers"]),
                "avg_customers": float(row["avg_customers"]),
                "train": int(row["train"]),
                "val": int(row["val"]),
                "test": int(row["test"]),
            }
            for row in family_rows
        },
        "split_sizes": {k: len(v) for k, v in split.items()},
        "split_files": {
            "train": str((split_dir / "train.txt").resolve()),
            "val": str((split_dir / "val.txt").resolve()),
            "test": str((split_dir / "test.txt").resolve()),
        },
        "artifact_files": {
            "instance_manifest_csv": str(manifest_path.resolve()),
            "family_summary_csv": str(family_summary_path.resolve()),
        },
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Split summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
