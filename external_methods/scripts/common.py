from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external_methods"
DATA_ROOT = EXTERNAL_ROOT / "data"
DOCS_ROOT = EXTERNAL_ROOT / "docs"
RESULTS_ROOT = EXTERNAL_ROOT / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(data: Any, path: str | Path) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path_obj


def dump_text(text: str, path: str | Path) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(text, encoding="utf-8")
    return path_obj


def repo_relative(path: str | Path) -> str:
    path_obj = Path(path).resolve()
    try:
        rel = path_obj.relative_to(REPO_ROOT)
    except ValueError:
        return str(path_obj)
    return str(rel).replace("\\", "/")

