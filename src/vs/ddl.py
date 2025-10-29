from __future__ import annotations

import json
import os
from typing import Any, Dict


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tdl(base_dir: str | None = None) -> Dict[str, Any]:
    base = base_dir or os.getenv("DATA_PATH", "./data")
    path = os.path.join(base, "ddl", "tdl.json")
    return _read_json(path)


def get_tdl_with_meta(base_dir: str | None = None) -> Dict[str, Any]:
    base = base_dir or os.getenv("DATA_PATH", "./data")
    path = os.path.join(base, "ddl", "tdl_with_meta.json")
    return _read_json(path)


def get_table_ddl(base_dir: str | None = None) -> Dict[str, Any]:
    base = base_dir or os.getenv("DATA_PATH", "./data")
    path = os.path.join(base, "ddl", "table_ddl.json")
    return _read_json(path)
