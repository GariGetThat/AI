# utils/io.py
"""JSON 직렬화 / 역직렬화 헬퍼"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)