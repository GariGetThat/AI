# utils/io.py
"""JSON 직렬화 / 역직렬화 헬퍼"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image(path: str | Path) -> np.ndarray | None:
    path = Path(path)

    if not path.exists():
        return None

    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)