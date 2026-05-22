# utils/crop.py
"""얼굴 crop 추출 및 저장 유틸리티"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def crop_face(
    frame: np.ndarray,
    bbox: BBox,
    padding: float = 0.35,
    min_size: int = 40,
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return frame[y1:y2, x1:x2].copy()


def crop_face_by_kps(
    frame: np.ndarray,
    kps: list | None,
    bbox: BBox | None = None,
    padding: float = 0.8,
    min_size: int = 40,
) -> np.ndarray | None:
    if kps is None:
        if bbox is None:
            return None
        return crop_face(frame, bbox, padding=0.25, min_size=min_size)

    h, w = frame.shape[:2]
    pts = np.array(kps, dtype=np.float32)

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        if bbox is None:
            return None
        return crop_face(frame, bbox, padding=0.25, min_size=min_size)

    size = int(max(bw, bh) * (1.0 + padding))
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    nx1 = max(0, cx - size // 2)
    ny1 = max(0, cy - size // 2)
    nx2 = min(w, cx + size // 2)
    ny2 = min(h, cy + size // 2)

    if (nx2 - nx1) < min_size or (ny2 - ny1) < min_size:
        return None

    return frame[ny1:ny2, nx1:nx2].copy()


def save_crop(
    crop: np.ndarray,
    save_path: str | Path,
    quality: int = 90,
) -> bool:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok, buf = cv2.imencode(
        ".jpg",
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if ok:
        buf.tofile(str(save_path))

    return ok


def _crop_quality_score(
    crop: np.ndarray,
    confidence: float,
) -> float:
    h, w = crop.shape[:2]
    area = h * w

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return confidence * 1.0 + min(area / 10000.0, 1.0) * 0.3 + min(sharpness / 100.0, 1.0) * 0.3


def select_best_crop(
    crops: List[Tuple[np.ndarray, float]],
) -> np.ndarray | None:
    if not crops:
        return None

    return max(
        crops,
        key=lambda x: _crop_quality_score(x[0], x[1]),
    )[0]