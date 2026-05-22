# utils/geometry.py
"""bbox 관련 기하 연산"""

from __future__ import annotations

from typing import Tuple


BBox = Tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def scale_bbox(bbox: BBox, sx: float, sy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def clip_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = bbox

    return (
        max(0, min(x1, w)),
        max(0, min(y1, h)),
        max(0, min(x2, w)),
        max(0, min(y2, h)),
    )