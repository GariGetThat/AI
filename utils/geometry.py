# utils/geometry.py
"""bbox 관련 기하 연산"""

from __future__ import annotations
from typing import Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def scale_bbox(bbox: BBox, sx: float, sy: float) -> BBox:
    """bbox 좌표를 스케일 팩터로 변환 (리사이즈 후 원본 좌표 복원 등)"""
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