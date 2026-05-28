# utils/video.py
"""영상 읽기 유틸리티"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration_sec(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0


def get_video_meta(video_path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    meta = VideoMeta(
        path=str(video_path),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    cap.release()
    return meta


def iter_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    step: int = 1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if end_frame is not None and idx >= end_frame:
            break

        if (idx - start_frame) % step == 0:
            yield idx, frame

        idx += 1

    cap.release()