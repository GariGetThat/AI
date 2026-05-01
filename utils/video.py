# utils/video.py
"""영상 읽기 / 윈도우 분할 유틸리티"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple

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
        return self.total_frames / self.fps


def get_video_meta(video_path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    meta = VideoMeta(
        path=str(video_path),
        fps=cap.get(cv2.CAP_PROP_FPS),
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
    """(frame_idx, frame_bgr) 제너레이터"""
    cap = cv2.VideoCapture(str(video_path))
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


@dataclass
class WindowRange:
    start_frame: int
    end_frame: int  # exclusive


def build_windows(
    total_frames: int,
    fps: float,
    window_sec: float,
    overlap_sec: float,
) -> List[WindowRange]:
    """슬라이딩 윈도우 리스트 생성"""
    window_frames  = int(window_sec  * fps)
    overlap_frames = int(overlap_sec * fps)
    stride         = window_frames - overlap_frames

    windows: List[WindowRange] = []
    start = 0
    while start < total_frames:
        end = min(start + window_frames, total_frames)
        windows.append(WindowRange(start, end))
        if end == total_frames:
            break
        start += stride

    return windows