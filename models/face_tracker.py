# models/face_tracker.py
"""
ByteTrack 얼굴 추적기 래퍼.

실제 ByteTrack 없이도 동작하는 DummyFaceTracker 포함.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from utils.geometry import iou
from db.schema import DetectionRecord, TrackRecord


class BaseFaceTracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: List[DetectionRecord],
        frame_idx: int,
    ) -> List[TrackRecord]:
        """
        현재 프레임의 detections 를 받아 track_id 가 부여된
        TrackRecord 리스트 반환.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """트래커 상태 초기화 (새 윈도우 시작 시 호출)"""
        ...


# ─── 실제 ByteTrack ──────────────────────────────────────
class ByteTrackTracker(BaseFaceTracker):
    """
    bytetracker 패키지 사용.
    pip install bytetracker 필요.

    참고: https://github.com/ifzhang/ByteTrack
    bytetracker 패키지에 따라 API 가 다를 수 있으니
    실제 연결 시 update() 내부를 맞춰서 수정할 것.
    """

    def __init__(
        self,
        track_thresh: float  = 0.5,
        high_thresh: float   = 0.6,
        match_thresh: float  = 0.8,
        max_time_lost: int   = 30,
        frame_rate: int      = 30,
    ):
        try:
            from bytetracker import BYTETracker
        except ImportError as e:
            raise ImportError("pip install bytetracker") from e

        class _Args:
            pass

        args = _Args()
        args.track_thresh  = track_thresh
        args.track_buffer  = max_time_lost
        args.match_thresh  = match_thresh
        args.mot20         = False

        self._tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(
        self,
        detections: List[DetectionRecord],
        frame_idx: int,
    ) -> List[TrackRecord]:
        if not detections:
            return []

        # bytetracker 입력: np.array([[x1,y1,x2,y2,score], ...])
        det_np = np.array([
            [*d.bbox, d.score] for d in detections
        ], dtype=np.float32)

        online_targets = self._tracker.update(
            det_np,
            [9999, 9999],   # img size (실제 사용 시 맞춰 수정)
            (9999, 9999),
        )

        records = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1 = tlwh[0], tlwh[1]
            x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
            track_bbox = [float(x1), float(y1), float(x2), float(y2)]

            best_det = None
            best_iou = 0.0
            for det in detections:
                score_iou = iou(tuple(track_bbox), tuple(det.bbox))
                if score_iou > best_iou:
                    best_iou = score_iou
                    best_det = det

            records.append(TrackRecord(
                frame_idx=frame_idx,
                track_id=int(t.track_id),
                bbox=track_bbox,
                score=float(t.score),
                kps=best_det.kps if best_det is not None else None,
                embedding=best_det.embedding if best_det is not None else None,
            ))
        return records

    def reset(self) -> None:
        self._tracker.reset()


# ─── 더미 ────────────────────────────────────────────────
class DummyFaceTracker(BaseFaceTracker):
    """
    detection bbox 에 단순히 track_id 를 부여하는 더미 트래커.
    IoU 기반 간단 매칭으로 track_id 를 이어붙임.
    """

    def __init__(self, iou_thresh: float = 0.3):
        self.iou_thresh  = iou_thresh
        self._next_id    = 1
        self._prev_tracks: List[TrackRecord] = []

    def update(
        self,
        detections: List[DetectionRecord],
        frame_idx: int,
    ) -> List[TrackRecord]:
        from utils.geometry import iou

        if not detections:
            self._prev_tracks = []
            return []

        matched_ids = [None] * len(detections)

        # 이전 track 과 IoU 매칭
        for pi, prev in enumerate(self._prev_tracks):
            best_iou, best_di = 0.0, -1
            for di, det in enumerate(detections):
                if matched_ids[di] is not None:
                    continue
                score = iou(tuple(prev.bbox), tuple(det.bbox))
                if score > best_iou:
                    best_iou, best_di = score, di
            if best_di >= 0 and best_iou >= self.iou_thresh:
                matched_ids[best_di] = prev.track_id

        # 새 track_id 부여
        new_tracks = []
        for di, det in enumerate(detections):
            tid = matched_ids[di] if matched_ids[di] else self._next_id
            if matched_ids[di] is None:
                self._next_id += 1
            new_tracks.append(TrackRecord(
                frame_idx=frame_idx,
                track_id=tid,
                bbox=list(det.bbox),
                score=det.score,
                kps=det.kps,
                embedding=det.embedding,
            ))

        self._prev_tracks = new_tracks
        return new_tracks

    def reset(self) -> None:
        # 윈도우 넘어서도 track_id 를 연속으로 유지하고 싶으면
        # _prev_tracks 만 초기화하고 _next_id 는 유지
        self._prev_tracks = []


# ─── 팩토리 ──────────────────────────────────────────────
def build_tracker(use_real: bool = False, **kwargs) -> BaseFaceTracker:
    if use_real:
        return ByteTrackTracker(**kwargs)
    print("[FaceTracker] DummyFaceTracker 사용")
    return DummyFaceTracker()