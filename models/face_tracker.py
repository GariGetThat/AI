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
    supervision.ByteTrack 사용.
    buffalo_l detection 결과 bbox를 ByteTrack에 넣어 track_id를 생성한다.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,   # supervision에서는 직접 안 쓸 수 있음
        match_thresh: float = 0.8,
        max_time_lost: int = 30,
        frame_rate: int = 30,
    ):
        try:
            import supervision as sv
        except ImportError as e:
            raise ImportError("pip install supervision") from e

        self.sv = sv

        try:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=track_thresh,
                lost_track_buffer=max_time_lost,
                minimum_matching_threshold=match_thresh,
                frame_rate=frame_rate,
            )
        except TypeError:
            self._tracker = sv.ByteTrack(
                track_thresh=track_thresh,
                track_buffer=max_time_lost,
                match_thresh=match_thresh,
                frame_rate=frame_rate,
            )

    def update(
        self,
        detections: List[DetectionRecord],
        frame_idx: int,
    ) -> List[TrackRecord]:
        if not detections:
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.score for d in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=int)

        sv_detections = self.sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self._tracker.update_with_detections(sv_detections)

        records = []

        for i in range(len(tracked)):
            bbox = tracked.xyxy[i].tolist()
            score = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            track_id = int(tracked.tracker_id[i])

            # tracked bbox와 원래 detection을 IoU로 다시 매칭해서
            # kps, embedding을 붙인다.
            best_det = None
            best_iou = 0.0

            from utils.geometry import iou

            for det in detections:
                iou_score = iou(tuple(bbox), tuple(det.bbox))
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = det

            records.append(
                TrackRecord(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    bbox=[float(v) for v in bbox],
                    score=score,
                    kps=best_det.kps if best_det is not None else None,
                    embedding=best_det.embedding if best_det is not None else None,
                )
            )

        return records

    def reset(self) -> None:
        # ByteTrack은 전체 영상에서 ID를 유지해야 하므로
        # window마다 reset하지 않는 것이 좋다.
        pass

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