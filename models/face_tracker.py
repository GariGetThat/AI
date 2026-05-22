# models/face_tracker.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from db.schema import DetectionRecord, TrackRecord
from utils.geometry import iou


class BaseFaceTracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: List[DetectionRecord],
        frame_idx: int,
    ) -> List[TrackRecord]:
        ...


class ByteTrackTracker(BaseFaceTracker):
    """
    supervision.ByteTrack 사용.
    buffalo_l detection 결과 bbox를 ByteTrack에 넣어 track_id를 생성한다.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,
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

        records: List[TrackRecord] = []

        for i in range(len(tracked)):
            bbox = tracked.xyxy[i].tolist()

            score = (
                float(tracked.confidence[i])
                if tracked.confidence is not None
                else 1.0
            )

            track_id = int(tracked.tracker_id[i])

            best_det = None
            best_iou = 0.0

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


def build_tracker(
    track_thresh: float = 0.5,
    high_thresh: float = 0.6,
    match_thresh: float = 0.8,
    max_time_lost: int = 30,
    frame_rate: int = 30,
) -> BaseFaceTracker:

    return ByteTrackTracker(
        track_thresh=track_thresh,
        high_thresh=high_thresh,
        match_thresh=match_thresh,
        max_time_lost=max_time_lost,
        frame_rate=frame_rate,
    )