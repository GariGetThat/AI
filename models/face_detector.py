# models/face_detector.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from db.schema import DetectionRecord


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionRecord]:
        ...


class BuffaloFaceDetector(BaseFaceDetector):
    """
    InsightFace FaceAnalysis buffalo_l 사용.
    detection + keypoint + embedding 추출.
    """

    def __init__(
        self,
        model_pack_name: str = "buffalo_l",
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.6,
        ctx_id: int = 0,
    ):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "pip install insightface onnxruntime-gpu"
            ) from e

        self.app = FaceAnalysis(name=model_pack_name)

        self.app.prepare(
            ctx_id=ctx_id,
            det_size=input_size,
        )

        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> List[DetectionRecord]:

        faces = self.app.get(frame)

        records: List[DetectionRecord] = []

        for face in faces:

            x1, y1, x2, y2 = face.bbox.tolist()
            score = float(face.det_score)

            if score < self.conf_thresh:
                continue

            kps = (
                face.kps.tolist()
                if hasattr(face, "kps") and face.kps is not None
                else None
            )

            embedding = (
                face.embedding.tolist()
                if hasattr(face, "embedding")
                and face.embedding is not None
                else None
            )

            records.append(
                DetectionRecord(
                    frame_idx=-1,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    score=score,
                    kps=kps,
                    embedding=embedding,
                )
            )

        return records


def build_detector(
    model_pack_name: str = "buffalo_l",
    **kwargs,
) -> BaseFaceDetector:

    return BuffaloFaceDetector(
        model_pack_name=model_pack_name,
        **kwargs,
    )