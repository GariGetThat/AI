# models/face_detector.py
"""
SCRFD 얼굴 검출기 래퍼.

실제 모델 없이도 동작하도록 DummyFaceDetector 포함.
weights 경로가 존재하면 SCRFDDetector, 아니면 DummyFaceDetector 반환하는
build_detector() 팩토리 함수 사용 권장.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from db.schema import DetectionRecord


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionRecord]:
        """BGR 프레임 → DetectionRecord 리스트"""
        ...


# ─── 실제 SCRFD ──────────────────────────────────────────
class SCRFDDetector(BaseFaceDetector):
    """
    insightface 의 SCRFD ONNX 모델 사용.
    pip install insightface onnxruntime 필요.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.5,
        nms_thresh: float  = 0.4,
    ):
        try:
            from insightface.model_zoo import get_model
        except ImportError as e:
            raise ImportError("pip install insightface onnxruntime") from e

        self.model = get_model(str(model_path))
        # self.model.prepare(ctx_id=0, input_size=input_size, det_thresh=conf_thresh)     # GPU용
        self.model.prepare(ctx_id=-1, input_size=input_size, det_thresh=conf_thresh)     # CPU용
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> List[DetectionRecord]:
        # insightface SCRFD 는 (bboxes, kpss) 반환
        bboxes, kpss = self.model.detect(frame)
        records: List[DetectionRecord] = []

        if bboxes is None:
            return records

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox
            if score < self.conf_thresh:
                continue
            kps = kpss[i].tolist() if (kpss is not None) else None
            records.append(DetectionRecord(
                frame_idx=-1,       # 호출 측에서 채움
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                score=float(score),
                kps=kps,
            ))
        return records

class BuffaloFaceDetector(BaseFaceDetector):
    """
    InsightFace FaceAnalysis buffalo_l 사용.
    bbox + keypoint + embedding 추출.
    """

    def __init__(
        self,
        model_pack_name: str = "buffalo_l",
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.5,
        ctx_id: int = -1,
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

        records = []

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
                    bbox=[x1, y1, x2, y2],
                    score=score,
                    kps=kps,
                    embedding=embedding,
                )
            )

        return records

# ─── 더미 (테스트용) ─────────────────────────────────────
class DummyFaceDetector(BaseFaceDetector):
    """
    실제 모델 없이 파이프라인 흐름을 테스트할 수 있는 더미 검출기.
    프레임마다 화면 중앙 근처에 랜덤한 가짜 bbox 1~2개를 반환.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def detect(self, frame: np.ndarray) -> List[DetectionRecord]:
        h, w = frame.shape[:2]
        n = self.rng.integers(1, 3)  # 1 or 2 faces
        records = []
        for _ in range(n):
            x1 = int(self.rng.uniform(0.1, 0.6) * w)
            y1 = int(self.rng.uniform(0.1, 0.6) * h)
            bw = int(self.rng.uniform(0.1, 0.2) * w)
            bh = int(bw * 1.3)
            x2, y2 = min(x1 + bw, w), min(y1 + bh, h)
            score = float(self.rng.uniform(0.6, 0.99))
            records.append(DetectionRecord(
                frame_idx=-1,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                score=score,
            ))
        return records


# ─── 팩토리 ──────────────────────────────────────────────
def build_detector(
    use_buffalo: bool = True,
    model_pack_name: str = "buffalo_l",
    model_path: str | Path | None = None,
    **kwargs,
) -> BaseFaceDetector:

    if use_buffalo:
        return BuffaloFaceDetector(
            model_pack_name=model_pack_name,
            **kwargs,
        )

    if model_path and Path(model_path).exists():
        return SCRFDDetector(model_path, **kwargs)

    print("[FaceDetector] 모델 파일 없음 → DummyFaceDetector 사용")
    return DummyFaceDetector()