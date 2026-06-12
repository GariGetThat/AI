# models/face_recognizer.py

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import config

class BuffaloFaceRecognizer:
    """
    PASS2에서 ArcFace embedding을 추출한다.

    기존 방식:
    - bbox crop을 112x112로 resize 후 embedding 추출

    개선 방식:
    - 가능하면 원본 frame + 얼굴 landmark(kps)를 이용해 얼굴 정렬(alignment)
    - 정렬된 얼굴에서 embedding 추출
    - embedding은 cosine distance 비교를 위해 L2 normalize
    """

    def __init__(
        self,
        model_pack_name: str = "buffalo_l",
        ctx_id: int = 0,
    ):
        from insightface.model_zoo import get_model

        model_path = (
            Path.home()
            / ".insightface"
            / "models"
            / model_pack_name
            / config.RECOGNIZER_MODEL_NAME
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Recognition model not found: {model_path}\n"
                "먼저 buffalo_l 모델이 다운로드되어 있는지 확인하세요."
            )

        providers = (
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )

        self.model = get_model(
            str(model_path),
            providers=providers,
        )

        self.model.prepare(ctx_id=ctx_id)

    def get_embedding(
        self,
        crop: np.ndarray | None = None,
        frame: np.ndarray | None = None,
        kps: list | np.ndarray | None = None,
    ) -> List[float] | None:
        """
        embedding 추출

        우선순위:
        1. frame + kps가 있으면 landmark 기반 얼굴 정렬 후 embedding 추출
        2. 없으면 기존 방식대로 crop resize 후 embedding 추출
        """

        face_img = None

        if frame is not None and frame.size > 0 and kps is not None:
            face_img = self._align_face(frame, kps)

        if face_img is None:
            if crop is None or crop.size == 0:
                return None

            face_img = cv2.resize(crop, config.RECOGNIZER_INPUT_SIZE)

        feat = self.model.get_feat(face_img)

        if feat is None:
            return None

        feat = np.asarray(feat, dtype=np.float32).reshape(-1)

        norm = np.linalg.norm(feat)

        if norm <= 1e-12:
            return None

        feat = feat / norm

        return feat.tolist()
    
    def _align_face(
        self,
        frame: np.ndarray,
        kps: list | np.ndarray,
    ) -> np.ndarray | None:
        """
        InsightFace landmark 기반 얼굴 정렬.

        ArcFace는 단순 crop보다 정렬된 얼굴 입력에서 더 안정적인 embedding을 만든다.
        """

        try:
            from insightface.utils import face_align

            kps_np = np.asarray(kps, dtype=np.float32)

            if kps_np.shape != (5, 2):
                return None

            aligned = face_align.norm_crop(
                frame,
                landmark=kps_np,
                image_size=config.RECOGNIZER_INPUT_SIZE[0],
            )

            return aligned

        except Exception:
            return None


def build_recognizer(
    model_pack_name: str = "buffalo_l",
    ctx_id: int = 0,
) -> BuffaloFaceRecognizer:
    return BuffaloFaceRecognizer(
        model_pack_name=model_pack_name,
        ctx_id=ctx_id,
    )