# models/face_recognizer.py

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import config


class BuffaloFaceRecognizer:
    """
    PASS2에서 representative crop 기준으로 ArcFace embedding만 추출.
    얼굴 detection은 다시 수행하지 않는다.
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
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )

        self.model = get_model(
            str(model_path),
            providers=providers,
        )

        self.model.prepare(ctx_id=ctx_id)

    def get_embedding(self, crop: np.ndarray) -> List[float] | None:
        if crop is None or crop.size == 0:
            return None

        # ArcFace 입력 크기: 112x112
        crop = cv2.resize(crop, config.RECOGNIZER_INPUT_SIZE)

        # recognition model만 직접 실행
        feat = self.model.get_feat(crop)

        if feat is None:
            return None

        feat = np.asarray(feat, dtype=np.float32).reshape(-1)

        return feat.tolist()


def build_recognizer(
    model_pack_name: str = "buffalo_l",
    ctx_id: int = 0,
) -> BuffaloFaceRecognizer:
    return BuffaloFaceRecognizer(
        model_pack_name=model_pack_name,
        ctx_id=ctx_id,
    )