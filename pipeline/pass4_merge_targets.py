# pipeline/pass4_merge_targets.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import config
from utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def _normalize_object_targets(raw_object_db: Any) -> List[Dict[str, Any]]:
    """
    PASS3 object_db.json을 SAM2 target 형식으로 변환.

    기대 출력 형식:
    {
        "id": "...",
        "type": "object" | "text" | ...,
        "start_frame": int,
        "end_frame": int,
        "bbox": [x1, y1, x2, y2]
    }
    """

    targets = []

    if raw_object_db is None:
        return targets

    if isinstance(raw_object_db, dict):
        items = raw_object_db.get("tracks", raw_object_db.get("objects", raw_object_db))
        if isinstance(items, dict):
            items = list(items.values())
    else:
        items = raw_object_db

    if not isinstance(items, list):
        logger.warning("object_db 형식을 해석할 수 없습니다. object target을 건너뜁니다.")
        return targets

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        bbox = (
            item.get("bbox")
            or item.get("box")
            or item.get("representative_box")
        )

        if bbox is None or len(bbox) != 4:
            continue

        start_frame = item.get("start_frame", item.get("frame_index", item.get("frame", 0)))
        end_frame = item.get("end_frame", start_frame)

        label = item.get("label", item.get("type", "object"))
        object_id = item.get("id", item.get("object_id", f"object_{idx:04d}"))

        targets.append(
            {
                "id": str(object_id),
                "type": str(label),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "bbox": [float(v) for v in bbox],
            }
        )

    return targets


def run_pass4(
    face_targets_path: str | Path = config.SAM2_INPUT_PATH,
    object_db_path: str | Path = config.OBJECT_DB_PATH,
    output_path: str | Path = config.SAM2_TARGETS_PATH,
) -> List[Dict[str, Any]]:
    face_targets_path = Path(face_targets_path)
    object_db_path = Path(object_db_path)
    output_path = Path(output_path)

    face_targets = []

    if face_targets_path.exists():
        face_targets = load_json(face_targets_path)
    else:
        logger.warning("face target 파일 없음: %s", face_targets_path)

    raw_object_db = None

    if object_db_path.exists():
        raw_object_db = load_json(object_db_path)
    else:
        logger.warning("object db 파일 없음: %s", object_db_path)

    object_targets = _normalize_object_targets(raw_object_db)

    merged_targets = []

    if isinstance(face_targets, list):
        merged_targets.extend(face_targets)
    else:
        logger.warning("face target 형식이 list가 아닙니다.")

    merged_targets.extend(object_targets)

    save_json(merged_targets, output_path)

    logger.info(
        "PASS4 완료 | face=%d | object=%d | total=%d | 저장=%s",
        len(face_targets) if isinstance(face_targets, list) else 0,
        len(object_targets),
        len(merged_targets),
        output_path,
    )

    return merged_targets


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s : %(message)s",
    )

    run_pass4()