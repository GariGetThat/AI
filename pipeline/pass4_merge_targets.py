# pipeline/pass4_merge_targets.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import config
from utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def _get_box(item: Dict[str, Any]) -> List[float] | None:
    """
    face/object 중간 결과에서 bbox 계열 좌표를 읽어온다.
    중간 단계에서는 bbox를 표준으로 쓰지만,
    혹시 기존 테스트 데이터에 box가 있으면 함께 허용한다.
    """
    box = (
        item.get("bbox")
        or item.get("box")
        or item.get("representative_box")
    )

    if box is None or len(box) != 4:
        return None

    return [float(v) for v in box]


def _normalize_face_targets(raw_face_targets: Any) -> List[Dict[str, Any]]:
    """
    export_for_sam2.py가 만든 face_sam2_input.json을
    SAM2 최종 target 형식으로 변환한다.

    최종 출력은 SAM2 코드 기준에 맞춰 box 키를 사용한다.
    """

    targets: List[Dict[str, Any]] = []

    if raw_face_targets is None:
        return targets

    if not isinstance(raw_face_targets, list):
        logger.warning("face target 형식이 list가 아닙니다. face target을 건너뜁니다.")
        return targets

    for idx, item in enumerate(raw_face_targets):
        if not isinstance(item, dict):
            continue

        box = _get_box(item)
        if box is None:
            logger.warning("face target bbox/box 없음: index=%d", idx)
            continue

        target_id = item.get("id", f"face_{idx:04d}")
        start_frame = item.get("start_frame", item.get("frame_index", item.get("frame", 0)))
        end_frame = item.get("end_frame", start_frame)

        targets.append(
            {
                "id": str(target_id),
                "type": "face",
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "box": box,
            }
        )

    return targets


def _normalize_object_targets(raw_object_db: Any) -> List[Dict[str, Any]]:
    """
    PASS3 object_db.json을 SAM2 최종 target 형식으로 변환한다.

    object_db는 list일 수도 있고,
    {"tracks": [...]}, {"objects": [...]} 형태일 수도 있어서 모두 처리한다.

    최종 출력은 SAM2 코드 기준에 맞춰 box 키를 사용한다.
    """

    targets: List[Dict[str, Any]] = []

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

        box = _get_box(item)
        if box is None:
            logger.warning("object target bbox/box 없음: index=%d", idx)
            continue

        object_id = item.get("id", item.get("object_id", f"object_{idx:04d}"))
        label = item.get("label", "object")

        start_frame = item.get("start_frame", item.get("frame_index", item.get("frame", 0)))
        end_frame = item.get("end_frame", start_frame)

        target = {
            "id": str(object_id),
            "type": "object",
            "label": str(label),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "box": box,
        }

        if "visible_text" in item:
            target["visible_text"] = item["visible_text"]

        targets.append(target)

    return targets


def run_pass4(
    face_targets_path: str | Path = config.FACE_SAM2_INPUT_PATH,
    object_db_path: str | Path = config.OBJECT_DB_PATH,
    output_path: str | Path = config.SAM2_TARGETS_PATH,
) -> List[Dict[str, Any]]:
    """
    PASS4: face target JSON과 object_db JSON을 합쳐
    PASS5/SAM2가 사용할 최종 sam2_targets.json을 생성한다.
    """

    face_targets_path = Path(face_targets_path)
    object_db_path = Path(object_db_path)
    output_path = Path(output_path)

    raw_face_targets = None
    raw_object_db = None

    if face_targets_path.exists():
        raw_face_targets = load_json(face_targets_path)
    else:
        logger.warning("face target 파일 없음: %s", face_targets_path)

    if object_db_path.exists():
        raw_object_db = load_json(object_db_path)
    else:
        logger.warning("object db 파일 없음: %s", object_db_path)

    face_targets = _normalize_face_targets(raw_face_targets)
    object_targets = _normalize_object_targets(raw_object_db)

    merged_targets: List[Dict[str, Any]] = []
    merged_targets.extend(face_targets)
    merged_targets.extend(object_targets)

    save_json(merged_targets, output_path)

    logger.info(
        "PASS4 완료 | face=%d | object=%d | total=%d | 저장=%s",
        len(face_targets),
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