from __future__ import annotations

import cv2
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from pipeline.pass1_face_detect_track import run_pass1
from pipeline.pass2_face_cluster import run_pass2
from pipeline.pass3_object_detect import run_pass3
from pipeline.export_for_sam2 import export_for_sam2
from pipeline.pass4_merge_targets import run_pass4
from models.sam2.chunk_processor import ChunkProcessor
from models.sam2.blur_processor import BlurProcessor
from utils.io import load_json


def get_file_hash(uploaded_file) -> str:
    """업로드 파일의 MD5 해시를 반환합니다."""
    data = bytes(uploaded_file.getbuffer())
    return hashlib.md5(data).hexdigest()


def save_uploaded_video(uploaded_file) -> Path:
    """업로드된 영상을 UUID 기반 안전한 파일명으로 저장합니다."""
    upload_dir = config.OUTPUT_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    video_path = upload_dir / safe_name

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return video_path

def put_korean_text(frame, text, position, font_size=22, color=(0, 255, 0)):
    # Mac 기본 한글 폰트
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

    if not Path(font_path).exists():
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_path, font_size)

    # PIL은 RGB 순서
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def run_detection_stage(
    video_path: str | Path,
    prompt: str,
    progress_callback=None,
) -> None:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"입력 영상이 없습니다: {video_path}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OBJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.DETECTION_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(10, "얼굴 탐지 중 (SCRFD + ByteTrack)...")

    run_pass1(
        video_path=video_path,
        output_path=config.TRACK_DB_PATH,
        debug=False,
    )

    if progress_callback:
        progress_callback(40, "얼굴 클러스터링 중...")

    run_pass2(
        track_db_path=config.TRACK_DB_PATH,
        person_db_path=config.PERSON_DB_PATH,
        top_n=config.TOP_N,
    )

    if progress_callback:
        progress_callback(70, "객체/텍스트 탐지 중 (PaddleOCR + Qwen2-VL)...")

    run_pass3(
        video_path=video_path,
        user_prompt=prompt,
        sample_fps=config.OBJECT_SAMPLE_FPS,
        output_dir=config.OBJECT_OUTPUT_DIR,
    )

    create_face_preview(video_path)
    create_object_preview(video_path)

    if progress_callback:
        progress_callback(100, "탐지 완료!")


def load_person_db() -> dict[str, Any]:
    if not config.PERSON_DB_PATH.exists():
        return {}
    return load_json(config.PERSON_DB_PATH)


def load_object_db() -> list[Any]:
    if not config.OBJECT_DB_PATH.exists():
        return []
    return load_json(config.OBJECT_DB_PATH)

def validate_keep_people_removed_from_targets(
    keep_person_ids: set[str],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    선명하게 유지하기로 선택한 인물이
    최종 SAM2 targets에 포함되어 있는지 검사합니다.

    규칙:
    - keep_person_ids에 있는 인물은 sam2_targets.json에 없어야 함
    - 있으면 ERROR
    """

    errors = []
    ok_items = []

    face_targets = [
        target for target in targets
        if target.get("type") == "face"
    ]

    for person_id in keep_person_ids:
        matched_targets = [
            target for target in face_targets
            if person_id in str(target.get("id", ""))
        ]

        if matched_targets:
            errors.append(
                {
                    "person_id": person_id,
                    "matched_targets": matched_targets,
                }
            )
        else:
            ok_items.append(person_id)

    return {
        "ok": len(errors) == 0,
        "ok_items": ok_items,
        "errors": errors,
    }


def run_face_export_and_merge(exclude_person_ids: set[str]) -> list[dict[str, Any]]:
    if not config.TRACK_DB_PATH.exists():
        raise FileNotFoundError("track_db.json이 없습니다. 탐지 단계를 먼저 실행해주세요.")

    if not config.PERSON_DB_PATH.exists():
        raise FileNotFoundError("person_db.json이 없습니다. 얼굴 클러스터링 단계를 먼저 실행해주세요.")

    if not config.OBJECT_DB_PATH.exists():
        raise FileNotFoundError("object_db.json이 없습니다. 탐지 단계를 먼저 실행해주세요.")

    export_for_sam2(
        track_db_path=config.TRACK_DB_PATH,
        person_db_path=config.PERSON_DB_PATH,
        output_path=config.FACE_SAM2_INPUT_PATH,
        exclude_person_ids=exclude_person_ids,
    )

    return run_pass4(
        face_targets_path=config.FACE_SAM2_INPUT_PATH,
        object_db_path=config.OBJECT_DB_PATH,
        output_path=config.SAM2_TARGETS_PATH,
    )


def load_sam2_targets() -> list[dict[str, Any]]:
    if not config.SAM2_TARGETS_PATH.exists():
        return []
    return load_json(config.SAM2_TARGETS_PATH)


def _read_frame(video_path: str | Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return frame


def _save_box_preview(
    video_path: str | Path,
    boxes: list[dict],
    output_path: str | Path,
    title: str,
) -> None:
    if not boxes:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_idx = int(boxes[0].get("frame_idx", boxes[0].get("start_frame", 0)))
    frame = _read_frame(video_path, frame_idx)

    if frame is None:
        return

    for item in boxes:
        box = item.get("bbox") or item.get("box")
        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        label = item.get("label") or item.get("id") or title

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame = put_korean_text(
            frame=frame,
            text=str(label),
            position=(x1, max(20, y1 - 26)),
            font_size=20,
            color=(0, 255, 0),
        )

    cv2.imwrite(str(output_path), frame)


def create_face_preview(video_path: str | Path) -> None:
    track_db = load_json(config.TRACK_DB_PATH) if config.TRACK_DB_PATH.exists() else {}

    boxes = []

    if isinstance(track_db, dict):
        for track_id, track in track_db.items():
            frames = track.get("frames", [])
            bboxes = track.get("bboxes", [])

            if not frames or not bboxes:
                continue

            mid = min(len(frames), len(bboxes)) // 2

            boxes.append(
                {
                    "id": f"track_{track_id}",
                    "frame_idx": frames[mid],
                    "bbox": bboxes[mid],
                    "label": f"face_{track_id}",
                }
            )

            if len(boxes) >= 5:
                break

    _save_box_preview(
        video_path=video_path,
        boxes=boxes,
        output_path=config.FACE_PREVIEW_PATH,
        title="face",
    )


def create_object_preview(video_path: str | Path) -> None:
    object_db = load_object_db()
    boxes = []

    if isinstance(object_db, list):
        for obj in object_db:
            box = obj.get("bbox") or obj.get("box")
            if box is None:
                continue

            boxes.append(
                {
                    "id": obj.get("id", "object"),
                    "frame_idx": obj.get("start_frame", 0),
                    "bbox": box,
                    "label": obj.get("label", "object"),
                }
            )

            if len(boxes) >= 5:
                break

    _save_box_preview(
        video_path=video_path,
        boxes=boxes,
        output_path=config.OBJECT_PREVIEW_PATH,
        title="object",
    )


def run_blur_stage(video_path: str | Path) -> Path:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"입력 영상이 없습니다: {video_path}")

    if not config.SAM2_TARGETS_PATH.exists():
        raise FileNotFoundError("sam2_targets.json이 없습니다. 통합 결과 생성 단계를 먼저 실행해주세요.")

    with open(config.SAM2_TARGETS_PATH, "r", encoding="utf-8") as f:
        targets = json.load(f)

    if not targets:
        raise ValueError("SAM2 target 목록이 비어 있습니다. 블러 처리할 대상이 없습니다.")

    processor = ChunkProcessor(
        model_cfg=config.SAM2_MODEL_CFG,
        checkpoint=str(config.SAM2_CHECKPOINT),
        fps=config.SAM2_FPS,
        chunk_seconds=config.SAM2_CHUNK_SECONDS,
    )

    results = processor.process(str(video_path), targets)

    blur = BlurProcessor(blur_strength=config.BLUR_STRENGTH)
    blur.process(
        str(video_path),
        results,
        targets,
        output_path=str(config.FINAL_OUTPUT_VIDEO_PATH),
    )

    return config.FINAL_OUTPUT_VIDEO_PATH