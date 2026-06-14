# models/face/debug_person_tracks.py

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import cv2

import config


def _load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_track_to_person_map(person_db: dict) -> dict[int, str]:
    track_to_person = {}

    for person_id, person in person_db.items():
        for track_id in person.get("track_ids", []):
            track_to_person[int(track_id)] = person_id

    return track_to_person


def _format_person_name(person_id: str | None) -> str:
    if not person_id:
        return "unknown"

    if person_id.startswith("person_"):
        number = person_id.replace("person_", "")

        if number.isdigit():
            return f"person {int(number) + 1}"

    return person_id


def draw_person_track_debug_video(
    video_path: str | Path,
    track_db_path: str | Path = config.TRACK_DB_PATH,
    person_db_path: str | Path = config.PERSON_DB_PATH,
    output_path: str | Path = config.OUTPUT_DIR / "debug_person_tracks.mp4",
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    track_db = _load_json(track_db_path)
    person_db = _load_json(person_db_path)
    track_to_person = _build_track_to_person_map(person_db)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    # frame_idx -> 그 프레임에 그릴 bbox 목록
    frame_items: dict[int, list[dict]] = {}

    for track_id_str, track in track_db.items():
        track_id = int(track_id_str)
        person_id = track_to_person.get(track_id)

        frames = track.get("frames", [])
        bboxes = track.get("bboxes", [])

        for frame_idx, bbox in zip(frames, bboxes):
            frame_items.setdefault(int(frame_idx), []).append(
                {
                    "track_id": track_id,
                    "person_id": person_id,
                    "bbox": bbox,
                }
            )

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        items = frame_items.get(frame_idx, [])

        for item in items:
            x1, y1, x2, y2 = map(int, item["bbox"])
            track_id = item["track_id"]
            person_id = item["person_id"]

            label = f"{_format_person_name(person_id)} | track {track_id}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "debug_person_tracks.mp4"))
    args = parser.parse_args()

    output = draw_person_track_debug_video(
        video_path=args.video,
        output_path=args.output,
    )

    print(f"saved: {output}")