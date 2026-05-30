# models/sam2/debug_boxes.py

import json
import cv2
from pathlib import Path


def draw_debug_boxes(
    video_path: str | Path,
    targets_path: str | Path,
    output_path: str | Path,
) -> None:
    video_path = Path(video_path)
    targets_path = Path(targets_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    with open(targets_path, "r", encoding="utf-8") as f:
        targets = json.load(f)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for target in targets:
            if target["start_frame"] <= frame_idx <= target["end_frame"]:
                x1, y1, x2, y2 = map(int, target["box"])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    target["id"],
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"debug box 영상 저장 완료: {output_path}")