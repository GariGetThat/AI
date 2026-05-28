# main.py

from __future__ import annotations

import argparse
from pathlib import Path

import config
from pipeline.pass3_object_detect import run_pass3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy Blur Pipeline - Object Detection Test Entry"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="입력 영상 경로",
    )

    parser.add_argument(
        "--prompt",
        default=config.OBJECT_DEFAULT_PROMPT,
        help="객체 탐지용 자연어 프롬프트",
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=config.OBJECT_SAMPLE_FPS,
        help="객체 탐지 샘플링 FPS",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {video_path}")

    config.OBJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_pass3(
        video_path=video_path,
        user_prompt=args.prompt,
        sample_fps=args.sample_fps,
        output_dir=config.OBJECT_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()