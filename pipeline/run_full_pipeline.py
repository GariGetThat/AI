# pipeline/run_full_pipeline.py

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from pipeline.pass1_face_detect_track import run_pass1
from pipeline.pass2_face_cluster import run_pass2
from pipeline.pass3_object_detect import run_pass3
from pipeline.export_for_sam2 import export_for_sam2
from pipeline.pass4_merge_targets import run_pass4

from models.sam2.chunk_processor import ChunkProcessor
from models.sam2.blur_processor import BlurProcessor


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full privacy blur pipeline runner")

    parser.add_argument("--video", type=str, required=True, help="입력 영상 경로")
    parser.add_argument(
        "--prompt",
        type=str,
        default="내 프라이버시가 유출될 만한 것들을 가려줘.",
        help="객체/텍스트 탐지용 자연어 프롬프트",
    )
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--output-video", type=str, default="output_video.mp4")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=config.TOP_N)
    parser.add_argument("--blur-main-person", action="store_true")
    parser.add_argument("--skip-pass3", action="store_true", help="객체 탐지 PASS3 생략")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def run_full_pipeline(
    video_path: str | Path,
    prompt: str,
    output_dir: str | Path,
    output_video: str | Path,
    sample_fps: float = 1.0,
    top_n: int = config.TOP_N,
    blur_main_person: bool = False,
    skip_pass3: bool = False,
    debug: bool = False,
) -> Path:
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"입력 영상이 없습니다: {video_path}")

    track_db_path = output_dir / "track_db.json"
    person_db_path = output_dir / "person_db.json"
    face_targets_path = output_dir / "face_sam2_input.json"
    object_db_path = output_dir / "object_db.json"
    final_targets_path = output_dir / "sam2_targets.json"

    output_video_path = Path(output_video)

    if not output_video_path.is_absolute():
        output_video_path = output_dir / output_video_path

    logger.info("FULL PIPELINE 시작")
    logger.info("video=%s", video_path)
    logger.info("output_dir=%s", output_dir)

    logger.info("[PASS1] 얼굴 탐지 + 트래킹 시작")
    run_pass1(
        video_path=video_path,
        output_path=track_db_path,
        debug=debug,
    )

    logger.info("[PASS2] 얼굴 클러스터링 + Top-N 선정 시작")
    run_pass2(
        track_db_path=track_db_path,
        person_db_path=person_db_path,
        top_n=top_n,
    )

    logger.info("[FACE EXPORT] SAM2 얼굴 입력 생성")
    export_for_sam2(
        track_db_path=track_db_path,
        person_db_path=person_db_path,
        output_path=face_targets_path,
        blur_main_person=blur_main_person,
    )

    if not skip_pass3:
        logger.info("[PASS3] 객체/텍스트 탐지 시작")
        run_pass3(
            video_path=video_path,
            user_prompt=prompt,
            sample_fps=sample_fps,
            output_dir=output_dir,
        )
    else:
        logger.info("[PASS3] 생략됨")

    logger.info("[PASS4] face target + object target 병합")
    run_pass4(
        face_targets_path=face_targets_path,
        object_db_path=object_db_path,
        output_path=final_targets_path,
    )

    logger.info("[PASS5] SAM2 마스크 생성 시작")
    processor = ChunkProcessor(
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint="checkpoints/sam2.1_hiera_large.pt",
    )

    targets = final_targets_path
    import json

    with open(targets, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    results = processor.process(str(video_path), target_data)

    logger.info("[PASS5] 블러 처리 시작")
    blur = BlurProcessor(blur_strength=31)
    blur.process(
        str(video_path),
        results,
        target_data,
        output_path=str(output_video_path),
    )

    logger.info("FULL PIPELINE 완료 | output=%s", output_video_path)

    return output_video_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s : %(message)s",
    )

    args = parse_args()

    run_full_pipeline(
        video_path=args.video,
        prompt=args.prompt,
        output_dir=args.output_dir,
        output_video=args.output_video,
        sample_fps=args.sample_fps,
        top_n=args.top_n,
        blur_main_person=args.blur_main_person,
        skip_pass3=args.skip_pass3,
        debug=args.debug,
    )