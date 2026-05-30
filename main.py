# main.py

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

import config
from pipeline.pass1_face_detect_track import run_pass1
from pipeline.pass2_face_cluster import run_pass2
from pipeline.pass3_object_detect import run_pass3
from pipeline.export_for_sam2 import export_for_sam2
from pipeline.run_full_pipeline import run_full_pipeline
from models.face_detector import build_detector
from models.face_tracker import build_tracker
from models.sam2.debug_boxes import draw_debug_boxes


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy Blur Pipeline Entry"
    )

    parser.add_argument(
        "--mode",
        choices=["face", "object", "full", "debug-boxes"],
        default="full",
        help="실행 모드 선택",
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

    parser.add_argument(
        "--top-n",
        type=int,
        default=config.TOP_N,
        help="블러에서 제외할 주요 인물 수",
    )

    parser.add_argument(
        "--blur-main-person",
        action="store_true",
        help="주요 인물도 포함해서 블러 처리",
    )

    parser.add_argument(
        "--skip-pass3",
        action="store_true",
        help="전체 파이프라인에서 객체 탐지 PASS3 생략",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="PASS1 얼굴 탐지 디버그 시각화 활성화",
    )

    parser.add_argument(
        "--targets",
        default=str(config.SAM2_TARGETS_PATH),
        help="debug-boxes 모드에서 사용할 target JSON 경로",
    )

    parser.add_argument(
        "--output",
        default=str(config.FINAL_OUTPUT_VIDEO_PATH),
        help="출력 영상 경로",
    )

    return parser.parse_args()


def run_face_pipeline(
    video_path: Path,
    top_n: int,
    blur_main_person: bool,
    debug: bool,
) -> None:
    logger.info("[MODE: face] 얼굴 파이프라인 실행")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CROPS_DIR.mkdir(parents=True, exist_ok=True)

    detector = build_detector(
        model_pack_name=config.INSIGHTFACE_MODEL_PACK,
        input_size=config.INSIGHTFACE_INPUT_SIZE,
        conf_thresh=config.INSIGHTFACE_CONF_THRESH,
        ctx_id=config.INSIGHTFACE_CTX_ID,
        allowed_modules=config.INSIGHTFACE_ALLOWED_MODULES,
    )

    tracker = build_tracker(
        track_thresh=config.BYTETRACK_TRACK_THRESH,
        high_thresh=config.BYTETRACK_HIGH_THRESH,
        match_thresh=config.BYTETRACK_MATCH_THRESH,
        max_time_lost=config.BYTETRACK_MAX_TIME_LOST,
    )

    logger.info("[PASS1] 얼굴 탐지 + 트래킹 시작")
    pass1_start = time.perf_counter()

    track_db = run_pass1(
        video_path=video_path,
        detector=detector,
        tracker=tracker,
        crops_dir=config.CROPS_DIR,
        output_path=config.TRACK_DB_PATH,
        debug=debug,
    )

    pass1_elapsed = time.perf_counter() - pass1_start

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = total_frames / pass1_elapsed if pass1_elapsed > 0 else 0.0

    logger.info(
        "[PASS1] 완료 | track=%d | %.2fs | %.2f FPS | 저장=%s",
        len(track_db),
        pass1_elapsed,
        fps,
        config.TRACK_DB_PATH,
    )

    logger.info("[PASS2] 얼굴 클러스터링 + Top-N 선정 시작")
    pass2_start = time.perf_counter()

    person_db = run_pass2(
        track_db_path=config.TRACK_DB_PATH,
        person_db_path=config.PERSON_DB_PATH,
        top_n=top_n,
    )

    pass2_elapsed = time.perf_counter() - pass2_start

    logger.info(
        "[PASS2] 완료 | person=%d | %.2fs | 저장=%s",
        len(person_db),
        pass2_elapsed,
        config.PERSON_DB_PATH,
    )

    logger.info("[FACE EXPORT] SAM2 얼굴 입력 생성 시작")

    sam2_inputs = export_for_sam2(
        track_db_path=config.TRACK_DB_PATH,
        person_db_path=config.PERSON_DB_PATH,
        output_path=config.FACE_SAM2_INPUT_PATH,
        blur_main_person=blur_main_person,
    )

    logger.info(
        "[FACE EXPORT] 완료 | target=%d | 저장=%s",
        len(sam2_inputs),
        config.FACE_SAM2_INPUT_PATH,
    )


def main() -> None:
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {video_path}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OBJECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "face":
        run_face_pipeline(
            video_path=video_path,
            top_n=args.top_n,
            blur_main_person=args.blur_main_person,
            debug=args.debug,
        )

    elif args.mode == "object":
        logger.info("[MODE: object] PASS3 객체/텍스트 탐지만 실행")

        run_pass3(
            video_path=video_path,
            user_prompt=args.prompt,
            sample_fps=args.sample_fps,
            output_dir=config.OBJECT_OUTPUT_DIR,
        )

    elif args.mode == "full":
        logger.info("[MODE: full] 전체 파이프라인 실행")

        run_full_pipeline(
            video_path=video_path,
            prompt=args.prompt,
            output_video=args.output,
            sample_fps=args.sample_fps,
            top_n=args.top_n,
            blur_main_person=args.blur_main_person,
            skip_pass3=args.skip_pass3,
            debug=args.debug,
        )

    elif args.mode == "debug-boxes":
        logger.info("[MODE: debug-boxes] target box 시각화")

        draw_debug_boxes(
            video_path=video_path,
            targets_path=Path(args.targets).expanduser().resolve(),
            output_path=Path(args.output).expanduser().resolve(),
        )


if __name__ == "__main__":
    main()