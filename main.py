# main.py

"""
실행 엔트리포인트.

사용 예:
    python main.py --video input.mp4 --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

import config
from models.face_detector import build_detector
from models.face_tracker import build_tracker
from AI.pipeline.pass1_face_detect_track import run_pass1
from AI.pipeline.pass2_face_cluster import run_pass2
from AI.pipeline.pass4_merge_targets import export_for_sam2


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Privacy Guard - 얼굴 탐지 및 추적 파이프라인"
    )

    p.add_argument(
        "--video",
        required=True,
        help="입력 영상 경로",
    )

    p.add_argument(
        "--debug",
        action="store_true",
        help="디버그 시각화 활성화",
    )

    return p.parse_args()


def main() -> None:

    args = parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        logger.error("영상 파일을 찾을 수 없습니다: %s", video_path)
        sys.exit(1)

    # ── detector ───────────────────────────────────────
    detector = build_detector(
        model_pack_name=config.INSIGHTFACE_MODEL_PACK,
        input_size=config.INSIGHTFACE_INPUT_SIZE,
        conf_thresh=config.INSIGHTFACE_CONF_THRESH,
        ctx_id=config.INSIGHTFACE_CTX_ID,
        allowed_modules=config.INSIGHTFACE_ALLOWED_MODULES,
    )

    # ── tracker ────────────────────────────────────────
    tracker = build_tracker(
        track_thresh=config.BYTETRACK_TRACK_THRESH,
        high_thresh=config.BYTETRACK_HIGH_THRESH,
        match_thresh=config.BYTETRACK_MATCH_THRESH,
        max_time_lost=config.BYTETRACK_MAX_TIME_LOST,
    )

    # ── PASS1 실행 ─────────────────────────────────────
    start_time = time.perf_counter()

    logger.info("=== PASS1 시작 ===")

    track_db = run_pass1(
        video_path=video_path,
        detector=detector,
        tracker=tracker,
        debug=args.debug,
    )

    elapsed = time.perf_counter() - start_time

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = total_frames / elapsed if elapsed > 0 else 0.0

    logger.info(
        "=== PASS1 완료 | track 수=%d | %.2f sec | %.2f FPS ===",
        len(track_db),
        elapsed,
        fps,
    )

    # ── PASS2 실행 ─────────────────────────────────────
    logger.info("=== PASS2 시작 ===")

    pass2_start = time.perf_counter()

    person_db = run_pass2(
        top_n=config.TOP_N,
    )

    pass2_elapsed = time.perf_counter() - pass2_start

    logger.info(
        "=== PASS2 완료 | person 수=%d | %.2f sec ===",
        len(person_db),
        pass2_elapsed,
    )

    # ── SAM2 export ────────────────────────────────────
    logger.info("=== SAM2 export 시작 ===")

    sam2_inputs = export_for_sam2()

    logger.info(
        "=== SAM2 export 완료 | 대상 수=%d ===",
        len(sam2_inputs),
    )

    logger.info("=== 전체 파이프라인 완료 ===")


if __name__ == "__main__":
    main()