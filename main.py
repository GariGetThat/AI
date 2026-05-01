# main.py
"""
실행 엔트리포인트.

사용 예:
    python main.py --video input.mp4 --top-n 2 --debug
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from pipeline.pass1_detect_track import run_pass1

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Privacy Guard - 영상 프라이버시 보호 AI 편집 도구")
    p.add_argument("--video",   required=True, help="입력 영상 경로")
    p.add_argument("--top-n",   type=int, default=config.TOP_N,
                   help="블러 제외할 주요 인물 수 (기본: %(default)s)")
    p.add_argument("--debug",   action="store_true", help="디버그 시각화 활성화")
    p.add_argument("--dummy",   action="store_true",
                   help="실제 모델 없이 더미로 파이프라인 테스트")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("영상 파일을 찾을 수 없습니다: %s", video_path)
        sys.exit(1)

    # ── 모델 빌드 ────────────────────────────────────────
    if args.dummy:
        logger.info("더미 모드 활성화 → 실제 모델 로드 생략")
        from models.face_detector import DummyFaceDetector
        from models.face_tracker  import DummyFaceTracker
        detector = DummyFaceDetector()
        tracker  = DummyFaceTracker()
    else:
        from models.face_detector import build_detector
        from models.face_tracker  import build_tracker
        detector = build_detector(config.SCRFD_MODEL_PATH)
        tracker  = build_tracker(use_real=False)   # ByteTrack 준비되면 True

    # ── PASS 1 ───────────────────────────────────────────
    logger.info("=== PASS 1 시작 ===")
    track_db = run_pass1(
        video_path=video_path,
        detector=detector,
        tracker=tracker,
        debug=args.debug,
    )
    logger.info("=== PASS 1 완료 : track 수 = %d ===", len(track_db))

    # TODO: PASS 2 (ArcFace + DBSCAN)
    # TODO: export_for_sam2


if __name__ == "__main__":
    main()