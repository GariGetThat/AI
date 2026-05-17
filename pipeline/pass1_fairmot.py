# pipeline/pass1_fairmot.py

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

import cv2

import config

logger = logging.getLogger(__name__)


def run_fairmot(
    video_path: str | Path,
    fairmot_dir: str | Path = config.FAIRMOT_DIR,
    output_dir: str | Path = config.FAIRMOT_OUTPUT_DIR,
    weights_path: str | Path = config.FAIRMOT_WEIGHTS,
) -> Dict[str, Any]:
    """
    FairMOT 비교 실험용 실행 함수.

    outputs/FairMOT/
      ├── fairmot_results.txt
      └── fairmot_summary.json

    주의:
    - FairMOT repo와 weight가 external/FairMOT 아래 준비되어 있어야 함.
    - FairMOT의 실제 demo 명령어는 설치한 repo 구조에 따라 수정 가능.
    """

    video_path = Path(video_path)
    fairmot_dir = Path(fairmot_dir)
    output_dir = Path(output_dir)
    weights_path = Path(weights_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not fairmot_dir.exists():
        raise FileNotFoundError(f"FairMOT repo가 없습니다: {fairmot_dir}")

    if not weights_path.exists():
        raise FileNotFoundError(f"FairMOT weight가 없습니다: {weights_path}")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    logger.info("=== FairMOT 시작 ===")
    logger.info("video: %s", video_path)
    logger.info("FairMOT dir: %s", fairmot_dir)
    logger.info("weights: %s", weights_path)

    start_time = time.perf_counter()

    # FairMOT repo의 demo.py 명령어 예시
    # 실제 repo에 따라 --input-video, --load_model 옵션명은 수정 필요할 수 있음
    cmd = [
        "python",
        "src/demo.py",
        "mot",
        "--load_model",
        str(weights_path),
        "--input-video",
        str(video_path),
        "--output-root",
        str(output_dir),
    ]

    logger.info("FairMOT command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(fairmot_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    elapsed = time.perf_counter() - start_time
    proc_fps = total_frames / elapsed if elapsed > 0 else 0.0

    log_path = output_dir / "fairmot_run.log"
    log_path.write_text(result.stdout, encoding="utf-8")

    summary = {
        "method": "FairMOT",
        "video": str(video_path),
        "total_frames": total_frames,
        "video_fps": video_fps,
        "elapsed_sec": elapsed,
        "processing_fps": proc_fps,
        "return_code": result.returncode,
        "log_path": str(log_path),
        "output_dir": str(output_dir),
    }

    summary_path = output_dir / "fairmot_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "=== FairMOT 완료 | 총 처리 시간 = %.2f sec | 평균 FPS = %.2f | 저장: %s ===",
        elapsed,
        proc_fps,
        output_dir,
    )

    if result.returncode != 0:
        logger.warning("FairMOT 실행 중 오류 발생. 로그 확인: %s", log_path)

    return summary