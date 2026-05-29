import argparse
import gc
import sys
import traceback
from pathlib import Path
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import torch
from models.object_detector import PrivacyReasoningEngine
import os
import cv2
import numpy as np
from PIL import Image
import config

def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PASS3 Object Detection"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="입력 영상 경로",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=config.OBJECT_DEFAULT_PROMPT,
        help="사용자 자연어 프롬프트",
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=config.OBJECT_SAMPLE_FPS,
        help="샘플링 FPS",
    )

    parser.add_argument(
        "--text-detector-lang",
        type=str,
        default=config.OBJECT_TEXT_DETECTOR_LANG,
        help="PaddleOCR language option",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.OBJECT_OUTPUT_DIR),
        help="object 결과 저장 디렉토리",
    )

    return parser.parse_args()

def run_pass3(
    video_path,
    user_prompt,
    sample_fps,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_crop_dir = output_dir / "debug_crops"
    json_output_path = config.OBJECT_DB_PATH

    engine = PrivacyReasoningEngine(
        qwen_model_name=config.OBJECT_QWEN_MODEL_NAME,
        device=config.DEVICE,
        crop_margin_ratio=config.OBJECT_CROP_MARGIN_RATIO,
        max_new_tokens_reason=config.OBJECT_MAX_NEW_TOKENS_REASON,
        verbose=True,
        save_verified_crops=True,
        debug_crop_dir=str(debug_crop_dir),
        track_iou_threshold=config.OBJECT_TRACK_IOU_THRESHOLD,
        min_text_box_area=config.OBJECT_MIN_TEXT_BOX_AREA,
        max_text_box_area_ratio=config.OBJECT_MAX_TEXT_BOX_AREA_RATIO,
        min_text_region_width=config.OBJECT_MIN_TEXT_REGION_WIDTH,
        min_text_region_height=config.OBJECT_MIN_TEXT_REGION_HEIGHT,
        text_detector_lang=config.OBJECT_TEXT_DETECTOR_LANG,
        min_rec_score=config.OBJECT_MIN_REC_SCORE,
        min_group_items=config.OBJECT_MIN_GROUP_ITEMS,
        group_x_gap_ratio=config.OBJECT_GROUP_X_GAP_RATIO,
        group_y_gap_ratio=config.OBJECT_GROUP_Y_GAP_RATIO,
        min_qwen_crop_size=config.OBJECT_MIN_QWEN_CROP_SIZE,
    )

    tracks = engine.process_video(
        video_path=str(video_path),
        user_prompt=user_prompt,
        sample_fps=sample_fps,
    )

    payload = engine.export_to_json(
        output_json_path=str(json_output_path),
        tracks=tracks,
    )

    return payload

if __name__ == "__main__":
    print("[PASS3] object detection 시작", flush=True)

    try:
        project_root = Path(__file__).resolve().parents[1]
        args = parse_args(project_root)

        video_path = Path(args.video).expanduser().resolve()

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = config.OBJECT_OUTPUT_DIR

        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일이 없음: {video_path}")

        run_pass3(
            video_path=video_path,
            user_prompt=args.prompt,
            sample_fps=args.sample_fps,
            output_dir=output_dir,
        )

        print("[PASS3] object detection 완료", flush=True)

    except Exception as e:
        print(f"[Fatal] PASS3 실행 중 예외 발생: {e}", flush=True)
        traceback.print_exc()
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()