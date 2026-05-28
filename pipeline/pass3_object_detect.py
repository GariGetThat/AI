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
    parser = argparse.ArgumentParser(description="PrivacyReasoningEngine grouped-object video runner")
    parser.add_argument(
        "--video",
        type=str,
        default=str(project_root / "test_input.mp4"),
        help="테스트할 비디오 경로",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="내 프라이버시가 유출될 만한 것들을 가려줘.",
        help="사용자 자연어 프롬프트",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="샘플링 FPS",
    )
    parser.add_argument(
        "--text-detector-lang",
        type=str,
        default="korean",
        help="PaddleOCR language option",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="결과물 저장 디렉토리 (기본값: project_root)",
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
    json_output_path = output_dir / "object_db.json"

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
    print("[Main] 프로그램 시작", flush=True)
    print(f"[Main] Python executable = {sys.executable}", flush=True)
    print(f"[Main] torch version = {torch.__version__}", flush=True)
    print(f"[Main] CUDA available = {torch.cuda.is_available()}", flush=True)

    if torch.cuda.is_available():
        print(f"[Main] CUDA device count = {torch.cuda.device_count()}", flush=True)
        print(f"[Main] current device = {torch.cuda.current_device()}", flush=True)
        print(f"[Main] device name = {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)

    try:
        project_root = Path(__file__).resolve().parent.parent
        args = parse_args(project_root)

        video_path = Path(args.video).expanduser().resolve()
        # output_dir 설정
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = project_root

        debug_crop_dir = output_dir / "debug_crops"
        from datetime import datetime
        _timestamp = datetime.now().strftime("%m%d_%H%M")
        debug_video_fullfps_path = output_dir / f"debug_bbox_output_fullfps_{_timestamp}.mp4"
        debug_video_sampled_path = output_dir / f"debug_bbox_output_sampled_{_timestamp}.mp4"
        json_output_path = output_dir / "privacy_runtime_bar.json"
        user_prompt = args.prompt
        sample_fps = args.sample_fps
        text_detector_lang = args.text_detector_lang

        print(f"[Main] project_root = {project_root}", flush=True)
        print(f"[Main] video_path = {video_path}", flush=True)
        print(f"[Main] debug_crop_dir = {debug_crop_dir}", flush=True)
        print(f"[Main] debug_video_fullfps_path = {debug_video_fullfps_path}", flush=True)
        print(f"[Main] debug_video_sampled_path = {debug_video_sampled_path}", flush=True)
        print(f"[Main] json_output_path = {json_output_path}", flush=True)
        print(f"[Main] user_prompt = {user_prompt}", flush=True)
        print(f"[Main] sample_fps = {sample_fps}", flush=True)
        print(f"[Main] text_detector_lang = {text_detector_lang}", flush=True)

        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일이 없음: {video_path}")

        engine = PrivacyReasoningEngine(
            qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
            crop_margin_ratio=0.35,
            max_new_tokens_reason=48,
            verbose=True,
            save_verified_crops=True,
            debug_crop_dir=str(debug_crop_dir),
            track_iou_threshold=0.30,
            min_text_box_area=16,
            max_text_box_area_ratio=0.75,
            min_text_region_width=3,
            min_text_region_height=3,
            text_detector_lang=text_detector_lang,
            min_rec_score=0.30,
            min_group_items=1,
            group_x_gap_ratio=1.6,
            group_y_gap_ratio=1.0,
            min_qwen_crop_size=56,
        )

        tracks = engine.process_video(
            video_path=str(video_path),
            user_prompt=user_prompt,
            sample_fps=sample_fps,
        )

        engine.export_to_json(
            output_json_path=str(json_output_path),
            tracks=tracks,
        )

        engine.write_debug_video_fullfps(
            input_video_path=str(video_path),
            output_video_path=str(debug_video_fullfps_path),
        )

        engine.write_debug_video_sampled(
            input_video_path=str(video_path),
            output_video_path=str(debug_video_sampled_path),
            sample_fps=sample_fps,
        )

        print("[Main] 비디오 분석 종료", flush=True)

        if len(engine.last_verified_detections) == 0:
            print("발견된 항목 없음", flush=True)
        else:
            print(f"[Main] 총 검증 통과 수 = {len(engine.last_verified_detections)}", flush=True)
            for det in engine.last_verified_detections:
                print(
                    [
                        det.frame_index,
                        det.x1,
                        det.y1,
                        det.x2,
                        det.y2,
                        det.label,
                        det.track_id,
                        det.qwen_visible_text,
                    ],
                    flush=True,
                )

        print(f"[Main] 총 트랙 수 = {len(tracks)}", flush=True)
        for track in tracks:
            print(
                {
                    "id": track.object_id,
                    "label": track.label,
                    "start_frame": track.start_frame,
                    "end_frame": track.end_frame,
                    "box": track.representative_box(),
                    "visible_text": track.representative_visible_text(),
                },
                flush=True,
            )

    except Exception as e:
        print(f"[Fatal] 메인 실행 중 예외 발생: {e}", flush=True)
        traceback.print_exc()
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[Main] 프로그램 종료", flush=True)
        