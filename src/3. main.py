import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import re
import gc
import sys
import json
import argparse
import traceback
import faulthandler
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

faulthandler.enable()

import cv2
import numpy as np
from PIL import Image

import torch

from transformers import AutoModelForVision2Seq, AutoProcessor
from paddleocr import PaddleOCR


@dataclass
class OCRTextItem:
    frame_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    score: float

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class TextGroupCandidate:
    frame_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    merged_text: str
    avg_score: float
    item_count: int
    items: List[OCRTextItem] = field(default_factory=list)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class VerifiedDetection:
    frame_index: int
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    detector_phrase: str
    detector_score: float
    qwen_visible_text: str = ""
    qwen_reason: str = ""
    track_id: Optional[str] = None

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class TrackState:
    object_id: str
    label: str
    start_frame: int
    end_frame: int
    last_frame: int
    last_box: Tuple[int, int, int, int]
    history_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    visible_text_samples: List[str] = field(default_factory=list)
    reason_samples: List[str] = field(default_factory=list)
    hit_count: int = 1

    def update(self, detection: VerifiedDetection) -> None:
        self.end_frame = detection.frame_index
        self.last_frame = detection.frame_index
        self.last_box = detection.box
        self.history_boxes.append(detection.box)
        if detection.qwen_visible_text:
            self.visible_text_samples.append(detection.qwen_visible_text)
        if detection.qwen_reason:
            self.reason_samples.append(detection.qwen_reason)
        self.hit_count += 1

    def representative_box(self) -> List[int]:
        if not self.history_boxes:
            return list(self.last_box)

        arr = np.array(self.history_boxes, dtype=np.float32)
        med = np.median(arr, axis=0)
        return [int(round(v)) for v in med.tolist()]

    def representative_visible_text(self) -> str:
        for text in reversed(self.visible_text_samples):
            if text and text.upper() != "NONE":
                return text
        return ""


class PrivacyReasoningEngine:
    LABEL_DISPLAY_MAP: Dict[str, str] = {
        "license plate": "번호판",
        "driver's license": "운전면허증",
        "credit card": "신용카드",
        "receipt": "영수증",
        "address document": "주소 문서",
        "other private text": "기타 개인정보 텍스트",
        "other": "기타",
    }

    WEAK_EVIDENCE_WORDS = {
        "card",
        "paper",
        "screen",
        "document",
        "printed text",
        "text",
        "visible text",
        "license plate",
        "label",
        "sign",
        "none",
    }

    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        crop_margin_ratio: float = 0.35,
        max_new_tokens_reason: int = 48,
        verbose: bool = True,
        save_verified_crops: bool = True,
        debug_crop_dir: str = "debug_crops",
        track_iou_threshold: float = 0.30,
        min_text_box_area: int = 16,
        max_text_box_area_ratio: float = 0.75,
        min_text_region_width: int = 3,
        min_text_region_height: int = 3,
        text_detector_lang: str = "korean",
        min_rec_score: float = 0.30,
        min_group_items: int = 1,
        group_x_gap_ratio: float = 1.6,
        group_y_gap_ratio: float = 1.0,
        min_qwen_crop_size: int = 56,
    ) -> None:
        self.verbose = bool(verbose)
        self.save_verified_crops = bool(save_verified_crops)
        self.debug_crop_dir = Path(debug_crop_dir)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.max_new_tokens_reason = int(max_new_tokens_reason)
        self.track_iou_threshold = float(track_iou_threshold)

        self.min_text_box_area = int(min_text_box_area)
        self.max_text_box_area_ratio = float(max_text_box_area_ratio)
        self.min_text_region_width = int(min_text_region_width)
        self.min_text_region_height = int(min_text_region_height)
        self.min_rec_score = float(min_rec_score)
        self.min_group_items = int(min_group_items)
        self.group_x_gap_ratio = float(group_x_gap_ratio)
        self.group_y_gap_ratio = float(group_y_gap_ratio)
        self.min_qwen_crop_size = int(min_qwen_crop_size)

        if self.save_verified_crops:
            self.debug_crop_dir.mkdir(parents=True, exist_ok=True)

        self._next_track_index = 1
        self.active_tracks: Dict[str, TrackState] = {}
        self.finished_tracks: List[TrackState] = []
        self.last_verified_detections: List[VerifiedDetection] = []
        self.last_tracks: List[TrackState] = []
        self.last_json_payload: List[Dict] = []

        self._log("엔진 초기화 시작")
        self._log(f"device = {self.device}")
        self._log(f"crop_margin_ratio = {self.crop_margin_ratio}")
        self._log(f"track_iou_threshold = {self.track_iou_threshold}")
        self._log(f"text_detector_lang = {text_detector_lang}")
        self._log(f"min_text_box_area = {self.min_text_box_area}")
        self._log(f"min_text_region_width = {self.min_text_region_width}")
        self._log(f"min_text_region_height = {self.min_text_region_height}")
        self._log(f"min_rec_score = {self.min_rec_score}")
        self._log(f"min_qwen_crop_size = {self.min_qwen_crop_size}")

        self.qwen_processor = AutoProcessor.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
        )
        self._log("Qwen2-VL Processor 로딩 완료")

        self.qwen_model = AutoModelForVision2Seq.from_pretrained(
            qwen_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.qwen_model.eval()
        self._log("Qwen2-VL Model 로딩 완료")

        if hasattr(self.qwen_model, "generation_config") and self.qwen_model.generation_config is not None:
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(self.qwen_model.generation_config, attr):
                    setattr(self.qwen_model.generation_config, attr, None)
            if hasattr(self.qwen_model.generation_config, "do_sample"):
                self.qwen_model.generation_config.do_sample = False

        self.text_detector = PaddleOCR(
            lang=text_detector_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self._log("PaddleOCR detector+recognizer 로딩 완료")

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Info] {message}", flush=True)

    def _log_group_found(self, frame_index: int, count: int) -> None:
        print(f"[Frame {frame_index}] Group 후보 {count}개 발견", flush=True)

    def _log_verified(self, detection: VerifiedDetection) -> None:
        print(
            f"[Frame {detection.frame_index}] 검증 통과: {detection.label} "
            f"| box=({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2}) "
            f"| id={detection.track_id}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Prompt helpers
    # -------------------------------------------------------------------------
    def build_privacy_reason_prompt(
        self,
        user_prompt: str,
        visible_text: str,
    ) -> str:
        return (
        "You are deciding whether a text group and its surrounding object should be privacy-blurred.\n"
        f"User request: {user_prompt}\n"
        f"Visible text from the grouped region:\n{visible_text}\n"
        "Decide based on the visible text and obvious visible context only.\n"
        "If this grouped region likely belongs to a privacy-sensitive object such as a receipt, waybill(delivery label with name/address/phone number), address label, card, ID document, license plate, or other personal text-bearing object, answer yes. Road signs, subtitles, and brand logos are NOT privacy-sensitive.\n"
        "Short noisy fragments such as a single digit or random short token alone are not enough unless the overall grouped region strongly suggests a sensitive object.\n"
        "If any part of the grouped region likely contains privacy-sensitive information, treat the whole grouped object as blur-worthy.\n"
        "You MUST return exactly three lines in this order, do NOT skip or merge any line:\n"
        "Decision: yes or no\n"
        "Label: receipt | waybill | credit card | driver's license | license plate | road sign | address document | other private text | other\n"
        "Reason: <one short sentence>\n"
        "Example output:\n"
        "Decision: yes\n"
        "Label: waybill\n"
        "Reason: Contains name, address and phone number typical of a delivery label.\n"
    )

    # -------------------------------------------------------------------------
    # Qwen helpers
    # -------------------------------------------------------------------------
    def ask_qwen_reason(
        self,
        image_pil: Image.Image,
        prompt_text: str,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text_prompt = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.qwen_processor(
            text=[text_prompt],
            images=[image_pil],
            padding=True,
            return_tensors="pt",
        )

        for key, value in inputs.items():
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)

        with torch.inference_mode():
            output_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_len:]

        answer = self.qwen_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return answer.strip()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process_video(
        self,
        video_path: str,
        user_prompt: str = "내 프라이버시가 유출될 만한 것들을 가려줘.",
        sample_fps: float = 1.0,
    ) -> List[TrackState]:
        print("[Start] process_video 시작", flush=True)
        print(f"[Start] video_path = {video_path}", flush=True)
        print(f"[Start] user_prompt = {user_prompt}", flush=True)
        print(f"[Start] sample_fps = {sample_fps}", flush=True)

        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0.")

        self.active_tracks = {}
        self.finished_tracks = []
        self.last_verified_detections = []
        self.last_tracks = []
        self.last_json_payload = []
        self._next_track_index = 1

        cap = cv2.VideoCapture(video_path)
        print(f"[Start] VideoCapture isOpened = {cap.isOpened()}", flush=True)

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if native_fps <= 0 or not np.isfinite(native_fps):
            native_fps = 30.0

        sample_interval_frames = max(1, int(round(native_fps / sample_fps)))
        max_track_gap_frames = max(sample_interval_frames * 2, sample_interval_frames + 1)

        print(f"[Start] total_frames = {total_frames}", flush=True)
        print(f"[Start] native_fps = {native_fps}", flush=True)
        print(f"[Start] resolution = {width}x{height}", flush=True)
        print(f"[Start] sample_interval_frames = {sample_interval_frames}", flush=True)
        print(f"[Start] max_track_gap_frames = {max_track_gap_frames}", flush=True)

        frame_index = 0
        sampled_count = 0

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_index % sample_interval_frames != 0:
                    frame_index += 1
                    continue

                sampled_count += 1

                try:
                    groups = self.detect_grouped_text_candidates(
                        frame_bgr=frame_bgr,
                        frame_index=frame_index,
                    )
                except Exception as e:
                    print(f"[Error] Frame {frame_index} detect/group 실패: {e}", flush=True)
                    traceback.print_exc()
                    frame_index += 1
                    continue

                if len(groups) > 0:
                    self._log_group_found(frame_index, len(groups))

                frame_verified: List[VerifiedDetection] = []

                for group in groups:
                    try:
                        verified_detection = self.verify_group_reason_only(
                            frame_bgr=frame_bgr,
                            group=group,
                            user_prompt=user_prompt,
                        )
                    except Exception as e:
                        print(f"[Error] Frame {frame_index} reason 실패: {e}", flush=True)
                        traceback.print_exc()
                        continue

                    if verified_detection is not None:
                        frame_verified.append(verified_detection)

                self.update_tracks_for_frame(
                    detections=frame_verified,
                    frame_index=frame_index,
                    max_track_gap_frames=max_track_gap_frames,
                )

                for det in frame_verified:
                    self.last_verified_detections.append(det)
                    self._log_verified(det)

                frame_index += 1

        finally:
            cap.release()

        self.finalize_all_tracks()
        self.last_tracks = list(self.finished_tracks)

        print(f"[Done] 샘플링된 프레임 수 = {sampled_count}", flush=True)
        print(f"[Done] 총 검증 통과 수 = {len(self.last_verified_detections)}", flush=True)
        print(f"[Done] 총 트랙 수 = {len(self.last_tracks)}", flush=True)

        return self.last_tracks

    # -------------------------------------------------------------------------
    # Stage 1: PaddleOCR detection + recognition + grouping
    # -------------------------------------------------------------------------
    def detect_grouped_text_candidates(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
    ) -> List[TextGroupCandidate]:
        height, width = frame_bgr.shape[:2]
        frame_area = float(width * height)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_results = self.text_detector.predict(frame_rgb)

        items = self.extract_text_items_from_predict_results(raw_results, frame_index)

        filtered_items: List[OCRTextItem] = []
        for item in items:
            x1, y1, x2, y2 = item.box
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            area = float(box_w * box_h)

            if area < self.min_text_box_area:
                continue
            if area > frame_area * self.max_text_box_area_ratio:
                continue
            if box_w < self.min_text_region_width or box_h < self.min_text_region_height:
                continue
            if item.score < self.min_rec_score:
                continue
            if self.is_missing_or_weak_visible_text(item.text):
                continue
            if self.visible_text_is_gibberish(item.text):
                continue

            filtered_items.append(item)

        print(f"[Debug][Frame {frame_index}] filtered text items = {len(filtered_items)}", flush=True)

        if len(filtered_items) == 0:
            return []

        groups = self.group_text_items(filtered_items)

        candidates: List[TextGroupCandidate] = []
        for group_items in groups:
            if len(group_items) < self.min_group_items:
                continue

            gx1 = min(item.x1 for item in group_items)
            gy1 = min(item.y1 for item in group_items)
            gx2 = max(item.x2 for item in group_items)
            gy2 = max(item.y2 for item in group_items)

            merged_text = self.build_group_text(group_items)
            avg_score = float(sum(item.score for item in group_items) / max(1, len(group_items)))

            if self.group_text_is_too_weak(merged_text, len(group_items)):
                continue

            candidates.append(
                TextGroupCandidate(
                    frame_index=frame_index,
                    x1=gx1,
                    y1=gy1,
                    x2=gx2,
                    y2=gy2,
                    merged_text=merged_text,
                    avg_score=avg_score,
                    item_count=len(group_items),
                    items=group_items,
                )
            )

        print(f"[Debug][Frame {frame_index}] group candidates = {len(candidates)}", flush=True)
        for i, cand in enumerate(candidates[:10]):
            print(
                f"[Debug][Frame {frame_index}] group[{i}] items={cand.item_count} score={cand.avg_score:.3f} text={repr(cand.merged_text)} box={cand.box}",
                flush=True,
            )

        return candidates

    def extract_text_items_from_predict_results(
        self,
        raw_results: Any,
        frame_index: int,
    ) -> List[OCRTextItem]:
        items: List[OCRTextItem] = []

        if raw_results is None:
            return items

        try:
            result_list = list(raw_results)
        except TypeError:
            result_list = [raw_results]

        if frame_index == 0:
            print(f"[Debug][Frame {frame_index}] predict result count = {len(result_list)}", flush=True)

        for item in result_list:
            data = None

            if hasattr(item, "res"):
                data = item.res
            elif isinstance(item, dict):
                data = item
            else:
                try:
                    data = dict(item)
                except Exception:
                    data = None

            if not isinstance(data, dict):
                continue

            if frame_index == 0:
                print(f"[Debug][Frame {frame_index}] result keys = {list(data.keys())}", flush=True)

            rec_texts = data.get("rec_texts", None)
            rec_scores = data.get("rec_scores", None)
            rec_boxes = data.get("rec_boxes", None)
            rec_polys = data.get("rec_polys", None)
            dt_polys = data.get("dt_polys", None)

            if rec_texts is None:
                continue

            rec_texts = list(rec_texts)
            rec_scores = list(rec_scores) if rec_scores is not None else [1.0] * len(rec_texts)

            box_source = None
            if rec_boxes is not None:
                box_source = rec_boxes
            elif rec_polys is not None:
                box_source = rec_polys
            elif dt_polys is not None:
                box_source = dt_polys

            if box_source is None:
                continue

            try:
                box_arr = np.asarray(box_source, dtype=np.float32)
            except Exception:
                continue

            if box_arr.ndim == 2 and box_arr.shape[-1] == 4:
                for i, text in enumerate(rec_texts):
                    if i >= len(box_arr):
                        break
                    x1, y1, x2, y2 = box_arr[i].tolist()
                    items.append(
                        OCRTextItem(
                            frame_index=frame_index,
                            x1=int(round(x1)),
                            y1=int(round(y1)),
                            x2=int(round(x2)),
                            y2=int(round(y2)),
                            text=str(text).strip(),
                            score=float(rec_scores[i]) if i < len(rec_scores) else 1.0,
                        )
                    )

            elif box_arr.ndim == 3 and box_arr.shape[-1] == 2:
                for i, text in enumerate(rec_texts):
                    if i >= len(box_arr):
                        break
                    poly = box_arr[i]
                    xs = poly[:, 0]
                    ys = poly[:, 1]
                    items.append(
                        OCRTextItem(
                            frame_index=frame_index,
                            x1=int(round(float(xs.min()))),
                            y1=int(round(float(ys.min()))),
                            x2=int(round(float(xs.max()))),
                            y2=int(round(float(ys.max()))),
                            text=str(text).strip(),
                            score=float(rec_scores[i]) if i < len(rec_scores) else 1.0,
                        )
                    )

        return items

    def group_text_items(self, items: List[OCRTextItem]) -> List[List[OCRTextItem]]:
        if len(items) == 0:
            return []

        items = sorted(items, key=lambda it: (it.y1, it.x1))

        groups: List[List[OCRTextItem]] = []
        used = [False] * len(items)

        for i in range(len(items)):
            if used[i]:
                continue

            current_group = [items[i]]
            used[i] = True

            changed = True
            while changed:
                changed = False
                gx1 = min(it.x1 for it in current_group)
                gy1 = min(it.y1 for it in current_group)
                gx2 = max(it.x2 for it in current_group)
                gy2 = max(it.y2 for it in current_group)
                gh = max(1, gy2 - gy1)
                gw = max(1, gx2 - gx1)

                for j in range(len(items)):
                    if used[j]:
                        continue

                    it = items[j]
                    iou = self.compute_iou((gx1, gy1, gx2, gy2), it.box)

                    x_gap = self.box_horizontal_gap((gx1, gy1, gx2, gy2), it.box)
                    y_gap = self.box_vertical_gap((gx1, gy1, gx2, gy2), it.box)

                    close_x = x_gap <= int(round(gh * self.group_x_gap_ratio))
                    close_y = y_gap <= int(round(gh * self.group_y_gap_ratio))
                    overlaps = iou > 0.0

                    within_band = (
                        it.x1 <= gx2 + int(round(gw * 0.3))
                        and it.x2 >= gx1 - int(round(gw * 0.3))
                        and it.y1 <= gy2 + int(round(gh * 0.8))
                        and it.y2 >= gy1 - int(round(gh * 0.8))
                    )

                    if (close_x and close_y) or overlaps or within_band:
                        current_group.append(it)
                        used[j] = True
                        changed = True

            groups.append(sorted(current_group, key=lambda it: (it.y1, it.x1)))

        return groups

    def build_group_text(self, items: List[OCRTextItem]) -> str:
        if len(items) == 0:
            return ""

        lines: List[List[OCRTextItem]] = []
        for item in sorted(items, key=lambda it: (it.y1, it.x1)):
            placed = False
            for line in lines:
                ly1 = min(x.y1 for x in line)
                ly2 = max(x.y2 for x in line)
                lh = max(1, ly2 - ly1)
                if abs(item.y1 - ly1) <= int(round(lh * 0.6)):
                    line.append(item)
                    placed = True
                    break
            if not placed:
                lines.append([item])

        out_lines: List[str] = []
        for line in lines:
            line_sorted = sorted(line, key=lambda it: it.x1)
            line_text = " ".join(it.text.strip() for it in line_sorted if it.text.strip())
            if line_text:
                out_lines.append(line_text)

        return "\n".join(out_lines)

    def group_text_is_too_weak(self, merged_text: str, item_count: int) -> bool:
        text = merged_text.strip()
        if text == "":
            return True

        norm = self.normalize_text(text)
        if norm in ("none", "unknown", "unreadable"):
            return True

        has_digits = re.search(r"\d", text) is not None
        has_hangul = re.search(r"[가-힣]", text) is not None
        has_alpha_word = re.search(r"[A-Za-z]{3,}", text) is not None
        has_cjk = re.search(r"[\u4e00-\u9fff]", text) is not None

        if item_count == 1:
            compact = re.sub(r"\s+", "", text)
            if len(compact) <= 2 and not has_hangul and not has_alpha_word:
                return True

        if not (has_digits or has_hangul or has_alpha_word or has_cjk):
            return True

        return False

    # -------------------------------------------------------------------------
    # Stage 2: Qwen reasoning only on grouped object
    # -------------------------------------------------------------------------
    def verify_group_reason_only(
        self,
        frame_bgr: np.ndarray,
        group: TextGroupCandidate,
        user_prompt: str,
    ) -> Optional[VerifiedDetection]:
        frame_index = group.frame_index
        visible_text = group.merged_text.strip()

        if self.group_text_is_too_weak(visible_text, group.item_count):
            return None

        crop_rgb = None
        try:
            crop_rgb = self.crop_with_margin(
                frame_bgr=frame_bgr,
                x1=group.x1,
                y1=group.y1,
                x2=group.x2,
                y2=group.y2,
                margin_ratio=self.crop_margin_ratio,
            )

            if crop_rgb is None:
                return None

            crop_rgb = self.ensure_min_crop_size(crop_rgb, self.min_qwen_crop_size)
            image_pil = Image.fromarray(crop_rgb)

            print(
                f"[Group-REC][Frame {frame_index}] items={group.item_count} score={group.avg_score:.3f} text={repr(visible_text)} box={group.box}",
                flush=True,
            )

            reason_prompt = self.build_privacy_reason_prompt(
                user_prompt=user_prompt,
                visible_text=visible_text,
            )
            raw_reason_answer = self.ask_qwen_reason(
                image_pil=image_pil,
                prompt_text=reason_prompt,
                max_new_tokens=self.max_new_tokens_reason,
            )

            decision, normalized_label, reason = self.parse_qwen_reason_answer(
                answer=raw_reason_answer,
                fallback_label="other private text",
            )

            print(
                f"[Qwen-Reason][Frame {frame_index}] decision={decision} label={normalized_label} text={repr(visible_text)} reason={reason} raw_answer={repr(raw_reason_answer)}",
                flush=True,
            )

            if not decision:
                return None

            det = VerifiedDetection(
                frame_index=frame_index,
                x1=group.x1,
                y1=group.y1,
                x2=group.x2,
                y2=group.y2,
                label=normalized_label,
                detector_phrase="grouped text object",
                detector_score=float(group.avg_score),
                qwen_visible_text=visible_text,
                qwen_reason=reason,
            )

            if self.save_verified_crops:
                self.save_debug_crop(crop_rgb=crop_rgb, detection=det)

            return det

        except Exception as e:
            print(f"[Error] Frame {frame_index} group reason 예외: {e}", flush=True)
            traceback.print_exc()
            return None

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def parse_qwen_reason_answer(
        self,
        answer: str,
        fallback_label: str,
    ) -> Tuple[bool, str, str]:
        text = answer.strip()

        decision_match = re.search(r"decision\s*:\s*(yes|no)", text, flags=re.IGNORECASE)
        label_match = re.search(
            r"label\s*:\s*(receipt|credit card|driver'?s license|license plate|address document|other private text|other)",
            text,
            flags=re.IGNORECASE,
        )
        reason_match = re.search(r"reason\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

        decision = False
        if decision_match is not None:
            decision = decision_match.group(1).lower() == "yes"
        else:
            decision = self.parse_yes_no(text)

        raw_label = fallback_label
        if label_match is not None:
            raw_label = label_match.group(1).lower().strip()

        reason = ""
        if reason_match is not None:
            reason = reason_match.group(1).strip().splitlines()[0].strip()

        normalized_label = self.normalize_label_name(raw_label)
        return decision, normalized_label, reason

    def is_missing_or_weak_visible_text(self, visible_text: str) -> bool:
        if visible_text is None:
            return True

        vt = visible_text.strip()
        if vt == "":
            return True

        vt_norm = self.normalize_text(vt)
        if vt_norm in ("none", "n a", "na", "unknown", "not visible", "unreadable"):
            return True

        if vt_norm in self.WEAK_EVIDENCE_WORDS:
            return True

        has_digits = re.search(r"\d", vt) is not None
        has_hangul = re.search(r"[가-힣]", vt) is not None
        has_alpha_word = re.search(r"[A-Za-z]{2,}", vt) is not None
        has_cjk = re.search(r"[\u4e00-\u9fff]", vt) is not None

        if not (has_digits or has_hangul or has_alpha_word or has_cjk):
            return True

        return False

    def visible_text_is_gibberish(self, visible_text: str) -> bool:
        vt = visible_text.strip()

        if len(vt) >= 8 and len(set(vt)) == 1:
            return True

        if re.fullmatch(r"(\d)\1{7,}", vt):
            return True
        if re.fullmatch(r"([A-Za-z])\1{7,}", vt):
            return True

        digits = re.sub(r"\D", "", vt)
        if len(digits) >= 12 and len(set(digits)) <= 2:
            return True

        cleaned = re.sub(r"[0-9A-Za-z가-힣\u4e00-\u9fff\s:/\-.(),#]", "", vt)
        if len(cleaned) >= 5 and len(vt) >= 10:
            return True

        return False

    def normalize_label_name(self, label: str) -> str:
        n = self.normalize_text(label)
        if n in ("driver s license", "driver's license", "driver license", "driving license", "id card", "identity card"):
            return "driver's license"
        if n in ("credit card", "debit card", "bank card", "payment card", "card"):
            return "credit card"
        if n in ("receipt", "bill", "paper receipt", "purchase receipt"):
            return "receipt"
        if n in ("license plate", "number plate", "plate", "car plate", "vehicle plate"):
            return "license plate"
        if n in ("address document",):
            return "address document"
        if n in ("waybill", "delivery label", "shipping label", "invoice"):
            return "waybill"
        if n in ("road sign", "street sign", "traffic sign", "sign"):
            return "road sign"
        if n in ("other private text", "other private", "private text"):
            return "other private text"
        return "other"

    def save_debug_crop(self, crop_rgb: np.ndarray, detection: VerifiedDetection) -> None:
        label = str(detection.label).replace(" ", "_").replace("/", "_").replace("'", "")
        filename = (
            f"frame_{detection.frame_index}_{label}_"
            f"{detection.x1}_{detection.y1}_{detection.x2}_{detection.y2}.jpg"
        )
        save_path = self.debug_crop_dir / filename
        Image.fromarray(crop_rgb).save(save_path)

    # -------------------------------------------------------------------------
    # Stage 3: Simple ID-based tracking
    # -------------------------------------------------------------------------
    def update_tracks_for_frame(
        self,
        detections: List[VerifiedDetection],
        frame_index: int,
        max_track_gap_frames: int,
    ) -> None:
        expired_ids = [
            track_id
            for track_id, track in self.active_tracks.items()
            if frame_index - track.last_frame > max_track_gap_frames
        ]
        for track_id in expired_ids:
            self.finished_tracks.append(self.active_tracks.pop(track_id))

        if len(detections) == 0:
            return

        track_ids = list(self.active_tracks.keys())
        candidate_pairs: List[Tuple[float, int, str]] = []

        for det_idx, det in enumerate(detections):
            for track_id in track_ids:
                track = self.active_tracks[track_id]
                # 라벨이 달라도 위치(IoU)가 겹치면 같은 객체로 인정
                # if track.label != det.label:
                #    continue

                iou = self.compute_iou(track.last_box, det.box)
                if iou >= self.track_iou_threshold:
                    candidate_pairs.append((iou, det_idx, track_id))

        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        matched_det_indices = set()
        matched_track_ids = set()

        for _, det_idx, track_id in candidate_pairs:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue

            det = detections[det_idx]
            track = self.active_tracks[track_id]
            track.update(det)
            det.track_id = track.object_id

            matched_det_indices.add(det_idx)
            matched_track_ids.add(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue

            new_track = self.create_track(det)
            self.active_tracks[new_track.object_id] = new_track
            det.track_id = new_track.object_id

    def create_track(self, detection: VerifiedDetection) -> TrackState:
        object_id = f"object_{self._next_track_index:03d}"
        self._next_track_index += 1

        return TrackState(
            object_id=object_id,
            label=detection.label,
            start_frame=detection.frame_index,
            end_frame=detection.frame_index,
            last_frame=detection.frame_index,
            last_box=detection.box,
            history_boxes=[detection.box],
            visible_text_samples=[detection.qwen_visible_text] if detection.qwen_visible_text else [],
            reason_samples=[detection.qwen_reason] if detection.qwen_reason else [],
            hit_count=1,
        )

    def finalize_all_tracks(self) -> None:
        for track_id in list(self.active_tracks.keys()):
            self.finished_tracks.append(self.active_tracks.pop(track_id))

    @staticmethod
    def compute_iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0

        return float(inter_area / union_area)

    @staticmethod
    def box_horizontal_gap(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> int:
        ax1, _, ax2, _ = box_a
        bx1, _, bx2, _ = box_b
        if ax2 < bx1:
            return bx1 - ax2
        if bx2 < ax1:
            return ax1 - bx2
        return 0

    @staticmethod
    def box_vertical_gap(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> int:
        _, ay1, _, ay2 = box_a
        _, by1, _, by2 = box_b
        if ay2 < by1:
            return by1 - ay2
        if by2 < ay1:
            return ay1 - by2
        return 0

    # -------------------------------------------------------------------------
    # Export / debug outputs
    # -------------------------------------------------------------------------
    def export_to_json(
        self,
        output_json_path: str,
        tracks: Optional[List[TrackState]] = None,
    ) -> List[Dict]:
        if tracks is None:
            tracks = list(self.last_tracks)  # 복사본으로
        # 중복 제거 전 디버그
        print(f"[Debug] 중복 제거 전 트랙 수: {len(tracks)}", flush=True)
        # IoU 기반 중복 트랙 제거

        # IoU 기반 중복 트랙 제거
        def compute_iou_box(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            if inter == 0:
                return 0.0
            area_a = (ax2-ax1) * (ay2-ay1)
            area_b = (bx2-bx1) * (by2-by1)
            return inter / (area_a + area_b - inter)

        merged_tracks = []
        used = set()
        for i, t1 in enumerate(tracks):
            if i in used:
                continue
            box1 = t1.representative_box()
            # 화면 너무 큰 박스 제거
            bw = box1[2] - box1[0]
            bh = box1[3] - box1[1]
            if bw * bh > 300000:
                used.add(i)
                continue
            best_start = t1.start_frame
            best_end = t1.end_frame
            for j, t2 in enumerate(tracks):
                if i == j or j in used:
                    continue
                box2 = t2.representative_box()
                if compute_iou_box(box1, box2) >= 0.5:
                    best_start = min(best_start, t2.start_frame)
                    best_end = max(best_end, t2.end_frame)
                    used.add(j)
            t1.start_frame = best_start
            t1.end_frame = best_end
            merged_tracks.append(t1)
            used.add(i)
        tracks = merged_tracks

        print(f"[Debug] 중복 제거 후 트랙 수: {len(tracks)}", flush=True)

        payload: List[Dict] = []
        for track in tracks:
            payload.append(
                {
                    "id": track.object_id,
                    "type": "object",
                    "label": self.LABEL_DISPLAY_MAP.get(track.label, track.label),
                    "start_frame": int(track.start_frame),
                    "end_frame": int(track.end_frame),
                    "box": track.representative_box(),
                    "visible_text": track.representative_visible_text(),
                }
            )

        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.last_json_payload = payload
        
        print(f"[Export] JSON 저장 완료: {output_json_path}", flush=True)
        return payload

    def group_detections_by_frame(
        self,
        detections: Optional[List[VerifiedDetection]] = None,
    ) -> Dict[int, List[VerifiedDetection]]:
        if detections is None:
            detections = self.last_verified_detections

        grouped: Dict[int, List[VerifiedDetection]] = {}
        for det in detections:
            grouped.setdefault(det.frame_index, []).append(det)
        return grouped

    def write_debug_video_fullfps(
        self,
        input_video_path: str,
        output_video_path: str,
        detections: Optional[List[VerifiedDetection]] = None,
    ) -> None:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or not np.isfinite(fps):
            fps = 24.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        grouped = self.group_detections_by_frame(detections)

        frame_index = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_dets = grouped.get(frame_index, [])
                for det in frame_dets:
                    self.draw_bbox(
                        frame_bgr,
                        det.x1,
                        det.y1,
                        det.x2,
                        det.y2,
                        f"{det.track_id}:{self.LABEL_DISPLAY_MAP.get(det.label, det.label)}",
                    )

                writer.write(frame_bgr)
                frame_index += 1

        finally:
            cap.release()
            writer.release()

        print(f"[Debug] full-fps bbox 영상 저장 완료: {output_video_path}", flush=True)

    def write_debug_video_sampled(
        self,
        input_video_path: str,
        output_video_path: str,
        sample_fps: float,
        detections: Optional[List[VerifiedDetection]] = None,
    ) -> None:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_video_path}")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if native_fps <= 0 or not np.isfinite(native_fps):
            native_fps = 24.0

        sample_interval_frames = max(1, int(round(native_fps / sample_fps)))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, sample_fps, (width, height))

        grouped = self.group_detections_by_frame(detections)

        frame_index = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_index % sample_interval_frames != 0:
                    frame_index += 1
                    continue

                frame_dets = grouped.get(frame_index, [])
                for det in frame_dets:
                    self.draw_bbox(
                        frame_bgr,
                        det.x1,
                        det.y1,
                        det.x2,
                        det.y2,
                        f"{det.track_id}:{self.LABEL_DISPLAY_MAP.get(det.label, det.label)}",
                    )

                writer.write(frame_bgr)
                frame_index += 1

        finally:
            cap.release()
            writer.release()

        print(f"[Debug] sampled bbox 영상 저장 완료: {output_video_path}", flush=True)

    @staticmethod
    def draw_bbox(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # PIL로 한글 텍스트 그리기
        from PIL import ImageFont, ImageDraw, Image
        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
        except:
            font = ImageFont.load_default()
        text_y = max(5, y1 - 25)
        draw.text((x1, text_y), label, font=font, fill=(0, 255, 0))
        frame_bgr[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def crop_with_margin(
        self,
        frame_bgr: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        margin_ratio: float = 0.35,
    ) -> Optional[np.ndarray]:
        height, width = frame_bgr.shape[:2]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        margin_x = int(round(box_w * margin_ratio))
        margin_y = int(round(box_h * margin_ratio))

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(width, x2 + margin_x)
        crop_y2 = min(height, y2 + margin_y)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop_bgr = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return crop_rgb

    def ensure_min_crop_size(self, crop_rgb: np.ndarray, min_size: int) -> np.ndarray:
        h, w = crop_rgb.shape[:2]
        new_h = max(h, min_size)
        new_w = max(w, min_size)

        if new_h == h and new_w == w:
            return crop_rgb

        pad_top = (new_h - h) // 2
        pad_bottom = new_h - h - pad_top
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left

        return cv2.copyMakeBorder(
            crop_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def parse_yes_no(answer: str) -> bool:
        text = answer.lower().strip()

        match = re.search(r"\b(yes|no)\b", text)
        if match is not None:
            return match.group(1) == "yes"

        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False

        return False

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip(" .,:;!?")

    @staticmethod
    def clamp_box(
        box_xyxy: Sequence[float],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box_xyxy
        x1 = int(max(0, min(width - 1, round(float(x1)))))
        y1 = int(max(0, min(height - 1, round(float(y1)))))
        x2 = int(max(0, min(width, round(float(x2)))))
        y2 = int(max(0, min(height, round(float(y2)))))
        return x1, y1, x2, y2


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
    return parser.parse_args()


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
        debug_crop_dir = project_root / "debug_crops"
        from datetime import datetime
        _timestamp = datetime.now().strftime("%m%d_%H%M")
        debug_video_fullfps_path = project_root / f"debug_bbox_output_fullfps_{_timestamp}.mp4"
        debug_video_sampled_path = project_root / f"debug_bbox_output_sampled_{_timestamp}.mp4"
        json_output_path = project_root / "privacy_runtime_bar.json"
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
        