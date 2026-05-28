import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import gc
import sys
import json
import argparse
import traceback
import faulthandler
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

faulthandler.enable()

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.ops import box_convert, nms

from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_utils import PreTrainedModel

# -----------------------------------------------------------------------------
# GroundingDINO compatibility patch for newer Transformers
# -----------------------------------------------------------------------------
if not hasattr(BertModel, "get_head_mask"):
    BertModel.get_head_mask = PreTrainedModel.get_head_mask

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict


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
    """
    Region proposal + OCR-aware privacy reasoning + simple IoU tracking
    """

    DEFAULT_REGION_TERMS: Tuple[str, ...] = (
        "document",
        "paper",
        "printed text",
        "label",
        "sign",
        "card",
        "license plate",
    )

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
        gdino_config_path: str,
        gdino_checkpoint_path: str,
        qwen_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        box_threshold: float = 0.30,
        text_threshold: float = 0.20,
        nms_iou_threshold: float = 0.50,
        crop_margin_ratio: float = 0.20,
        max_new_tokens: int = 28,
        verbose: bool = True,
        save_verified_crops: bool = True,
        debug_crop_dir: str = "debug_crops",
        track_iou_threshold: float = 0.30,
    ) -> None:
        self.verbose = bool(verbose)
        self.save_verified_crops = bool(save_verified_crops)
        self.debug_crop_dir = Path(debug_crop_dir)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.max_new_tokens = int(max_new_tokens)
        self.track_iou_threshold = float(track_iou_threshold)

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
        self._log(f"gdino_config_path = {gdino_config_path}")
        self._log(f"gdino_checkpoint_path = {gdino_checkpoint_path}")
        self._log(f"box_threshold = {self.box_threshold}")
        self._log(f"text_threshold = {self.text_threshold}")
        self._log(f"crop_margin_ratio = {self.crop_margin_ratio}")
        self._log(f"track_iou_threshold = {self.track_iou_threshold}")

        self.gdino_model = load_model(
            model_config_path=gdino_config_path,
            model_checkpoint_path=gdino_checkpoint_path,
            device=self.device,
        )
        self._log("GroundingDINO 로딩 완료")

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

        self.gdino_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Info] {message}", flush=True)

    def _log_candidate_found(self, frame_index: int, count: int) -> None:
        print(f"[Frame {frame_index}] DINO 후보 {count}개 발견", flush=True)

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
    def build_detector_prompt(self, user_prompt: str) -> str:
        terms = list(self.DEFAULT_REGION_TERMS)
        p = self.normalize_text(user_prompt)

        if any(tok in p for tok in ["address", "주소", "home address", "집 주소"]):
            for term in ["document", "paper", "label", "sign", "printed text"]:
                if term not in terms:
                    terms.append(term)

        if any(tok in p for tok in ["card", "카드", "credit", "payment", "결제"]):
            for term in ["card", "document", "paper"]:
                if term not in terms:
                    terms.append(term)

        if any(tok in p for tok in ["id", "identity", "신분증", "면허증", "운전면허증"]):
            for term in ["document", "card", "printed text"]:
                if term not in terms:
                    terms.append(term)

        return " . ".join(terms) + " ."

    def build_verification_question(
        self,
        user_prompt: str,
        detector_phrase: str,
    ) -> str:
        return (
            "You are an OCR-first privacy inspector for video editing.\n"
            f"User request: {user_prompt}\n"
            f"Detector hint region: {detector_phrase}\n"
            "Step 1: Read any visible text in the crop.\n"
            "Step 2: Decide whether the visible text or document/card/plate content reveals privacy-sensitive information relevant to the user's request.\n"
            "Privacy-sensitive examples include names, addresses, ID information, card numbers, payment information, receipt information, personal numbers, and identifying written clues.\n"
            "If the object is partially cut off but still reveals private text or identifying information, answer yes.\n"
            "Do NOT answer yes based only on object shape. Keyboard keys, paper texture, screen glow, reflections, silhouettes, faces, bodies, mirrors, and glass reflections are NOT enough.\n"
            "If you cannot read any privacy-relevant text or document/card/plate content, answer no.\n"
            "Return exactly these four lines:\n"
            "Decision: yes or no\n"
            "VisibleText: <copied visible text or NONE>\n"
            "Label: receipt | waybill | credit card | driver's license | license plate | road sign | address document | other private text | other\n"
            "Reason: <short reason>"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process_video(
        self,
        video_path: str,
        user_prompt: str = "내 프라이버시가 유출될 만한 것들을 가려줘.",
        sample_fps: float = 1.0,
        detector_prompt: Optional[str] = None,
        sam2_refresh_interval_frames: int = 450,
    ) -> List[TrackState]:
        print("[Start] process_video 시작", flush=True)
        print(f"[Start] video_path = {video_path}", flush=True)
        print(f"[Start] user_prompt = {user_prompt}", flush=True)
        print(f"[Start] sample_fps = {sample_fps}", flush=True)
        print(f"[Start] sam2_refresh_interval_frames = {sam2_refresh_interval_frames}", flush=True)

        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0.")

        if detector_prompt is None:
            detector_prompt = self.build_detector_prompt(user_prompt)

        print(f"[Start] detector_prompt = {detector_prompt}", flush=True)

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
                    candidates = self.detect_candidates(
                        frame_bgr=frame_bgr,
                        frame_index=frame_index,
                        detector_prompt=detector_prompt,
                    )
                except Exception as e:
                    print(f"[Error] Frame {frame_index} detect 실패: {e}", flush=True)
                    traceback.print_exc()
                    frame_index += 1
                    continue

                if len(candidates) > 0:
                    self._log_candidate_found(frame_index, len(candidates))

                frame_verified: List[VerifiedDetection] = []

                for bbox_data in candidates:
                    try:
                        verified_detection = self.verify_candidate_text_centric(
                            frame_bgr=frame_bgr,
                            bbox_data=bbox_data,
                            user_prompt=user_prompt,
                        )
                    except Exception as e:
                        print(f"[Error] Frame {frame_index} verify 실패: {e}", flush=True)
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
    # Stage 1: GroundingDINO candidate proposal
    # -------------------------------------------------------------------------
    def detect_candidates(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detector_prompt: str,
    ) -> List[Dict]:
        image_tensor = self.preprocess_for_gdino(frame_bgr)

        boxes, logits, phrases = predict(
            model=self.gdino_model,
            image=image_tensor,
            caption=detector_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
            remove_combined=True,
        )

        if boxes.numel() == 0:
            return []

        height, width = frame_bgr.shape[:2]
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")

        keep_indices = self.apply_nms(boxes_xyxy, logits)

        candidates: List[Dict] = []
        for idx in keep_indices:
            phrase = self.normalize_text(str(phrases[idx]))

            if "face" in phrase:
                continue

            x1, y1, x2, y2 = self.clamp_box(
                box_xyxy=boxes_xyxy[idx].tolist(),
                width=width,
                height=height,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            candidates.append(
                {
                    "frame_index": int(frame_index),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "phrase": phrase,
                    "score": float(logits[idx].item()),
                }
            )

        self._log(f"Frame {frame_index}: detect_candidates -> {len(candidates)}개")
        return candidates

    def preprocess_for_gdino(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        image_tensor, _ = self.gdino_transform(image_pil, None)
        return image_tensor

    def apply_nms(self, boxes_xyxy: torch.Tensor, scores: torch.Tensor) -> List[int]:
        if boxes_xyxy.numel() == 0:
            return []
        keep = nms(boxes_xyxy, scores, self.nms_iou_threshold)
        return [int(i) for i in keep.detach().cpu().tolist()]

    # -------------------------------------------------------------------------
    # Stage 2: Qwen2-VL OCR-first privacy reasoning
    # -------------------------------------------------------------------------
    def verify_candidate_text_centric(
        self,
        frame_bgr: np.ndarray,
        bbox_data: Dict,
        user_prompt: str,
    ) -> Optional[VerifiedDetection]:
        frame_index = int(bbox_data["frame_index"])
        detector_phrase = str(bbox_data.get("phrase", "region"))

        crop_rgb = None

        try:
            crop_rgb = self.crop_with_margin(
                frame_bgr=frame_bgr,
                x1=int(bbox_data["x1"]),
                y1=int(bbox_data["y1"]),
                x2=int(bbox_data["x2"]),
                y2=int(bbox_data["y2"]),
                margin_ratio=self.crop_margin_ratio,
            )

            if crop_rgb is None:
                return None

            image_pil = Image.fromarray(crop_rgb)
            question = self.build_verification_question(
                user_prompt=user_prompt,
                detector_phrase=detector_phrase,
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {"type": "text", "text": question},
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
                    max_new_tokens=self.max_new_tokens,
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

            decision, visible_text, normalized_label, reason = self.parse_qwen_ocr_answer(
                answer=answer,
                detector_phrase=detector_phrase,
            )

            print(
                f"[Qwen][Frame {frame_index}] phrase={detector_phrase} "
                f"decision={decision} label={normalized_label} visible_text={repr(visible_text)} "
                f"reason={reason} raw_answer={repr(answer)}",
                flush=True,
            )

            if not decision:
                return None

            det = VerifiedDetection(
                frame_index=frame_index,
                x1=int(bbox_data["x1"]),
                y1=int(bbox_data["y1"]),
                x2=int(bbox_data["x2"]),
                y2=int(bbox_data["y2"]),
                label=normalized_label,
                detector_phrase=detector_phrase,
                detector_score=float(bbox_data["score"]),
                qwen_visible_text=visible_text,
                qwen_reason=reason,
            )

            if self.save_verified_crops and crop_rgb is not None:
                self.save_debug_crop(crop_rgb=crop_rgb, detection=det)

            return det

        except Exception as e:
            print(f"[Error] Frame {frame_index} verify 예외: {e}", flush=True)
            traceback.print_exc()
            return None

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def parse_qwen_ocr_answer(
        self,
        answer: str,
        detector_phrase: str,
    ) -> Tuple[bool, str, str, str]:
        text = answer.strip()

        decision_match = re.search(r"decision\s*:\s*(yes|no)", text, flags=re.IGNORECASE)
        visible_text_match = re.search(r"visibletext\s*:\s*(.+)", text, flags=re.IGNORECASE)
        label_match = re.search(
            r"label\s*:\s*(receipt|credit card|driver'?s license|license plate|address document|other private text|other)",
            text,
            flags=re.IGNORECASE,
        )
        reason_match = re.search(r"reason\s*:\s*(.+)", text, flags=re.IGNORECASE)

        decision = False
        if decision_match is not None:
            decision = decision_match.group(1).lower() == "yes"
        else:
            decision = self.parse_yes_no(text)

        visible_text = "NONE"
        if visible_text_match is not None:
            visible_text = visible_text_match.group(1).strip()

        raw_label = "other"
        if label_match is not None:
            raw_label = label_match.group(1).lower().strip()

        reason = ""
        if reason_match is not None:
            reason = reason_match.group(1).strip()

        normalized_label = self.normalize_label_name(raw_label)

        if not decision:
            return False, visible_text, normalized_label, reason

        if self.is_missing_or_weak_visible_text(visible_text):
            return False, visible_text, normalized_label, reason

        if self.visible_text_is_just_detector_hint(visible_text, detector_phrase):
            return False, visible_text, normalized_label, reason

        return True, visible_text, normalized_label, reason

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
        has_alpha_word = re.search(r"[A-Za-z]{3,}", vt) is not None

        if not (has_digits or has_hangul or has_alpha_word):
            return True

        return False

    def visible_text_is_just_detector_hint(self, visible_text: str, detector_phrase: str) -> bool:
        vt = self.normalize_text(visible_text)
        dp = self.normalize_text(detector_phrase)

        if vt == dp:
            return True

        if vt in self.WEAK_EVIDENCE_WORDS:
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
        if n in ("other private text",):
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
                if track.label != det.label:
                    continue

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

    # -------------------------------------------------------------------------
    # Export / debug outputs
    # -------------------------------------------------------------------------
    def export_to_json(
        self,
        output_json_path: str,
        tracks: Optional[List[TrackState]] = None,
    ) -> List[Dict]:
        if tracks is None:
            tracks = self.last_tracks

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
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_y = max(25, y1 - 10)
        cv2.putText(
            frame_bgr,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

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
        margin_ratio: float = 0.20,
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
    parser = argparse.ArgumentParser(description="PrivacyReasoningEngine video test runner")
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
        "--sam2-refresh-interval-frames",
        type=int,
        default=450,
        help="SAM2 bbox refresh interval",
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

        gdino_config_path = project_root / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        gdino_checkpoint_path = project_root / "models" / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
        video_path = Path(args.video).expanduser().resolve()
        debug_crop_dir = project_root / "debug_crops"
        debug_video_fullfps_path = project_root / "debug_bbox_output_fullfps.mp4"
        debug_video_sampled_path = project_root / "debug_bbox_output_sampled.mp4"
        json_output_path = project_root / "privacy_runtime_bar.json"
        user_prompt = args.prompt
        sample_fps = args.sample_fps
        sam2_refresh_interval_frames = args.sam2_refresh_interval_frames

        print(f"[Main] project_root = {project_root}", flush=True)
        print(f"[Main] gdino_config_path = {gdino_config_path}", flush=True)
        print(f"[Main] gdino_checkpoint_path = {gdino_checkpoint_path}", flush=True)
        print(f"[Main] video_path = {video_path}", flush=True)
        print(f"[Main] debug_crop_dir = {debug_crop_dir}", flush=True)
        print(f"[Main] debug_video_fullfps_path = {debug_video_fullfps_path}", flush=True)
        print(f"[Main] debug_video_sampled_path = {debug_video_sampled_path}", flush=True)
        print(f"[Main] json_output_path = {json_output_path}", flush=True)
        print(f"[Main] user_prompt = {user_prompt}", flush=True)
        print(f"[Main] sample_fps = {sample_fps}", flush=True)
        print(f"[Main] sam2_refresh_interval_frames = {sam2_refresh_interval_frames}", flush=True)

        if not gdino_config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config 파일이 없음: {gdino_config_path}")
        if not gdino_checkpoint_path.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint 파일이 없음: {gdino_checkpoint_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일이 없음: {video_path}")

        engine = PrivacyReasoningEngine(
            gdino_config_path=str(gdino_config_path),
            gdino_checkpoint_path=str(gdino_checkpoint_path),
            qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
            box_threshold=0.30,
            text_threshold=0.20,
            nms_iou_threshold=0.50,
            crop_margin_ratio=0.20,
            max_new_tokens=28,
            verbose=True,
            save_verified_crops=True,
            debug_crop_dir=str(debug_crop_dir),
            track_iou_threshold=0.30,
        )

        tracks = engine.process_video(
            video_path=str(video_path),
            user_prompt=user_prompt,
            sample_fps=sample_fps,
            sam2_refresh_interval_frames=sam2_refresh_interval_frames,
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
        