import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import gc
import sys
import traceback
import faulthandler
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


class PrivacyReasoningEngine:
    """
    Stage 1: GroundingDINO proposes candidate boxes
    Stage 2: Qwen2-VL verifies whether each crop should be blurred

    Output format:
      [
        [frame_index, x1, y1, x2, y2, label],
        ...
      ]
    """

    DEFAULT_LABEL_ALIASES: Dict[str, Tuple[str, ...]] = {
        "license plate": (
            "license plate",
            "number plate",
            "plate",
            "car plate",
            "vehicle plate",
        ),
        "driver's license": (
            "driver's license",
            "driver license",
            "driving license",
            "license card",
            "id card",
            "identity card",
        ),
        "credit card": (
            "credit card",
            "debit card",
            "bank card",
            "payment card",
            "card",
        ),
        "receipt": (
            "receipt",
            "bill",
            "paper receipt",
            "purchase receipt",
        ),
        "face": ("face",),
    }

    DEFAULT_TARGET_LABELS: Tuple[str, ...] = (
        "license plate",
        "driver's license",
        "credit card",
        "receipt",
    )

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
        max_new_tokens: int = 10,
        label_aliases: Optional[Dict[str, Sequence[str]]] = None,
        verbose: bool = False,
        save_verified_crops: bool = True,
        debug_crop_dir: str = "debug_crops",
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

        merged_aliases: Dict[str, Tuple[str, ...]] = dict(self.DEFAULT_LABEL_ALIASES)
        if label_aliases is not None:
            for key, values in label_aliases.items():
                merged_aliases[key] = tuple(values)
        self.label_aliases = merged_aliases

        if self.save_verified_crops:
            self.debug_crop_dir.mkdir(parents=True, exist_ok=True)

        self._log("엔진 초기화 시작")
        self._log(f"device = {self.device}")
        self._log(f"gdino_config_path = {gdino_config_path}")
        self._log(f"gdino_checkpoint_path = {gdino_checkpoint_path}")
        self._log(f"crop_margin_ratio = {self.crop_margin_ratio}")

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
            torch_dtype=torch.float16,
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

    def _log_verified(self, frame_index: int, label: str, x1: int, y1: int, x2: int, y2: int) -> None:
        print(
            f"[Frame {frame_index}] 검증 통과: {label} | box=({x1}, {y1}, {x2}, {y2})",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Prompt / label inference
    # -------------------------------------------------------------------------
    def infer_target_labels_from_user_prompt(self, user_prompt: str) -> List[str]:
        """
        Convert user natural-language prompt into detector labels.
        This is conservative and grounded in the supported privacy categories.
        """
        p = self.normalize_text(user_prompt)
        if not p:
            return list(self.DEFAULT_TARGET_LABELS)

        matched: List[str] = []

        def contains_any(tokens: Sequence[str]) -> bool:
            return any(tok in p for tok in tokens)

        if contains_any(["receipt", "bill", "영수증", "결제내역", "구매내역"]):
            matched.append("receipt")

        if contains_any([
            "credit card", "debit card", "bank card", "payment card", "card",
            "신용카드", "체크카드", "카드", "결제카드"
        ]):
            matched.append("credit card")

        if contains_any([
            "driver's license", "driver license", "driving license", "id card", "identity card",
            "운전면허증", "면허증", "신분증", "id"
        ]):
            matched.append("driver's license")

        if contains_any([
            "license plate", "number plate", "plate",
            "번호판", "차량번호판"
        ]):
            matched.append("license plate")

        # Broad privacy / address requests widen to the supported set
        if contains_any([
            "privacy", "private information", "personal information", "sensitive",
            "address", "home address", "confidential",
            "프라이버시", "개인정보", "민감정보", "유출", "주소", "집 주소"
        ]):
            for label in self.DEFAULT_TARGET_LABELS:
                if label not in matched:
                    matched.append(label)

        if len(matched) == 0:
            matched = list(self.DEFAULT_TARGET_LABELS)

        return matched

    def build_detector_prompt(self, target_labels: Sequence[str]) -> str:
        normalized = [self.normalize_text(label) for label in target_labels]
        return " . ".join(normalized) + " ."

    def build_verification_question(
        self,
        user_prompt: str,
        label: str,
        detector_phrase: str,
    ) -> str:
        """
        Qwen is asked to judge whether the crop should be blurred under the user's request.
        It should say yes when private text/object is visible even partially.
        It should reject people/faces/reflections unless actual sensitive text/object appears.
        """
        label_norm = self.normalize_text(label)

        if label_norm == "receipt":
            focus_line = (
                "Decide whether this crop contains a receipt, a partial receipt, or receipt-like payment text."
            )
        elif label_norm == "credit card":
            focus_line = (
                "Decide whether this crop contains a payment card, a partial payment card, or card-related private/payment information."
            )
        elif label_norm == "driver's license":
            focus_line = (
                "Decide whether this crop contains a driver's license, an ID card, a partial ID document, or identity-related private text."
            )
        elif label_norm == "license plate":
            focus_line = (
                "Decide whether this crop contains a vehicle license plate or a partial visible plate number."
            )
        else:
            focus_line = (
                "Decide whether this crop contains privacy-sensitive text or objects relevant to the user's request."
            )

        return (
            "You are checking whether this image crop should be privacy-blurred.\n"
            f"User request: {user_prompt}\n"
            f"Detector candidate label: {label}\n"
            f"Detector phrase: {detector_phrase}\n"
            f"{focus_line}\n"
            "Answer yes if the crop contains privacy-sensitive content relevant to the request, even if it is partially cut off.\n"
            "Examples of privacy-sensitive content include receipts, payment cards, driver's licenses, ID cards, addresses, names, numbers, payment info, or identity-related text.\n"
            "Do NOT answer yes just because a person, face, body, reflection in glass, or mirror image appears.\n"
            "Only answer yes for people/reflections if actual privacy-sensitive text, documents, cards, or identifying written information is visible.\n"
            "Answer only yes or no."
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process_video(
        self,
        video_path: str = "test_input.mp4",
        user_prompt: str = "Blur privacy-sensitive content.",
        sample_fps: float = 1.0,
        detector_prompt: Optional[str] = None,
        target_labels: Optional[Sequence[str]] = None,
    ) -> List[List]:
        print("[Start] process_video 시작", flush=True)
        print(f"[Start] video_path = {video_path}", flush=True)
        print(f"[Start] user_prompt = {user_prompt}", flush=True)
        print(f"[Start] sample_fps = {sample_fps}", flush=True)

        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0.")

        if target_labels is None:
            target_labels = self.infer_target_labels_from_user_prompt(user_prompt)

        if detector_prompt is None:
            detector_prompt = self.build_detector_prompt(target_labels)

        print(f"[Start] target_labels = {list(target_labels)}", flush=True)
        print(f"[Start] detector_prompt = {detector_prompt}", flush=True)

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

        print(f"[Start] total_frames = {total_frames}", flush=True)
        print(f"[Start] native_fps = {native_fps}", flush=True)
        print(f"[Start] resolution = {width}x{height}", flush=True)
        print(f"[Start] sample_interval_frames = {sample_interval_frames}", flush=True)

        results: List[List] = []
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
                        target_labels=target_labels,
                    )
                except Exception as e:
                    print(f"[Error] Frame {frame_index} detect 실패: {e}", flush=True)
                    traceback.print_exc()
                    frame_index += 1
                    continue

                if len(candidates) > 0:
                    self._log_candidate_found(frame_index, len(candidates))

                for bbox_data in candidates:
                    try:
                        verified = self.verify_candidate(
                            frame_bgr=frame_bgr,
                            bbox_data=bbox_data,
                            user_prompt=user_prompt,
                        )
                    except Exception as e:
                        print(f"[Error] Frame {frame_index} verify 실패: {e}", flush=True)
                        traceback.print_exc()
                        continue

                    if verified:
                        results.append([
                            int(bbox_data["frame_index"]),
                            int(bbox_data["x1"]),
                            int(bbox_data["y1"]),
                            int(bbox_data["x2"]),
                            int(bbox_data["y2"]),
                            bbox_data["label"],
                        ])
                        self._log_verified(
                            frame_index=int(bbox_data["frame_index"]),
                            label=bbox_data["label"],
                            x1=int(bbox_data["x1"]),
                            y1=int(bbox_data["y1"]),
                            x2=int(bbox_data["x2"]),
                            y2=int(bbox_data["y2"]),
                        )

                frame_index += 1

        finally:
            cap.release()

        print(f"[Done] 샘플링된 프레임 수 = {sampled_count}", flush=True)
        print(f"[Done] 총 검증 통과 수 = {len(results)}", flush=True)
        return results

    # -------------------------------------------------------------------------
    # Stage 1: GroundingDINO
    # -------------------------------------------------------------------------
    def detect_candidates(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        detector_prompt: str,
        target_labels: Sequence[str],
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
        target_labels_norm = [self.normalize_text(label) for label in target_labels]

        candidates: List[Dict] = []

        for idx in keep_indices:
            phrase = self.normalize_text(str(phrases[idx]))

            if "face" in phrase:
                continue

            canonical_label = self.map_phrase_to_canonical_label(
                phrase=phrase,
                target_labels_norm=target_labels_norm,
            )
            if canonical_label is None or canonical_label == "face":
                continue

            x1, y1, x2, y2 = self.clamp_box(
                box_xyxy=boxes_xyxy[idx].tolist(),
                width=width,
                height=height,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            bbox_data = {
                "frame_index": int(frame_index),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "label": canonical_label,
                "phrase": phrase,
                "score": float(logits[idx].item()),
            }
            candidates.append(bbox_data)

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
    # Stage 2: Qwen2-VL verification
    # -------------------------------------------------------------------------
    def verify_candidate(
        self,
        frame_bgr: np.ndarray,
        bbox_data: Dict,
        user_prompt: str,
    ) -> bool:
        label = bbox_data["label"]
        frame_index = int(bbox_data["frame_index"])
        detector_phrase = str(bbox_data.get("phrase", label))

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
                return False

            image_pil = Image.fromarray(crop_rgb)

            question = self.build_verification_question(
                user_prompt=user_prompt,
                label=label,
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

            parsed = self.parse_yes_no(answer)

            if self.verbose:
                print(
                    f"[Qwen][Frame {frame_index}] label={label} phrase={detector_phrase} raw_answer={repr(answer)}",
                    flush=True,
                )

            if parsed and self.save_verified_crops and crop_rgb is not None:
                self.save_debug_crop(crop_rgb=crop_rgb, bbox_data=bbox_data)

            self._log(f"Frame {frame_index}: Qwen parsed={parsed}")
            return parsed

        except Exception as e:
            print(f"[Error] Frame {frame_index} verify 예외: {e}", flush=True)
            traceback.print_exc()
            return False

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def save_debug_crop(self, crop_rgb: np.ndarray, bbox_data: Dict) -> None:
        frame_index = int(bbox_data["frame_index"])
        label = str(bbox_data["label"]).replace(" ", "_").replace("/", "_")
        x1 = int(bbox_data["x1"])
        y1 = int(bbox_data["y1"])
        x2 = int(bbox_data["x2"])
        y2 = int(bbox_data["y2"])

        filename = f"frame_{frame_index}_{label}_{x1}_{y1}_{x2}_{y2}.jpg"
        save_path = self.debug_crop_dir / filename
        Image.fromarray(crop_rgb).save(save_path)

    # -------------------------------------------------------------------------
    # Debug video writers
    # -------------------------------------------------------------------------
    def group_results_by_frame(self, results: List[List]) -> Dict[int, List[List]]:
        grouped: Dict[int, List[List]] = {}
        for item in results:
            frame_idx = int(item[0])
            grouped.setdefault(frame_idx, []).append(item)
        return grouped

    def write_debug_video_fullfps(
        self,
        input_video_path: str,
        output_video_path: str,
        results: List[List],
    ) -> None:
        """
        Writes a full-FPS debug video.
        Bounding boxes are drawn only on frames where detections actually exist.
        """
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

        results_by_frame = self.group_results_by_frame(results)

        frame_index = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_results = results_by_frame.get(frame_index, [])

                for item in frame_results:
                    _, x1, y1, x2, y2, label = item
                    self.draw_bbox(frame_bgr, x1, y1, x2, y2, str(label))

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
        results: List[List],
        sample_fps: float,
    ) -> None:
        """
        Writes a sampled-only debug video containing only analyzed frames.
        This is often easier to inspect than full-FPS video.
        """
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

        results_by_frame = self.group_results_by_frame(results)

        frame_index = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_index % sample_interval_frames != 0:
                    frame_index += 1
                    continue

                frame_results = results_by_frame.get(frame_index, [])

                for item in frame_results:
                    _, x1, y1, x2, y2, label = item
                    self.draw_bbox(frame_bgr, x1, y1, x2, y2, str(label))

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
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def map_phrase_to_canonical_label(
        self,
        phrase: str,
        target_labels_norm: Sequence[str],
    ) -> Optional[str]:
        phrase_norm = self.normalize_text(phrase)

        for label in target_labels_norm:
            if label in phrase_norm or phrase_norm in label:
                return label

        for canonical_label, aliases in self.label_aliases.items():
            canonical_norm = self.normalize_text(canonical_label)

            if canonical_norm not in target_labels_norm and canonical_norm != "face":
                continue

            for alias in aliases:
                alias_norm = self.normalize_text(alias)
                if alias_norm in phrase_norm or phrase_norm in alias_norm:
                    return canonical_norm

        return None

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

        gdino_config_path = project_root / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        gdino_checkpoint_path = project_root / "models" / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"
        video_path = project_root / "test_input.mp4"
        debug_crop_dir = project_root / "debug_crops"
        debug_video_fullfps_path = project_root / "debug_bbox_output_fullfps.mp4"
        debug_video_sampled_path = project_root / "debug_bbox_output_sampled.mp4"

        if len(sys.argv) > 1:
            user_prompt = " ".join(sys.argv[1:])
        else:
            user_prompt = "내 프라이버시가 유출될 만한 것들을 가려줘."

        print(f"[Main] project_root = {project_root}", flush=True)
        print(f"[Main] gdino_config_path = {gdino_config_path}", flush=True)
        print(f"[Main] gdino_checkpoint_path = {gdino_checkpoint_path}", flush=True)
        print(f"[Main] video_path = {video_path}", flush=True)
        print(f"[Main] debug_crop_dir = {debug_crop_dir}", flush=True)
        print(f"[Main] debug_video_fullfps_path = {debug_video_fullfps_path}", flush=True)
        print(f"[Main] debug_video_sampled_path = {debug_video_sampled_path}", flush=True)
        print(f"[Main] user_prompt = {user_prompt}", flush=True)

        if not gdino_config_path.exists():
            raise FileNotFoundError(f"GroundingDINO config 파일이 없음: {gdino_config_path}")
        if not gdino_checkpoint_path.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint 파일이 없음: {gdino_checkpoint_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"비디오 파일이 없음: {video_path}")

        sample_fps = 1.0

        engine = PrivacyReasoningEngine(
            gdino_config_path=str(gdino_config_path),
            gdino_checkpoint_path=str(gdino_checkpoint_path),
            qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
            box_threshold=0.30,
            text_threshold=0.20,
            nms_iou_threshold=0.50,
            crop_margin_ratio=0.20,
            max_new_tokens=10,
            verbose=False,
            save_verified_crops=True,
            debug_crop_dir=str(debug_crop_dir),
        )

        results = engine.process_video(
            video_path=str(video_path),
            user_prompt=user_prompt,
            sample_fps=sample_fps,
        )

        print("[Main] 비디오 분석 종료", flush=True)

        if len(results) == 0:
            print("발견된 항목 없음", flush=True)
        else:
            print(f"[Main] 총 {len(results)}개 발견", flush=True)
            for item in results:
                print(item, flush=True)

        engine.write_debug_video_fullfps(
            input_video_path=str(video_path),
            output_video_path=str(debug_video_fullfps_path),
            results=results,
        )

        engine.write_debug_video_sampled(
            input_video_path=str(video_path),
            output_video_path=str(debug_video_sampled_path),
            results=results,
            sample_fps=sample_fps,
        )

    except Exception as e:
        print(f"[Fatal] 메인 실행 중 예외 발생: {e}", flush=True)
        traceback.print_exc()
        raise

    finally:
        print("[Main] 프로그램 종료", flush=True)
