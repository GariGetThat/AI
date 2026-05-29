# config.py

from pathlib import Path
import torch

# =========================
# Project Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OBJECT_OUTPUT_DIR = OUTPUT_DIR / "object"

OBJECT_DB_PATH = OBJECT_OUTPUT_DIR / "object_db.json"
OBJECT_DEBUG_CROP_DIR = OBJECT_OUTPUT_DIR / "debug_crops"

# =========================
# Device
# =========================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# =========================
# Object Detection / Privacy Reasoning
# =========================
OBJECT_QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
OBJECT_DEFAULT_PROMPT = "내 프라이버시가 유출될 만한 것들을 가려줘."
OBJECT_SAMPLE_FPS = 1.0
OBJECT_TEXT_DETECTOR_LANG = "korean"

OBJECT_CROP_MARGIN_RATIO = 0.35
OBJECT_MAX_NEW_TOKENS_REASON = 48
OBJECT_TRACK_IOU_THRESHOLD = 0.30

OBJECT_MIN_TEXT_BOX_AREA = 16
OBJECT_MAX_TEXT_BOX_AREA_RATIO = 0.75
OBJECT_MIN_TEXT_REGION_WIDTH = 3
OBJECT_MIN_TEXT_REGION_HEIGHT = 3
OBJECT_MIN_REC_SCORE = 0.30
OBJECT_MIN_GROUP_ITEMS = 1
OBJECT_GROUP_X_GAP_RATIO = 1.6
OBJECT_GROUP_Y_GAP_RATIO = 1.0
OBJECT_MIN_QWEN_CROP_SIZE = 56

# =========================
# Pass4 Merge Targets
# =========================
SAM2_INPUT_PATH = OUTPUT_DIR / "sam2_input_face.json"
SAM2_TARGETS_PATH = OUTPUT_DIR / "sam2_targets.json"
