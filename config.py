from pathlib import Path
import os
import platform
import torch

# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------
# Face Pipeline
# ---------------------------------------------------------

CROPS_DIR = OUTPUT_DIR / "crops"
DEBUG_DIR = OUTPUT_DIR / "debug"

TRACK_DB_PATH = OUTPUT_DIR / "track_db.json"
PERSON_DB_PATH = OUTPUT_DIR / "person_db.json"

# PASS2/export_for_sam2 → PASS4
FACE_SAM2_INPUT_PATH = OUTPUT_DIR / "face_sam2_input.json"

# ---------------------------------------------------------
# Object Pipeline
# ---------------------------------------------------------

OBJECT_OUTPUT_DIR = OUTPUT_DIR / "object"
OBJECT_DB_PATH = OBJECT_OUTPUT_DIR / "object_db.json"
OBJECT_DEBUG_CROP_DIR = OBJECT_OUTPUT_DIR / "debug_crops"

# ---------------------------------------------------------
# PASS4 → PASS5
# ---------------------------------------------------------

SAM2_TARGETS_PATH = OUTPUT_DIR / "sam2_targets.json"

# ---------------------------------------------------------
# SAM2 Model
# ---------------------------------------------------------

SAM2_ROOT = PROJECT_ROOT / "third_party" / "sam2"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"

FINAL_OUTPUT_VIDEO_PATH = OUTPUT_DIR / "output_video.mp4"

# =========================================================
# Device
# =========================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# =========================================================
# InsightFace Buffalo
# =========================================================

INSIGHTFACE_MODEL_PACK = "buffalo_l"
INSIGHTFACE_ALLOWED_MODULES = ["detection"]

INSIGHTFACE_INPUT_SIZE = (640, 640)
INSIGHTFACE_CONF_THRESH = 0.6

RECOGNIZER_MODEL_NAME = "w600k_r50.onnx"
RECOGNIZER_INPUT_SIZE = (112, 112)

# Apple 서버 기준: 기본 CPU Provider 사용
# 필요하면 환경변수로 덮어쓰기 가능
if "INSIGHTFACE_CTX_ID" in os.environ:
    INSIGHTFACE_CTX_ID = int(os.environ["INSIGHTFACE_CTX_ID"])
elif platform.system() == "Darwin":
    INSIGHTFACE_CTX_ID = -1
else:
    INSIGHTFACE_CTX_ID = 0

# =========================================================
# ByteTrack
# =========================================================

BYTETRACK_TRACK_THRESH = 0.5
BYTETRACK_HIGH_THRESH = 0.6
BYTETRACK_MATCH_THRESH = 0.7
BYTETRACK_MAX_TIME_LOST = 90

# =========================================================
# Track Filtering
# =========================================================

MIN_TRACK_FRAMES = 20

# =========================================================
# DBSCAN
# =========================================================

DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 1

# =========================================================
# Top-N Main Person Selection
# =========================================================

TOP_N = 2

# =========================================================
# Representative Crop
# =========================================================

REPR_CROP_INTERVAL = 30
REPR_CROP_MIN_SIZE = 40
REPR_CROP_QUALITY = 90

# =========================================================
# Person Post Processing
# =========================================================

MIN_PERSON_FRAMES = 120
PERSON_MERGE_MAX_TIME_OVERLAP = 1.1
PERSON_MERGE_MAX_CENTER_DIST = 1.0
PERSON_MERGE_SIM_THRESH = 0.65

# =========================================================
# Object Detection / Privacy Reasoning
# =========================================================

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

# =========================================================
# SAM2
# =========================================================

SAM2_DEVICE = DEVICE
SAM2_FPS = 25
SAM2_CHUNK_SECONDS = 15
BLUR_STRENGTH = 31