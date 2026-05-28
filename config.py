import os
import platform
from pathlib import Path

# ─── 경로 ────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
OUTPUT_DIR  = ROOT_DIR / "outputs"
CROPS_DIR   = OUTPUT_DIR / "crops"
DEBUG_DIR   = OUTPUT_DIR / "debug"

TRACK_DB_PATH   = OUTPUT_DIR / "track_db.json"
PERSON_DB_PATH  = OUTPUT_DIR / "person_db.json"
SAM2_INPUT_PATH = OUTPUT_DIR / "sam2_input.json"

# ─── InsightFace Buffalo ─────────────────────────────────
INSIGHTFACE_MODEL_PACK = "buffalo_l"
# PASS1 detector용
INSIGHTFACE_ALLOWED_MODULES = ["detection"]
INSIGHTFACE_INPUT_SIZE = (640, 640)
INSIGHTFACE_CONF_THRESH = 0.6
# PASS2 recognizer용
RECOGNIZER_MODEL_NAME = "w600k_r50.onnx"
RECOGNIZER_INPUT_SIZE = (112, 112)

# 환경변수 우선
if "INSIGHTFACE_CTX_ID" in os.environ:
    INSIGHTFACE_CTX_ID = int(os.environ["INSIGHTFACE_CTX_ID"])

elif platform.system() == "Darwin":
    INSIGHTFACE_CTX_ID = -1

else:
    INSIGHTFACE_CTX_ID = 0

# ─── ByteTrack ───────────────────────────────────────────
BYTETRACK_TRACK_THRESH  = 0.5
BYTETRACK_HIGH_THRESH   = 0.6
BYTETRACK_MATCH_THRESH  = 0.7
BYTETRACK_MAX_TIME_LOST = 90

# ─── track filtering ─────────────────────────────────────
MIN_TRACK_FRAMES = 20

# ─── DBSCAN ──────────────────────────────────────────────
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 1

# ─── Top-N ───────────────────────────────────────────────
TOP_N = 2

# ─── 대표 crop ───────────────────────────────────────────
REPR_CROP_INTERVAL = 30
REPR_CROP_MIN_SIZE = 40
REPR_CROP_QUALITY = 90

# ─── person post-processing ──────────────────────────────
MIN_PERSON_FRAMES = 120
PERSON_MERGE_MAX_TIME_OVERLAP = 1.1
PERSON_MERGE_MAX_CENTER_DIST = 1.0
PERSON_MERGE_SIM_THRESH = 0.65