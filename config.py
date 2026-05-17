# config.py
from pathlib import Path

# ─── 경로 ────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
OUTPUT_DIR  = ROOT_DIR / "outputs"
CROPS_DIR   = OUTPUT_DIR / "crops"
DEBUG_DIR   = OUTPUT_DIR / "debug"

TRACK_DB_PATH   = OUTPUT_DIR / "track_db.json"
PERSON_DB_PATH  = OUTPUT_DIR / "person_db.json"
SAM2_INPUT_PATH = OUTPUT_DIR / "sam2_input.json"

# ─── 영상 처리 ────────────────────────────────────────────
VIDEO_WINDOW_SEC     = 5       # sliding window 크기 (초)
VIDEO_OVERLAP_SEC    = 1       # window 간 overlap (초)

# ─── InsightFace Buffalo ─────────────────────────────
INSIGHTFACE_MODEL_PACK = "buffalo_l"
INSIGHTFACE_INPUT_SIZE = (640, 640)
INSIGHTFACE_CONF_THRESH = 0.5
INSIGHTFACE_CTX_ID = -1   # CPU
# GPU면 0

# ─── ByteTrack ───────────────────────────────────────────
BYTETRACK_TRACK_THRESH  = 0.5
BYTETRACK_HIGH_THRESH   = 0.6
BYTETRACK_MATCH_THRESH  = 0.8
BYTETRACK_MAX_TIME_LOST = 30   # frames

# ─── ArcFace ─────────────────────────────────────────────
ARCFACE_MODEL_PATH  = ROOT_DIR / "weights" / "arcface_r100.onnx"
ARCFACE_INPUT_SIZE  = (112, 112)
ARCFACE_EMB_DIM     = 512

# ─── DBSCAN ──────────────────────────────────────────────
DBSCAN_EPS          = 0.45     # cosine distance 기준
DBSCAN_MIN_SAMPLES  = 1

# ─── Top-N ───────────────────────────────────────────────
TOP_N = 2   # 블러 제외할 주요 인물 수 (사용자 설정 가능)

# ─── 대표 crop ───────────────────────────────────────────
REPR_CROP_INTERVAL   = 30   # N 프레임마다 후보 수집
REPR_CROP_MIN_SIZE   = 40   # crop 최소 크기 (px)
REPR_CROP_QUALITY    = 95   # jpg 저장 품질