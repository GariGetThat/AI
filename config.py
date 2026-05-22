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
INSIGHTFACE_INPUT_SIZE = (640, 640)
INSIGHTFACE_CONF_THRESH = 0.6
INSIGHTFACE_CTX_ID = 0   # Colab T4 GPU

# ─── ByteTrack ───────────────────────────────────────────
BYTETRACK_TRACK_THRESH  = 0.5
BYTETRACK_HIGH_THRESH   = 0.6
BYTETRACK_MATCH_THRESH  = 0.8
BYTETRACK_MAX_TIME_LOST = 30

# ─── track filtering ─────────────────────────────────────
MIN_TRACK_FRAMES = 10

# ─── DBSCAN ──────────────────────────────────────────────
DBSCAN_EPS = 0.45
DBSCAN_MIN_SAMPLES = 1

# ─── Top-N ───────────────────────────────────────────────
TOP_N = 2

# ─── 대표 crop ───────────────────────────────────────────
REPR_CROP_INTERVAL = 60
REPR_CROP_MIN_SIZE = 50
REPR_CROP_QUALITY = 90