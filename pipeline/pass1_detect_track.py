# pipeline/pass1_detect_track.py
"""
PASS 1 : 얼굴 검출 + 추적 + track_db 생성 + 대표 crop 추출

흐름:
    영상 → 슬라이딩 윈도우 분할
         → 각 프레임: SCRFD 검출 → ByteTrack 추적
         → track_db 누적
         → 주기마다 대표 crop 선정 · 저장
    → track_db.json 저장
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from db.schema import TrackDBEntry
from models.face_detector import BaseFaceDetector, build_detector
from models.face_tracker import BaseFaceTracker, build_tracker
from utils.crop import crop_face_by_kps, save_crop, select_best_crop
from utils.io import save_json
from utils.video import VideoMeta, build_windows, get_video_meta, iter_frames

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 내부 헬퍼 타입
# ─────────────────────────────────────────────────────────
# track_id → (crop_img, score) 후보 리스트
CropCandidates = Dict[int, List[Tuple[np.ndarray, float]]]


# ─────────────────────────────────────────────────────────
# 핵심 함수
# ─────────────────────────────────────────────────────────

def run_pass1(
    video_path: str | Path,
    detector: Optional[BaseFaceDetector] = None,
    tracker: Optional[BaseFaceTracker]   = None,
    crops_dir: str | Path  = config.CROPS_DIR,
    output_path: str | Path = config.TRACK_DB_PATH,
    window_sec: float       = config.VIDEO_WINDOW_SEC,
    overlap_sec: float      = config.VIDEO_OVERLAP_SEC,
    repr_interval: int      = config.REPR_CROP_INTERVAL,
    min_crop_size: int      = config.REPR_CROP_MIN_SIZE,
    crop_quality: int       = config.REPR_CROP_QUALITY,
    debug: bool             = False,
) -> Dict[int, TrackDBEntry]:
    """
    PASS1 전체 실행.

    Returns
    -------
    track_db : {track_id: TrackDBEntry}
    """
    # ── 초기화 ───────────────────────────────────────────
    crops_dir   = Path(crops_dir)
    output_path = Path(output_path)
    crops_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = build_detector(config.SCRFD_MODEL_PATH)
    if tracker is None:
        tracker  = build_tracker()

    meta    = get_video_meta(video_path)
    windows = build_windows(
        meta.total_frames, meta.fps, window_sec, overlap_sec
    )

    logger.info(
        "PASS1 시작 | 영상: %s | %.1f fps | %d frames | %d windows",
        Path(video_path).name, meta.fps, meta.total_frames, len(windows),
    )

    track_db: Dict[int, TrackDBEntry] = {}
    crop_candidates: CropCandidates   = {}   # track_id → [(crop, score)]

    # ── 윈도우 순회 ──────────────────────────────────────
    for win_idx, window in enumerate(windows):
        logger.debug(
            "window [%d/%d] frames %d ~ %d",
            win_idx + 1, len(windows),
            window.start_frame, window.end_frame,
        )
        tracker.reset()

        for frame_idx, frame in iter_frames(
            video_path, window.start_frame, window.end_frame
        ):
            # 1) 검출
            detections = detector.detect(frame)
            for d in detections:
                d.frame_idx = frame_idx

            # 2) 추적
            tracks = tracker.update(detections, frame_idx)

            # 2.5) 로그 출력
            if frame_idx % 30 == 0:
                logger.info(
                    "frame %d | detections=%d | tracks=%d",
                    frame_idx,
                    len(detections),
                    len(tracks),
                )

            # 3) track_db 누적
            for tr in tracks:
                _update_track_db(track_db, tr)

            # 4) 대표 crop 후보 수집 (repr_interval 프레임마다)
            if frame_idx % repr_interval == 0:
                for tr in tracks:
                    _collect_crop_candidate(
                        crop_candidates, tr, frame, min_crop_size
                    )

            # 5) 디버그 시각화
            if debug:
                _draw_debug(frame, tracks, frame_idx)
                cv2.imshow("PASS1 debug", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if debug:
        cv2.destroyAllWindows()

    # ── 대표 crop 저장 ───────────────────────────────────
    _save_repr_crops(track_db, crop_candidates, crops_dir, crop_quality)

    # ── duration 계산 ────────────────────────────────────
    for entry in track_db.values():
        entry.duration = len(entry.frames)

    # ── JSON 저장 ────────────────────────────────────────
    save_json(
        {str(k): v.to_dict() for k, v in track_db.items()},
        output_path,
    )
    logger.info(
        "PASS1 완료 | track 수: %d | 저장: %s",
        len(track_db), output_path,
    )

    return track_db


# ─────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────

def _update_track_db(
    track_db: Dict[int, TrackDBEntry],
    tr,   # TrackRecord
) -> None:
    """track_db 에 TrackRecord 를 누적."""
    tid = tr.track_id
    if tid not in track_db:
        track_db[tid] = TrackDBEntry(
            track_id=tid,
            start_frame=tr.frame_idx,
        )
    entry = track_db[tid]

    # 중복 프레임 방지 (overlap window 때문에 같은 프레임이 두 번 올 수 있음)
    if not entry.frames or entry.frames[-1] != tr.frame_idx:
        entry.frames.append(tr.frame_idx)
        entry.bboxes.append(list(tr.bbox))

    entry.end_frame = max(entry.end_frame, tr.frame_idx)


def _collect_crop_candidate(
    crop_candidates: CropCandidates,
    tr,             # TrackRecord
    frame: np.ndarray,
    min_crop_size: int,
) -> None:
    """현재 프레임에서 track 의 crop 후보를 수집."""
    tid  = tr.track_id
    bbox = tuple(int(v) for v in tr.bbox)

    crop = crop_face_by_kps(
        frame=frame,
        kps=tr.kps,
        bbox=bbox,
        min_size=min_crop_size,
    )

    if crop is None:
        return

    if tid not in crop_candidates:
        crop_candidates[tid] = []
    crop_candidates[tid].append((crop, tr.score))


def _save_repr_crops(
    track_db: Dict[int, TrackDBEntry],
    crop_candidates: CropCandidates,
    crops_dir: Path,
    quality: int,
) -> None:
    """각 track 의 대표 crop 을 선정하여 저장하고 경로를 track_db 에 기록."""
    for tid, entry in track_db.items():
        candidates = crop_candidates.get(tid, [])
        best_crop  = select_best_crop(candidates)
        if best_crop is None:
            logger.warning("track %d: 대표 crop 없음", tid)
            continue

        save_path = crops_dir / f"track_{tid:04d}.jpg"
        ok = save_crop(best_crop, save_path, quality=quality)
        if ok:
            entry.repr_crop_path = str(save_path)
        else:
            logger.warning("track %d: crop 저장 실패", tid)


def _draw_debug(frame: np.ndarray, tracks, frame_idx: int) -> None:
    cv2.putText(
        frame, f"frame {frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(
            frame, f"id:{tr.track_id}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2,
        )