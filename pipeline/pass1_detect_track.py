# pipeline/pass1_detect_track.py

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
from utils.video import get_video_meta, iter_frames

logger = logging.getLogger(__name__)

CropCandidates = Dict[int, List[Tuple[np.ndarray, float]]]


def run_pass1(
    video_path: str | Path,
    detector: Optional[BaseFaceDetector] = None,
    tracker: Optional[BaseFaceTracker] = None,
    crops_dir: str | Path = config.CROPS_DIR,
    output_path: str | Path = config.TRACK_DB_PATH,
    repr_interval: int = config.REPR_CROP_INTERVAL,
    min_crop_size: int = config.REPR_CROP_MIN_SIZE,
    crop_quality: int = config.REPR_CROP_QUALITY,
    debug: bool = False,
) -> Dict[int, TrackDBEntry]:

    crops_dir = Path(crops_dir)
    output_path = Path(output_path)
    crops_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = build_detector(
            use_buffalo=True,
            model_pack_name=config.INSIGHTFACE_MODEL_PACK,
            input_size=config.INSIGHTFACE_INPUT_SIZE,
            conf_thresh=config.INSIGHTFACE_CONF_THRESH,
            ctx_id=config.INSIGHTFACE_CTX_ID,
        )

    if tracker is None:
        tracker = build_tracker(
            track_thresh=config.BYTETRACK_TRACK_THRESH,
            high_thresh=config.BYTETRACK_HIGH_THRESH,
            match_thresh=config.BYTETRACK_MATCH_THRESH,
            max_time_lost=config.BYTETRACK_MAX_TIME_LOST,
        )
        
    meta = get_video_meta(video_path)

    logger.info(
        "PASS1 시작 | 영상: %s | %.1f fps | %d frames",
        Path(video_path).name,
        meta.fps,
        meta.total_frames,
    )

    track_db: Dict[int, TrackDBEntry] = {}
    crop_candidates: CropCandidates = {}

    for frame_idx, frame in iter_frames(video_path):
        detections = detector.detect(frame)

        for d in detections:
            d.frame_idx = frame_idx

        tracks = tracker.update(detections, frame_idx)

        if frame_idx % 30 == 0:
            logger.info(
                "frame %d | detections=%d | tracks=%d",
                frame_idx,
                len(detections),
                len(tracks),
            )

        for tr in tracks:
            _update_track_db(track_db, tr)

        if frame_idx % repr_interval == 0:
            for tr in tracks:
                _collect_crop_candidate(
                    crop_candidates,
                    tr,
                    frame,
                    min_crop_size,
                )

        if debug:
            _draw_debug(frame, tracks, frame_idx)
            cv2.imshow("PASS1 debug", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if debug:
        cv2.destroyAllWindows()

    for entry in track_db.values():
        entry.duration = len(entry.frames)

    track_db = _filter_short_tracks(
        track_db,
        min_frames=config.MIN_TRACK_FRAMES,
    )

    _save_repr_crops(
        track_db,
        crop_candidates,
        crops_dir,
        crop_quality,
    )

    save_json(
        {str(k): v.to_dict() for k, v in track_db.items()},
        output_path,
    )

    logger.info(
        "PASS1 완료 | track 수: %d | 저장: %s",
        len(track_db),
        output_path,
    )

    return track_db


def _update_track_db(
        track_db: Dict[int, TrackDBEntry],
        tr,
    ) -> None:
        tid = tr.track_id

        if tid not in track_db:
            track_db[tid] = TrackDBEntry(
                track_id=tid,
                start_frame=tr.frame_idx,
            )

        entry = track_db[tid]

        if not entry.frames or entry.frames[-1] != tr.frame_idx:
            entry.frames.append(tr.frame_idx)
            entry.bboxes.append(list(tr.bbox))

        entry.end_frame = max(entry.end_frame, tr.frame_idx)

        # PASS2 clustering을 위해 track 대표 embedding 저장
        if tr.embedding is not None and entry.embedding is None:
            entry.embedding = tr.embedding


def _collect_crop_candidate(
    crop_candidates: CropCandidates,
    tr,
    frame: np.ndarray,
    min_crop_size: int,
) -> None:
    tid = tr.track_id
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


def _filter_short_tracks(
    track_db: Dict[int, TrackDBEntry],
    min_frames: int,
) -> Dict[int, TrackDBEntry]:
    filtered: Dict[int, TrackDBEntry] = {}

    for tid, entry in track_db.items():
        if len(entry.frames) >= min_frames:
            filtered[tid] = entry

    return filtered


def _save_repr_crops(
    track_db: Dict[int, TrackDBEntry],
    crop_candidates: CropCandidates,
    crops_dir: Path,
    quality: int,
) -> None:
    for tid, entry in track_db.items():
        candidates = crop_candidates.get(tid, [])
        best_crop = select_best_crop(candidates)

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
        frame,
        f"frame {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 128, 255),
            2,
        )

        cv2.putText(
            frame,
            f"id:{tr.track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 255),
            2,
        )