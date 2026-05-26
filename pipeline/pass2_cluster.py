# pipeline/pass2_cluster.py

from __future__ import annotations

import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

import config
from db.schema import PersonDBEntry, TrackDBEntry
from utils.io import load_json, save_json

import cv2
from models.face_recognizer import build_recognizer

logger = logging.getLogger(__name__)


def run_pass2(
    track_db_path: str | Path = config.TRACK_DB_PATH,
    person_db_path: str | Path = config.PERSON_DB_PATH,
    person_crops_dir: str | Path | None = None,
    eps: float = config.DBSCAN_EPS,
    min_samples: int = config.DBSCAN_MIN_SAMPLES,
    top_n: int = config.TOP_N,
) -> Dict[str, PersonDBEntry]:

    track_db_path = Path(track_db_path)
    person_db_path = Path(person_db_path)

    if person_crops_dir is None:
        person_crops_dir = config.OUTPUT_DIR / "person_crops"

    person_crops_dir = Path(person_crops_dir)
    person_crops_dir.mkdir(parents=True, exist_ok=True)

    raw_track_db = load_json(track_db_path)

    track_db: Dict[int, TrackDBEntry] = {
        int(tid): TrackDBEntry.from_dict(entry)
        for tid, entry in raw_track_db.items()
    }

    recognizer = build_recognizer(
        model_pack_name=config.INSIGHTFACE_MODEL_PACK,
        ctx_id=config.INSIGHTFACE_CTX_ID,
    )

    embedding_extract_time = 0.0
    embedding_success = 0
    embedding_fail = 0

    valid_tracks: List[TrackDBEntry] = []

    for entry in track_db.values():
        if entry.repr_crop_path is None:
            logger.warning("track %d: repr_crop_path 없음. embedding 추출 제외", entry.track_id)
            continue

        crop_path = Path(entry.repr_crop_path)

        if not crop_path.exists():
            logger.warning("track %d: crop 파일 없음: %s", entry.track_id, crop_path)
            continue

        crop = cv2.imread(str(crop_path))

        if crop is None:
            logger.warning("track %d: crop 읽기 실패: %s", entry.track_id, crop_path)
            continue

        t0 = time.perf_counter()

        embedding = recognizer.get_embedding(crop)

        embedding_extract_time += time.perf_counter() - t0

        if embedding is None:
            embedding_fail += 1
            logger.warning("track %d: embedding 추출 실패", entry.track_id)
            continue

        embedding_success += 1

        entry.embedding = embedding
        track_db[entry.track_id].embedding = embedding
        valid_tracks.append(entry)

    if not valid_tracks:
        logger.warning(
            "embedding이 있는 track이 없습니다. pass1에서 embedding 저장 여부를 확인하세요."
        )
        save_json({}, person_db_path)
        return {}

    embeddings = np.array(
        [entry.embedding for entry in valid_tracks],
        dtype=np.float32,
    )

    embeddings = _l2_normalize(embeddings)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
    )

    labels = clustering.fit_predict(embeddings)

    person_db: Dict[str, PersonDBEntry] = {}

    for entry, label in zip(valid_tracks, labels):
        if label >= 0:
            person_id = f"person_{int(label):03d}"
        else:
            person_id = f"noise_{entry.track_id:04d}"

        entry.person_id = person_id
        track_db[entry.track_id].person_id = person_id

        if person_id not in person_db:
            person_db[person_id] = PersonDBEntry(
                person_id=person_id,
                track_ids=[],
                start_frame=entry.start_frame,
                end_frame=entry.end_frame,
                total_frames=0,
                repr_image=None,
                is_main=False,
            )

        person = person_db[person_id]
        person.track_ids.append(entry.track_id)
        person.start_frame = min(person.start_frame, entry.start_frame)
        person.end_frame = max(person.end_frame, entry.end_frame)
        person.total_frames += entry.duration

    sorted_persons = sorted(
        person_db.values(),
        key=lambda p: p.total_frames,
        reverse=True,
    )

    for rank, person in enumerate(sorted_persons):
        person.is_main = rank < top_n

    logger.info("PASS2 person summary:")

    for person in sorted_persons:
        logger.info(
            (
                "person_id=%s | tracks=%s | total_frames=%d | "
                "start=%d | end=%d | is_main=%s"
            ),
            person.person_id,
            person.track_ids,
            person.total_frames,
            person.start_frame,
            person.end_frame,
            person.is_main,
        )
    for person in person_db.values():
        best_track = _select_best_track_for_person(person, track_db)

        if best_track is None or best_track.repr_crop_path is None:
            logger.warning("%s: 대표 crop 없음", person.person_id)
            continue

        src = Path(best_track.repr_crop_path)

        if not src.exists():
            logger.warning("%s: crop 파일 없음: %s", person.person_id, src)
            continue

        dst = person_crops_dir / f"{person.person_id}.jpg"
        shutil.copyfile(src, dst)
        person.repr_image = str(dst)

    save_json(
        {pid: person.to_dict() for pid, person in person_db.items()},
        person_db_path,
    )

    save_json(
        {str(tid): entry.to_dict() for tid, entry in track_db.items()},
        track_db_path,
    )

    logger.info(
        (
            "PASS2 embedding summary | "
            "success=%d | fail=%d | avg_embedding=%.3fs"
        ),
        embedding_success,
        embedding_fail,
        embedding_extract_time / max(embedding_success, 1),
    )

    return person_db


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def _select_best_track_for_person(
    person: PersonDBEntry,
    track_db: Dict[int, TrackDBEntry],
) -> TrackDBEntry | None:

    candidates = [
        track_db[tid]
        for tid in person.track_ids
        if tid in track_db and track_db[tid].repr_crop_path is not None
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda e: e.duration,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s : %(message)s",
    )

    run_pass2()