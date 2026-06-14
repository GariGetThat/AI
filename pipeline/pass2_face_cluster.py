# pipeline/pass2_face_cluster.py

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import config
from db.schema import PersonDBEntry, TrackDBEntry
from models.face_recognizer import build_recognizer
from utils.io import load_json, save_json, read_image

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
        t0 = time.perf_counter()

        embedding = _extract_track_embedding(
            entry=entry,
            recognizer=recognizer,
            top_k=config.TRACK_EMBEDDING_TOP_K,
            min_embeddings=config.TRACK_MIN_EMBEDDINGS,
        )

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
        logger.warning("embedding을 추출한 track이 없습니다.")
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

    person_db = _merge_short_persons(
        person_db=person_db,
        track_db=track_db,
        min_person_frames=config.MIN_PERSON_FRAMES,
        merge_dist_thresh=config.PERSON_MERGE_SIM_THRESH,
    )

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

    logger.info(
        "PASS2 완료 | person 수=%d | 저장=%s",
        len(person_db),
        person_db_path,
    )

    return person_db


def _merge_short_persons(
    person_db: Dict[str, PersonDBEntry],
    track_db: Dict[int, TrackDBEntry],
    min_person_frames: int,
    merge_dist_thresh: float,
) -> Dict[str, PersonDBEntry]:

    long_persons = {
        pid: p
        for pid, p in person_db.items()
        if p.total_frames >= min_person_frames
    }

    short_persons = {
        pid: p
        for pid, p in person_db.items()
        if p.total_frames < min_person_frames
    }

    for short_pid, short_person in short_persons.items():
        short_emb = _mean_embedding(short_person, track_db)

        if short_emb is None:
            logger.info("short person keep | %s | embedding 없음", short_pid)
            long_persons[short_pid] = short_person
            continue

        best_pid = None
        best_dist = float("inf")

        for long_pid, long_person in long_persons.items():
            long_emb = _mean_embedding(long_person, track_db)

            if long_emb is None:
                continue

            time_overlap = _time_overlap_ratio(short_person, long_person)

            if time_overlap > config.PERSON_MERGE_MAX_TIME_OVERLAP:
                logger.info(
                    "merge skip by time overlap | %s -> %s | overlap=%.3f",
                    short_pid,
                    long_pid,
                    time_overlap,
                )
                continue

            center_dist = _spatial_center_distance(
                short_person,
                long_person,
                track_db,
            )

            if center_dist > config.PERSON_MERGE_MAX_CENTER_DIST:
                logger.info(
                    "merge skip by spatial distance | %s -> %s | center_dist=%.3f",
                    short_pid,
                    long_pid,
                    center_dist,
                )
                continue

            dist = _cosine_distance(short_emb, long_emb)

            if dist < best_dist:
                best_dist = dist
                best_pid = long_pid

        if best_pid is not None and best_dist <= merge_dist_thresh:
            target = long_persons[best_pid]

            target.track_ids.extend(short_person.track_ids)
            target.start_frame = min(target.start_frame, short_person.start_frame)
            target.end_frame = max(target.end_frame, short_person.end_frame)
            target.total_frames += short_person.total_frames

            for tid in short_person.track_ids:
                if tid in track_db:
                    track_db[tid].person_id = best_pid

            logger.info(
                "short person merge | %s -> %s | cosine_dist=%.3f",
                short_pid,
                best_pid,
                best_dist,
            )
        else:
            logger.info(
                "short person keep | %s | best=%s | cosine_dist=%.3f",
                short_pid,
                best_pid,
                best_dist,
            )
            long_persons[short_pid] = short_person

    return long_persons

def _extract_track_embedding(
    entry: TrackDBEntry,
    recognizer: Any,
    top_k: int,
    min_embeddings: int,
) -> List[float] | None:
    """
    track 하나에 대한 대표 embedding 생성.

    현재 단계:
    - 여러 crop 경로가 있으면 top_k개 평균
    - 없으면 기존 repr_crop_path 1개 사용

    이후 확장:
    - frame + kps가 TrackDB에 저장되면 recognizer.get_embedding(frame=..., kps=...)로 교체 가능
    """

    crop_paths = _collect_crop_paths(entry)

    if not crop_paths:
        logger.warning(
            "track %d: crop path 없음. embedding 추출 제외",
            entry.track_id,
        )
        return None

    embeddings: List[List[float]] = []

    for crop_path in crop_paths[:top_k]:
        crop = read_image(crop_path)

        if crop is None:
            logger.warning(
                "track %d: crop 읽기 실패: %s",
                entry.track_id,
                crop_path,
            )
            continue

        emb = recognizer.get_embedding(crop=crop)

        if emb is None:
            continue

        embeddings.append(emb)

    if len(embeddings) < min_embeddings:
        logger.warning(
            "track %d: embedding 개수 부족 | got=%d | required=%d",
            entry.track_id,
            len(embeddings),
            min_embeddings,
        )
        return None

    return _average_embedding_list(embeddings)


def _collect_crop_paths(entry: TrackDBEntry) -> List[Path]:
    """
    TrackDBEntry에서 사용할 crop path 목록 수집.

    현재 schema에서는 repr_crop_path만 있을 가능성이 크다.
    나중에 crop_paths / repr_crop_paths 같은 필드를 추가해도 바로 대응되게 작성.
    """

    paths: List[Path] = []

    for attr_name in ("crop_paths", "repr_crop_paths"):
        value = getattr(entry, attr_name, None)

        if not value:
            continue

        for p in value:
            path = Path(p)

            if path.exists():
                paths.append(path)

    if entry.repr_crop_path is not None:
        path = Path(entry.repr_crop_path)

        if path.exists():
            paths.append(path)
        else:
            logger.warning(
                "track %d: repr_crop_path 파일 없음: %s",
                entry.track_id,
                path,
            )

    # 중복 제거
    unique_paths = []
    seen = set()

    for path in paths:
        key = str(path)

        if key in seen:
            continue

        seen.add(key)
        unique_paths.append(path)

    return unique_paths


def _average_embedding_list(
    embeddings: List[List[float]],
) -> List[float] | None:
    """
    여러 embedding을 평균낸 뒤 다시 L2 normalize.
    """

    if not embeddings:
        return None

    arr = np.asarray(embeddings, dtype=np.float32)
    arr = _l2_normalize(arr)

    mean_emb = np.mean(arr, axis=0, keepdims=True)
    mean_emb = _l2_normalize(mean_emb)

    return mean_emb.reshape(-1).tolist()


def _mean_embedding(
    person: PersonDBEntry,
    track_db: Dict[int, TrackDBEntry],
) -> np.ndarray | None:

    embs = []

    for tid in person.track_ids:
        if tid not in track_db:
            continue

        emb = track_db[tid].embedding

        if emb is None:
            continue

        embs.append(emb)

    if not embs:
        return None

    embs = np.array(embs, dtype=np.float32)
    embs = _l2_normalize(embs)

    mean_emb = np.mean(embs, axis=0, keepdims=True)
    mean_emb = _l2_normalize(mean_emb)

    return mean_emb.reshape(-1)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


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

def _time_overlap_ratio(
    a: PersonDBEntry,
    b: PersonDBEntry,
) -> float:
    overlap_start = max(a.start_frame, b.start_frame)
    overlap_end = min(a.end_frame, b.end_frame)

    overlap = max(0, overlap_end - overlap_start + 1)

    short_duration = min(
        max(1, a.end_frame - a.start_frame + 1),
        max(1, b.end_frame - b.start_frame + 1),
    )

    return overlap / short_duration


def _person_center(
    person: PersonDBEntry,
    track_db: Dict[int, TrackDBEntry],
) -> np.ndarray | None:
    centers = []

    for tid in person.track_ids:
        if tid not in track_db:
            continue

        track = track_db[tid]

        for bbox in track.bboxes:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])

    if not centers:
        return None

    return np.mean(np.array(centers, dtype=np.float32), axis=0)


def _spatial_center_distance(
    a: PersonDBEntry,
    b: PersonDBEntry,
    track_db: Dict[int, TrackDBEntry],
) -> float:
    ca = _person_center(a, track_db)
    cb = _person_center(b, track_db)

    if ca is None or cb is None:
        return 0.0

    # 대략 1920x1080 기준 정규화 대신 bbox 좌표 자체 스케일 영향 줄이기
    # 현재 영상 해상도를 직접 모르면 diagonal을 넉넉히 2200으로 둠
    dist = np.linalg.norm(ca - cb)
    return float(dist / 2200.0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s : %(message)s",
    )

    run_pass2()