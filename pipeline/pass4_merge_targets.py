# pipeline/export_for_sam2.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import config
from db.schema import PersonDBEntry, SAM2InputEntry, TrackDBEntry
from utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def export_for_sam2(
    track_db_path: str | Path = config.TRACK_DB_PATH,
    person_db_path: str | Path = config.PERSON_DB_PATH,
    output_path: str | Path = config.SAM2_INPUT_PATH,
    blur_main_person: bool = False,
) -> List[SAM2InputEntry]:
    """
    SAM2 입력 JSON 생성.

    Parameters
    ----------
    blur_main_person:
        False이면 Top-N 주요 인물은 blur 대상에서 제외.
        True이면 주요 인물도 포함해서 전부 blur 대상.
    """

    track_db_path = Path(track_db_path)
    person_db_path = Path(person_db_path)
    output_path = Path(output_path)

    raw_track_db = load_json(track_db_path)
    raw_person_db = load_json(person_db_path)

    track_db: Dict[int, TrackDBEntry] = {
        int(tid): TrackDBEntry.from_dict(entry)
        for tid, entry in raw_track_db.items()
    }

    person_db: Dict[str, PersonDBEntry] = {
        pid: PersonDBEntry.from_dict(entry)
        for pid, entry in raw_person_db.items()
    }

    blur_person_ids = _select_blur_person_ids(
        person_db=person_db,
        blur_main_person=blur_main_person,
    )

    sam2_inputs: List[SAM2InputEntry] = []

    for track in track_db.values():
        if track.person_id is None:
            continue

        if track.person_id not in blur_person_ids:
            continue

        if not track.frames or not track.bboxes:
            continue

        start_frame = track.frames[0]
        end_frame = track.frames[-1]
        start_bbox = track.bboxes[0]

        sam2_inputs.append(
            SAM2InputEntry(
                id=f"{track.person_id}_track_{track.track_id:04d}",
                type="face",
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                bbox=[float(v) for v in start_bbox],
            )
        )

    save_json(
        [entry.to_dict() for entry in sam2_inputs],
        output_path,
    )

    logger.info(
        "SAM2 input export 완료 | 대상 track 수=%d | 저장=%s",
        len(sam2_inputs),
        output_path,
    )

    return sam2_inputs


def _select_blur_person_ids(
    person_db: Dict[str, PersonDBEntry],
    blur_main_person: bool = False,
) -> set[str]:
    if blur_main_person:
        return set(person_db.keys())

    return {
        person_id
        for person_id, person in person_db.items()
        if not person.is_main
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s : %(message)s",
    )

    export_for_sam2()