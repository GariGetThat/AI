# db/schema.py
"""track_db / person_db 스키마 정의 및 생성 헬퍼"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ─── Detection Record ────────────────────────────────────
@dataclass
class DetectionRecord:
    frame_idx: int
    bbox: List[float]
    score: float
    kps: Optional[List[List[float]]] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Track Record (프레임별) ─────────────────────────────
@dataclass
class TrackRecord:
    frame_idx: int
    track_id: int
    bbox: List[float]
    score: float
    kps: Optional[List[List[float]]] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Track DB Entry (track 단위 요약) ────────────────────
@dataclass
class TrackDBEntry:
    track_id: int
    frames: List[int]           = field(default_factory=list)
    bboxes: List[List[float]]   = field(default_factory=list)
    start_frame: int            = 0
    end_frame: int              = 0
    duration: int               = 0
    repr_crop_path: Optional[str] = None
    embedding: Optional[List[float]] = None   # track 대표 embedding
    person_id: Optional[str] = None           # PASS2 clustering 후 채움

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackDBEntry":
        return cls(**d)


# ─── Person DB Entry ─────────────────────────────────────
@dataclass
class PersonDBEntry:
    person_id: str
    track_ids: List[int]        = field(default_factory=list)
    start_frame: int            = 0
    end_frame: int              = 0
    total_frames: int           = 0
    repr_image: Optional[str]   = None
    is_main: bool               = False   # Top-N 여부

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PersonDBEntry":
        return cls(**d)


# ─── SAM2 Input Entry ────────────────────────────────────
@dataclass
class SAM2InputEntry:
    id: str
    type: str                   # "face"
    start_frame: int
    end_frame: int
    bbox: List[float]           # start_frame 시점의 bbox

    def to_dict(self) -> dict:
        return asdict(self)