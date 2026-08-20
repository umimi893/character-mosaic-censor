from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    label: str
    score: float
    source: str = "full"


@dataclass(frozen=True)
class BodyRegion:
    box: tuple[int, int, int, int]
    kind: str
    score: float = 1.0
    person_index: int = -1
    source: str = ""


@dataclass(frozen=True)
class PosePoint:
    x: float
    y: float
    score: float
    label: str
    person_index: int


@dataclass(frozen=True)
class PoseEdge:
    start: tuple[float, float]
    end: tuple[float, float]
    person_index: int
    label: str = ""


@dataclass(frozen=True)
class CandidateEvidence:
    detection: Detection
    decision: str = "keep"  # keep | review | suppress
    positive_signals: tuple[str, ...] = tuple()
    negative_signals: tuple[str, ...] = tuple()
    matched_persons: tuple[int, ...] = tuple()
    pelvis_distance_ratio: float | None = None


@dataclass(frozen=True)
class PreviewFrame:
    """One UI preview update emitted by the processing pipeline.

    ``image`` may be a downscaled display copy. ``coordinate_size`` is the
    coordinate system used by detections/censor boxes, normally the processed
    image's full resolution. Keeping those separate prevents 3K/8K images from
    being copied through Qt on every detector pass while preserving exact box
    placement in the monitor.
    """

    stage: str
    source: Path
    image: "Image.Image"
    detections: tuple[Detection, ...] = tuple()
    censor_boxes: tuple[tuple[int, int, int, int], ...] = tuple()
    status: str = ""
    coordinate_size: tuple[int, int] | None = None
    body_regions: tuple[BodyRegion, ...] = tuple()
    pose_points: tuple[PosePoint, ...] = tuple()
    pose_edges: tuple[PoseEdge, ...] = tuple()
    candidate_evidence: tuple[CandidateEvidence, ...] = tuple()
    analysis_status: str = ""


@dataclass(frozen=True)
class ProcessResult:
    source: Path
    output: Path | None
    detections: tuple[Detection, ...]
    review_required: bool
    censor_boxes: tuple[tuple[int, int, int, int], ...] = tuple()
    skipped: bool = False
    cancelled: bool = False
    error: str | None = None
    fatal_error: bool = False
    review_path: Path | None = None
    count_mismatch: bool = False
    manual_review_path: Path | None = None
    elapsed_seconds: float = 0.0
    anatomy_suppressed: tuple[Detection, ...] = tuple()
    anatomy_suppression_reasons: tuple[str, ...] = tuple()
    anatomy_filter_status: str = ""
    body_regions: tuple[BodyRegion, ...] = tuple()
    pose_points: tuple[PosePoint, ...] = tuple()
    pose_edges: tuple[PoseEdge, ...] = tuple()
    candidate_evidence: tuple[CandidateEvidence, ...] = tuple()
