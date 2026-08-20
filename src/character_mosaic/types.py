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
