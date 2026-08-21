from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from PIL import Image

from .types import CandidateEvidence, Detection


# These are diagnostic groups, not production thresholds.  The shadow probe is
# deliberately non-authoritative until real user data demonstrates separation.
_GENITAL_TAGS = (
    "pussy",
    "spread_pussy",
    "pussy_juice",
    "pussy_peek",
    "clitoris",
    "labia",
    "vulva",
    "cameltoe",
)
_ARMPIT_TAGS = (
    "armpits",
    "armpit",
)
_BODY_CONTEXT_TAGS = (
    "stomach",
    "navel",
    "back",
    "lower_back",
    "thighs",
    "knees",
    "feet",
    "breasts",
    "ass",
)


@dataclass(frozen=True)
class SemanticProbeResult:
    source: str
    box: tuple[int, int, int, int]
    detector_source: str
    detector_score: float
    current_decision: str
    positive_signals: tuple[str, ...]
    negative_signals: tuple[str, ...]
    matched_persons: tuple[int, ...]
    pelvis_distance_ratio: float | None
    crop_box: tuple[int, int, int, int]
    rating: dict[str, float]
    relevant_tags: dict[str, float]
    top_tags: tuple[tuple[str, float], ...]

    @property
    def genital_score(self) -> float:
        return max((self.relevant_tags.get(tag, 0.0) for tag in _GENITAL_TAGS), default=0.0)

    @property
    def armpit_score(self) -> float:
        return max((self.relevant_tags.get(tag, 0.0) for tag in _ARMPIT_TAGS), default=0.0)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "box": list(self.box),
            "detector_source": self.detector_source,
            "detector_score": self.detector_score,
            "current_decision": self.current_decision,
            "positive_signals": list(self.positive_signals),
            "negative_signals": list(self.negative_signals),
            "matched_persons": list(self.matched_persons),
            "pelvis_distance_ratio": self.pelvis_distance_ratio,
            "crop_box": list(self.crop_box),
            "rating": self.rating,
            "relevant_tags": self.relevant_tags,
            "genital_score": self.genital_score,
            "armpit_score": self.armpit_score,
            "top_tags": [[tag, score] for tag, score in self.top_tags],
        }


def candidate_context_crop(
    image: Image.Image,
    detection: Detection,
    *,
    scale: float = 3.5,
    min_side: int = 256,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return a square-ish context crop around one detector candidate.

    The crop is intentionally much wider than the raw detector box.  A tiny
    crease-only crop can make an armpit and a genital target look deceptively
    similar; surrounding anatomy is the semantic verifier's main advantage.
    """

    x0, y0, x1, y1 = (float(value) for value in detection.box)
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    side = max(float(min_side), max(width, height) * max(1.0, float(scale)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))

    image_w, image_h = image.size
    left = max(0, left)
    top = max(0, top)
    right = min(image_w, right)
    bottom = min(image_h, bottom)
    if right <= left:
        right = min(image_w, left + 1)
    if bottom <= top:
        bottom = min(image_h, top + 1)

    crop_box = (left, top, right, bottom)
    return image.crop(crop_box), crop_box


def summarize_wd14(
    rating: Mapping[str, float],
    features: Mapping[str, float],
    *,
    top_k: int = 20,
) -> tuple[dict[str, float], dict[str, float], tuple[tuple[str, float], ...]]:
    rating_out = {str(key): float(value) for key, value in rating.items()}
    feature_out = {str(key): float(value) for key, value in features.items()}
    relevant_names = tuple(dict.fromkeys(_GENITAL_TAGS + _ARMPIT_TAGS + _BODY_CONTEXT_TAGS))
    relevant = {name: float(feature_out.get(name, 0.0)) for name in relevant_names}
    top = tuple(
        sorted(feature_out.items(), key=lambda item: item[1], reverse=True)[: max(1, int(top_k))]
    )
    return rating_out, relevant, top


def probe_evidence(
    source: Path | str,
    image: Image.Image,
    evidence: CandidateEvidence,
    *,
    tagger: Callable | None = None,
    crop_scale: float = 3.5,
    min_crop_side: int = 256,
    top_k: int = 20,
) -> SemanticProbeResult:
    """Run WD14 on one candidate without changing the production decision."""

    if tagger is None:
        from imgutils.tagging import get_wd14_tags

        tagger = get_wd14_tags

    crop, crop_box = candidate_context_crop(
        image,
        evidence.detection,
        scale=crop_scale,
        min_side=min_crop_side,
    )
    rating, features = tagger(
        crop,
        general_threshold=0.05,
        character_threshold=1.0,
        fmt=("rating", "general"),
    )
    rating_out, relevant, top = summarize_wd14(rating, features, top_k=top_k)
    return SemanticProbeResult(
        source=str(source),
        box=tuple(int(value) for value in evidence.detection.box),
        detector_source=str(evidence.detection.source),
        detector_score=float(evidence.detection.score),
        current_decision=str(evidence.decision),
        positive_signals=tuple(evidence.positive_signals),
        negative_signals=tuple(evidence.negative_signals),
        matched_persons=tuple(int(value) for value in evidence.matched_persons),
        pelvis_distance_ratio=(
            float(evidence.pelvis_distance_ratio)
            if evidence.pelvis_distance_ratio is not None
            else None
        ),
        crop_box=crop_box,
        rating=rating_out,
        relevant_tags=relevant,
        top_tags=top,
    )


def interesting_tag_names() -> Sequence[str]:
    return tuple(dict.fromkeys(_GENITAL_TAGS + _ARMPIT_TAGS + _BODY_CONTEXT_TAGS))
