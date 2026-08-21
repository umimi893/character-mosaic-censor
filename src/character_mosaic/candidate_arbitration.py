from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .anatomy_filter import AnatomyFilterResult, AnatomySuppression
from .types import CandidateEvidence, Detection


# Production-only arbitration for a pattern observed across a 1,400-image
# regression run: one strong, full-frame-confirmed target plus a spatially
# separate weak auxiliary-only false positive on the same person.
#
# This is deliberately narrower than a generic confidence threshold. A weak
# candidate is suppressed only when an already-kept full-frame anchor exists,
# the anatomy map looks like a single-person scene, and no pelvis/groin signal
# protects the weak candidate.
_ANCHOR_MIN_SCORE = 0.60
_WEAK_AUX_MAX_SCORE = 0.50
_MIN_SCORE_GAP = 0.12
_MIN_SAFE_PELVIS_RATIO = 0.50
_MAX_ANCHOR_OVERLAP = 0.20
_REASON = "weaker_aux_same_person_with_full_anchor"


class CandidateArbitrationDetector:
    """Remove weak remote auxiliary candidates when a strong anchor exists.

    Geometry v2 runs before this layer, so directional groin and cross-person
    pelvis protection have already had a chance to mark candidates. Any such
    positive signal prevents arbitration. Multi-person anatomy maps fail open.
    """

    def __init__(self, detector):
        self.detector = detector
        self.last_filter_result = getattr(detector, "last_filter_result", AnatomyFilterResult(tuple()))

    @property
    def requires_review(self) -> bool:
        return bool(getattr(self.last_filter_result, "requires_review", False))

    def reset_filter_state(self) -> None:
        reset = getattr(self.detector, "reset_filter_state", None)
        if callable(reset):
            reset()
        self.last_filter_result = getattr(self.detector, "last_filter_result", AnatomyFilterResult(tuple()))

    def detect(self, image, progress=None, stop_requested=None):
        detections = list(self.detector.detect(image, progress=progress, stop_requested=stop_requested))
        result = getattr(self.detector, "last_filter_result", None)
        if result is None or not getattr(result, "evidence", None):
            self.last_filter_result = result or AnatomyFilterResult(tuple(detections), status="not_run")
            return detections
        final = apply_candidate_arbitration(result)
        self.last_filter_result = final
        return list(final.kept)

    def __getattr__(self, name: str):
        return getattr(self.detector, name)


def apply_candidate_arbitration(result: AnatomyFilterResult) -> AnatomyFilterResult:
    """Suppress only weak auxiliary extras next to a strong same-person anchor."""

    if len(result.kept) < 2 or not result.evidence:
        return result

    # Close-contact/multi-person scenes remain fail-open. Requiring one person
    # region and at most one head also protects against a person detector that
    # split the image correctly but candidates happened to share an index.
    person_regions = [region for region in result.body_regions if region.kind == "person"]
    head_regions = [region for region in result.body_regions if region.kind == "head"]
    if len(person_regions) != 1 or len(head_regions) > 1:
        return result

    kept_set = set(result.kept)
    active = [
        evidence
        for evidence in result.evidence
        if evidence.decision == "keep" and evidence.detection in kept_set
    ]
    if len(active) < 2:
        return result

    anchors: dict[int, CandidateEvidence] = {}
    for evidence in active:
        if len(evidence.matched_persons) != 1:
            continue
        if evidence.negative_signals:
            continue
        detection = evidence.detection
        if detection.score < _ANCHOR_MIN_SCORE or not _source_has_full(detection.source):
            continue
        person_index = int(evidence.matched_persons[0])
        previous = anchors.get(person_index)
        if previous is None or detection.score > previous.detection.score:
            anchors[person_index] = evidence

    if not anchors:
        return result

    decisions: dict[Detection, tuple[AnatomySuppression, str]] = {}
    for evidence in active:
        if len(evidence.matched_persons) != 1:
            continue
        person_index = int(evidence.matched_persons[0])
        anchor = anchors.get(person_index)
        if anchor is None or anchor.detection == evidence.detection:
            continue

        detection = evidence.detection
        if not _source_is_auxiliary_only(detection.source):
            continue
        if detection.score > _WEAK_AUX_MAX_SCORE:
            continue
        score_gap = anchor.detection.score - detection.score
        if score_gap < _MIN_SCORE_GAP:
            continue
        if evidence.negative_signals:
            continue
        if not _detector_only_positive(evidence.positive_signals):
            continue
        if (
            evidence.pelvis_distance_ratio is not None
            and evidence.pelvis_distance_ratio < _MIN_SAFE_PELVIS_RATIO
        ):
            continue
        if _intersection_over_smaller(detection.box, anchor.detection.box) > _MAX_ANCHOR_OVERLAP:
            continue

        strength = max(0.0, min(1.0, score_gap / max(anchor.detection.score, 1e-9)))
        signal = f"{_REASON}:p{person_index}:{strength:.3f}"
        suppression = AnatomySuppression(
            detection=detection,
            reason=_REASON,
            person_index=person_index,
            joint_distance_ratio=max(0.0, 1.0 - strength),
            pelvis_distance_ratio=float(
                evidence.pelvis_distance_ratio
                if evidence.pelvis_distance_ratio is not None
                else 999.0
            ),
        )
        decisions[detection] = (suppression, signal)

    if not decisions:
        return result

    evidence_out: list[CandidateEvidence] = []
    new_suppressions: list[AnatomySuppression] = []
    for evidence in result.evidence:
        decision = decisions.get(evidence.detection)
        if decision is None:
            evidence_out.append(evidence)
            continue
        suppression, signal = decision
        evidence_out.append(
            replace(
                evidence,
                decision="suppress",
                negative_signals=tuple(evidence.negative_signals) + (signal,),
            )
        )
        new_suppressions.append(suppression)

    return replace(
        result,
        kept=tuple(detection for detection in result.kept if detection not in decisions),
        suppressed=tuple(result.suppressed) + tuple(new_suppressions),
        evidence=tuple(evidence_out),
    )


def _source_parts(source: str) -> tuple[str, ...]:
    return tuple(part for part in str(source).split("+") if part)


def _source_has_full(source: str) -> bool:
    return "full" in _source_parts(source)


def _source_is_auxiliary_only(source: str) -> bool:
    parts = _source_parts(source)
    return bool(parts) and "full" not in parts and all(
        part.startswith(("tile_", "retry_")) for part in parts
    )


def _detector_only_positive(signals: Sequence[str]) -> bool:
    return bool(signals) and all(signal.startswith("detector:") for signal in signals)


def _intersection_over_smaller(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / min(area_a, area_b)
