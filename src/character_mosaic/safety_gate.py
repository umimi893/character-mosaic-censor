from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .anatomy_filter import AnatomyFilterResult
from .types import CandidateEvidence, Detection


# Geometry/body-location heuristics are allowed to reject obvious false
# positives, but they must not be the final authority when an independently
# strong target signal exists immediately next to the same person's pelvis.
# This is a general recall safety gate, not a body-part-specific exception.
_RESCUABLE_BODY_REASONS = frozenset({
    "inside_torso_back",
    "inside_upper_back",
    "inside_torso",
    "near_right_armpit",
    "near_left_armpit",
    "near_right_armpit_v2",
    "near_left_armpit_v2",
    "on_right_thigh",
    "on_left_thigh",
})
_MIN_DETECTOR_SCORE = 0.55
_MAX_PELVIS_DISTANCE = 0.18
_SIGNAL = "strong_full_same_person_pelvis_safety"


class SafetyGateDetector:
    """Final recall guard after body geometry, before learned/memory vetoes."""

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
        final = apply_safety_gate(result)
        self.last_filter_result = final
        return list(final.kept)

    def __getattr__(self, name: str):
        return getattr(self.detector, name)


def apply_safety_gate(result: AnatomyFilterResult) -> AnatomyFilterResult:
    if not result.suppressed or not result.evidence:
        return result

    suppression_by_detection = {item.detection: item for item in result.suppressed}
    rescued: set[Detection] = set()
    evidence_out: list[CandidateEvidence] = []

    for evidence in result.evidence:
        suppression = suppression_by_detection.get(evidence.detection)
        if suppression is None or not _should_rescue(evidence, suppression.reason):
            evidence_out.append(evidence)
            continue

        matched_person = int(evidence.matched_persons[0])
        signal = f"{_SIGNAL}:p{matched_person}:{float(evidence.pelvis_distance_ratio):.3f}"
        positive = list(evidence.positive_signals)
        if signal not in positive:
            positive.append(signal)
        evidence_out.append(
            replace(evidence, decision="keep", positive_signals=tuple(positive))
        )
        rescued.add(evidence.detection)

    if not rescued:
        return result

    kept = list(result.kept)
    for evidence in evidence_out:
        if evidence.detection in rescued and evidence.detection not in kept:
            kept.append(evidence.detection)

    return replace(
        result,
        kept=tuple(kept),
        suppressed=tuple(item for item in result.suppressed if item.detection not in rescued),
        evidence=tuple(evidence_out),
    )


def _should_rescue(evidence: CandidateEvidence, reason: str) -> bool:
    if reason not in _RESCUABLE_BODY_REASONS:
        return False
    if len(evidence.matched_persons) != 1:
        return False
    if evidence.detection.score < _MIN_DETECTOR_SCORE:
        return False
    if "full" not in _source_parts(evidence.detection.source):
        return False
    if evidence.pelvis_distance_ratio is None or evidence.pelvis_distance_ratio > _MAX_PELVIS_DISTANCE:
        return False

    person = int(evidence.matched_persons[0])
    return person in _signal_people(evidence.positive_signals, "near_pelvis")


def _source_parts(source: str) -> tuple[str, ...]:
    return tuple(part for part in str(source).split("+") if part)


def _signal_people(signals: Sequence[str], prefix: str) -> set[int]:
    people: set[int] = set()
    needle = prefix + ":p"
    for signal in signals:
        if not signal.startswith(needle):
            continue
        text = signal[len(needle):].split(":", 1)[0]
        try:
            people.add(int(text))
        except ValueError:
            pass
    return people
