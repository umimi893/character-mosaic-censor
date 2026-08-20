from __future__ import annotations

from dataclasses import replace

from .anatomy_filter import AnatomyFilterResult, AnatomySuppression
from .experience_store import ExperienceStore, candidate_crop, candidate_fingerprint
from .types import CandidateEvidence, Detection


class NegativeMemoryDetector:
    """Use repeated GOLD hard-negatives as a conservative visual memory.

    This is intentionally not a free-running self-trained model. It only vetoes
    a kept candidate when the compact crop is nearly identical to several GOLD
    negatives already accumulated from strong body/face evidence. Any pelvis or
    directional groin evidence disables the memory veto.
    """

    def __init__(self, detector, *, enabled: bool = True, store: ExperienceStore | None = None):
        self.detector = detector
        self.enabled = bool(enabled)
        self.store = store if store is not None else (ExperienceStore() if enabled else None)
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
        if not self.enabled or self.store is None or result is None or not getattr(result, "evidence", None):
            self.last_filter_result = result or AnatomyFilterResult(tuple(detections), status="not_run")
            return detections
        final = apply_negative_memory(result, image, self.store)
        self.last_filter_result = final
        return list(final.kept)

    def __getattr__(self, name: str):
        return getattr(self.detector, name)


def apply_negative_memory(result: AnatomyFilterResult, image, store: ExperienceStore) -> AnatomyFilterResult:
    existing_suppressed = {item.detection: item for item in result.suppressed}
    kept: list[Detection] = []
    suppressed: list[AnatomySuppression] = []
    evidence_out: list[CandidateEvidence] = []

    for evidence in result.evidence:
        detection = evidence.detection
        previous = existing_suppressed.get(detection)
        if previous is not None or evidence.decision == "suppress":
            if previous is not None:
                suppressed.append(previous)
            evidence_out.append(evidence)
            continue

        protected = any(
            signal.startswith(("near_pelvis:", "inside_groin_zone:"))
            for signal in evidence.positive_signals
        )
        if protected:
            kept.append(detection)
            evidence_out.append(evidence)
            continue

        crop = candidate_crop(image, detection)
        fingerprint = candidate_fingerprint(crop)
        matches = store.close_negative_matches(fingerprint, max_hamming=4, limit=200)

        # Three near-identical GOLD negatives are enough only for a moderate
        # detector score. Five or more are treated as strong repeated evidence.
        should_suppress = matches >= 5 or (matches >= 3 and detection.score < 0.65)
        if should_suppress:
            negative = tuple(evidence.negative_signals) + (f"negative_memory:{matches}",)
            final = replace(evidence, decision="suppress", negative_signals=negative)
            evidence_out.append(final)
            suppressed.append(
                AnatomySuppression(
                    detection=detection,
                    reason="known_negative_memory",
                    person_index=(evidence.matched_persons[0] if evidence.matched_persons else -1),
                    joint_distance_ratio=0.0,
                    pelvis_distance_ratio=float(
                        evidence.pelvis_distance_ratio
                        if evidence.pelvis_distance_ratio is not None
                        else 999.0
                    ),
                )
            )
            continue

        kept.append(detection)
        evidence_out.append(evidence)

    return replace(
        result,
        kept=tuple(kept),
        suppressed=tuple(suppressed),
        evidence=tuple(evidence_out),
    )
