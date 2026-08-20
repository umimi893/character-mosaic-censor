from __future__ import annotations

from dataclasses import replace

from .anatomy_filter import AnatomyFilterResult, AnatomySuppression
from .experience_store import (
    ExperienceStore,
    candidate_crop,
    candidate_fingerprint,
    fingerprint_hamming,
)
from .types import CandidateEvidence, Detection


class NegativeMemoryDetector:
    """Use repeated GOLD hard-negatives as a conservative visual memory.

    This is intentionally not a free-running self-trained model. It only vetoes
    a kept candidate when the compact crop is extremely close to many distinct
    GOLD negative fingerprints accumulated from independent anatomical
    evidence. Pelvis/groin evidence, missing person association, or a very high
    detector score keeps the current candidate instead.
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
        # Visual hashes are intentionally never allowed to make a decision when
        # body association is missing or when the candidate is still reasonably
        # close to a pelvis despite not crossing the explicit KEEP threshold.
        near_pelvis_numeric = (
            evidence.pelvis_distance_ratio is not None
            and float(evidence.pelvis_distance_ratio) < 0.60
        )
        if (
            protected
            or near_pelvis_numeric
            or not evidence.matched_persons
            or detection.score >= 0.85
        ):
            kept.append(detection)
            evidence_out.append(evidence)
            continue

        crop = candidate_crop(image, detection)
        fingerprint = candidate_fingerprint(crop)
        matches = _distinct_close_negative_matches(
            store,
            fingerprint,
            max_hamming=4,
            limit=600,
        )

        # Perceptual hashing is only a memory aid, not semantic understanding.
        # Require multiple *distinct* near fingerprints so duplicate/re-encoded
        # copies cannot create false confidence by sheer repetition.
        should_suppress = matches >= 8 or (matches >= 5 and detection.score < 0.45)
        if should_suppress:
            negative = tuple(evidence.negative_signals) + (f"negative_memory:{matches}",)
            final = replace(evidence, decision="suppress", negative_signals=negative)
            evidence_out.append(final)
            suppressed.append(
                AnatomySuppression(
                    detection=detection,
                    reason="known_negative_memory",
                    person_index=evidence.matched_persons[0],
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


def _distinct_close_negative_matches(
    store: ExperienceStore,
    fingerprint: str,
    *,
    max_hamming: int,
    limit: int,
) -> int:
    if not fingerprint:
        return 0
    # Prefix indexing deliberately biases toward false negatives (missed memory
    # matches) rather than scanning the entire database and risking a broad
    # accidental visual match. That is the safer failure direction here.
    prefix = fingerprint[:4]
    with store.connect() as db:
        rows = db.execute(
            """
            SELECT DISTINCT fingerprint
            FROM candidates
            WHERE fingerprint_prefix=?
              AND pseudo_label='negative'
              AND quality_tier='gold'
              AND fingerprint IS NOT NULL
            LIMIT ?
            """,
            (prefix, int(limit)),
        ).fetchall()
    return sum(
        1
        for row in rows
        if fingerprint_hamming(fingerprint, str(row["fingerprint"])) <= max_hamming
    )
