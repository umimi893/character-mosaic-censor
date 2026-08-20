from __future__ import annotations

from .types import CandidateEvidence


# GOLD is intentionally narrower than the runtime suppression policy. Runtime
# can suppress a candidate for product UX reasons (for example a final
# review_without_pelvis), but only independently strong spatial evidence is
# allowed to become trusted hard-negative memory/training material.


def classify_pseudo_label(
    evidence: CandidateEvidence,
    suppression_reason: str | None,
) -> tuple[str, str, str | None]:
    """Return ``(pseudo_label, quality_tier, negative_kind)`` conservatively."""
    if evidence.decision == "suppress":
        kind = _gold_negative_kind(suppression_reason)
        if kind is not None:
            return "negative", "gold", kind
        # Runtime policy suppressions without a strong independent anatomical
        # reason remain useful to inspect/train later, but never seed Negative
        # Memory as GOLD automatically.
        return "negative", "silver", _weak_negative_kind(suppression_reason)

    has_groin = any(
        signal.startswith(("near_pelvis:", "inside_groin_zone:"))
        for signal in evidence.positive_signals
    )
    has_negative = bool(evidence.negative_signals)
    if (
        evidence.decision == "keep"
        and has_groin
        and not has_negative
        and evidence.detection.score >= 0.75
    ):
        # Positive seeds are never GOLD in v1.4 because the legacy corpus is
        # noisy and no manually verified target annotation is assumed.
        return "positive_seed", "silver", None

    return "quarantine", "quarantine", None


def _gold_negative_kind(reason: str | None) -> str | None:
    text = (reason or "").strip().lower()
    if not text:
        return None

    # Geometry-v2 and the older pose rules are explicit body-location evidence.
    if "armpit" in text:
        return "armpit"
    if text in {"inside_upper_back", "inside_torso_back"}:
        return "back"
    if text == "inside_torso":
        return "torso"
    if "thigh" in text or "lower_leg" in text or "knee" in text:
        return "leg"

    # This anatomy suppression requires simultaneous eye + face + head overlap,
    # unlike face/head-only REVIEW evidence.
    if text == "inside_eye_face_head":
        return "face"
    return None


def _weak_negative_kind(reason: str | None) -> str:
    text = (reason or "").strip().lower()
    if "review_without_pelvis" in text:
        return "review_only"
    if "negative_memory" in text or "known_negative_memory" in text:
        return "memory"
    if "face" in text or "head" in text or "eye" in text:
        return "head_face"
    return "other"
