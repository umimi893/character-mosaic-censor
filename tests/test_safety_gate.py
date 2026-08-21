from __future__ import annotations

from character_mosaic.anatomy_filter import AnatomyFilterResult, AnatomySuppression
from character_mosaic.safety_gate import apply_safety_gate
from character_mosaic.types import CandidateEvidence, Detection


def _suppressed(
    *,
    score=0.716,
    source="tile_2x2_4of4+tile_2x2_3of4+full",
    pelvis=0.115,
    reason="inside_torso",
    positive=("detector:0.716", "near_pelvis:p0:0.115"),
):
    detection = Detection((597, 892, 677, 1098), "pussy", score, source)
    evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        positive_signals=tuple(positive),
        negative_signals=(f"{reason}:p0:0.778",),
        matched_persons=(0,),
        pelvis_distance_ratio=pelvis,
    )
    suppression = AnatomySuppression(
        detection=detection,
        reason=reason,
        person_index=0,
        joint_distance_ratio=0.222,
        pelvis_distance_ratio=pelvis if pelvis is not None else 999.0,
    )
    return detection, AnatomyFilterResult(
        kept=tuple(),
        suppressed=(suppression,),
        evidence=(evidence,),
        status="applied",
    )


def test_strong_full_target_extremely_near_same_person_pelvis_is_rescued():
    detection, result = _suppressed()

    final = apply_safety_gate(result)

    assert final.kept == (detection,)
    assert final.suppressed == tuple()
    assert final.evidence[0].decision == "keep"
    assert any(
        signal.startswith("strong_full_same_person_pelvis_safety:p0:")
        for signal in final.evidence[0].positive_signals
    )


def test_tile_only_candidate_is_not_rescued():
    detection, result = _suppressed(source="tile_2x2_4of4")
    final = apply_safety_gate(result)
    assert final.kept == tuple()
    assert final.suppressed[0].detection == detection


def test_low_confidence_candidate_is_not_rescued():
    detection, result = _suppressed(score=0.36)
    final = apply_safety_gate(result)
    assert final.kept == tuple()
    assert final.suppressed[0].detection == detection


def test_not_extremely_near_pelvis_is_not_rescued():
    detection, result = _suppressed(
        pelvis=0.35,
        positive=("detector:0.716", "near_pelvis:p0:0.350"),
    )
    final = apply_safety_gate(result)
    assert final.kept == tuple()
    assert final.suppressed[0].detection == detection


def test_semantic_face_suppression_is_never_rescued():
    detection, result = _suppressed(reason="inside_eye_face_head")
    final = apply_safety_gate(result)
    assert final.kept == tuple()
    assert final.suppressed[0].detection == detection
