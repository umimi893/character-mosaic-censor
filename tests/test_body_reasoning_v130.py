from __future__ import annotations

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.body_reasoning import _derive_torso_regions, enhance_anatomy_result
from character_mosaic.types import CandidateEvidence, Detection, PosePoint


def _pose_points(person: int = 0, x_offset: float = 0.0, *, horizontal: bool = False):
    if horizontal:
        coords = {
            "right_shoulder": (60, 100),
            "left_shoulder": (60, 140),
            "right_hip": (180, 105),
            "left_hip": (180, 145),
        }
    else:
        coords = {
            "right_shoulder": (70, 80),
            "left_shoulder": (130, 80),
            "right_hip": (80, 220),
            "left_hip": (120, 220),
        }
    return tuple(
        PosePoint(x + x_offset, y, 0.95, label, person)
        for label, (x, y) in coords.items()
    )


def test_torso_back_candidate_is_suppressed():
    detection = Detection((90, 135, 110, 165), "pussy", 0.72)
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.720",),
        matched_persons=(0,),
        pelvis_distance_ratio=0.70,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        pose_points=_pose_points(),
        status="applied",
    )

    final = enhance_anatomy_result(result, (300, 400))

    assert final.kept == tuple()
    assert final.evidence[0].decision == "suppress"
    assert final.suppressed[0].reason == "inside_torso_back"
    assert any(region.kind == "torso_back_zone" for region in final.body_regions)


def test_same_person_pelvis_does_not_save_clear_torso_core():
    detection = Detection((90, 175, 110, 205), "pussy", 0.72)
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.720", "near_pelvis:p0:0.350"),
        matched_persons=(0,),
        pelvis_distance_ratio=0.35,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        pose_points=_pose_points(),
        status="applied",
    )

    final = enhance_anatomy_result(result, (300, 400))

    assert final.evidence[0].decision == "suppress"
    assert final.suppressed[0].reason == "inside_torso_back"


def test_other_person_pelvis_protects_close_contact_torso_overlap():
    detection = Detection((90, 135, 110, 165), "pussy", 0.85)
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.850", "near_pelvis:p1:0.200"),
        matched_persons=(0, 1),
        pelvis_distance_ratio=0.20,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        pose_points=_pose_points(0) + _pose_points(1, x_offset=150),
        status="applied",
    )

    final = enhance_anatomy_result(result, (500, 400))

    assert final.kept == (detection,)
    assert final.suppressed == tuple()
    assert final.evidence[0].decision == "keep"


def test_anatomy_review_without_pelvis_is_removed():
    detection = Detection((85, 45, 110, 75), "pussy", 0.60)
    evidence = CandidateEvidence(
        detection=detection,
        decision="review",
        positive_signals=("detector:0.600",),
        negative_signals=("inside_face:p0:0.900", "inside_head:p0:0.950"),
        matched_persons=(0,),
        pelvis_distance_ratio=1.2,
    )
    result = AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied")

    final = enhance_anatomy_result(result, (300, 400))

    assert final.kept == tuple()
    assert final.requires_review is False
    assert final.evidence[0].decision == "suppress"
    assert final.suppressed[0].reason == "review_without_pelvis"


def test_horizontal_pose_does_not_create_axis_aligned_torso_zone():
    regions = _derive_torso_regions(_pose_points(horizontal=True), (300, 300))
    assert regions == []
