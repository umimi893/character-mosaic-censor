from __future__ import annotations

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.body_reasoning import _derive_torso_regions, enhance_anatomy_result
from character_mosaic.types import BodyRegion, CandidateEvidence, Detection, PosePoint


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


def _regions(person_box, head_box, person: int = 0):
    return (
        BodyRegion(tuple(person_box), "person", 0.90, person, "detect_person"),
        BodyRegion(tuple(head_box), "head", 0.90, person, "detect_head"),
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


def test_retry_rot90_low_confidence_waist_candidate_is_suppressed_without_lower_pose():
    # Regression from a real false positive: a waist crease was the only
    # detection after retry_rot90 while DWPose returned shoulders/arms but no
    # hips, knees, or ankles. Detector confidence alone must not auto-censor it.
    detection = Detection((604, 892, 639, 949), "pussy", 0.30588167905807495, "retry_rot90")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.306",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )
    upper_pose = (
        PosePoint(775.8, 545.9, 0.61, "right_shoulder", 0),
        PosePoint(402.6, 310.2, 0.61, "right_elbow", 0),
        PosePoint(1081.7, 680.6, 0.57, "left_shoulder", 0),
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        pose_points=upper_pose,
        body_regions=_regions((102, 1, 1343, 1725), (675, 13, 1230, 563)),
        status="applied",
    )

    final = enhance_anatomy_result(result, (1386, 1728))

    assert final.kept == tuple()
    assert final.evidence[0].decision == "suppress"
    assert final.suppressed[0].reason == "upper_body_retry_without_lower_pose"
    assert any(
        signal.startswith("upper_body_retry_without_lower_pose:p0:")
        for signal in final.evidence[0].negative_signals
    )


def test_retry_flip_low_confidence_waist_candidate_is_suppressed_with_face_only_pose():
    # Second real regression: only face landmarks survived DWPose and a
    # retry_rot180+retry_vflip candidate landed on the side of the waist.
    detection = Detection(
        (675, 749, 748, 812),
        "pussy",
        0.3647313714027405,
        "retry_rot180+retry_vflip",
    )
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.365",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )
    face_only_pose = (
        PosePoint(957.8, 400.2, 0.73, "nose", 0),
        PosePoint(904.5, 335.7, 0.76, "right_eye", 0),
        PosePoint(935.4, 332.9, 0.73, "left_eye", 0),
        PosePoint(719.2, 397.4, 0.70, "right_ear", 0),
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        pose_points=face_only_pose,
        body_regions=_regions((228, 1, 1143, 1726), (449, 8, 1015, 571)),
        status="applied",
    )

    final = enhance_anatomy_result(result, (1280, 1728))

    assert final.kept == tuple()
    assert final.suppressed[0].reason == "upper_body_retry_without_lower_pose"


def test_upper_body_retry_fallback_keeps_high_confidence_candidate():
    detection = Detection((604, 892, 639, 949), "pussy", 0.72, "retry_rot90")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.720",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        body_regions=_regions((102, 1, 1343, 1725), (675, 13, 1230, 563)),
        status="regions_only",
    )

    final = enhance_anatomy_result(result, (1386, 1728))

    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_upper_body_retry_fallback_does_not_affect_normal_full_pass():
    detection = Detection((604, 892, 639, 949), "pussy", 0.31, "full")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.310",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        body_regions=_regions((102, 1, 1343, 1725), (675, 13, 1230, 563)),
        status="regions_only",
    )

    final = enhance_anatomy_result(result, (1386, 1728))

    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_upper_body_retry_fallback_fails_open_when_lower_body_pose_exists():
    detection = Detection((30, 120, 50, 145), "pussy", 0.31, "retry_rot90")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.310",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        pose_points=(PosePoint(40, 220, 0.95, "right_hip", 0),),
        body_regions=_regions((0, 0, 200, 400), (50, 10, 150, 100)),
        status="applied",
    )

    final = enhance_anatomy_result(result, (200, 400))

    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_upper_body_retry_fallback_preserves_other_person_pelvis_protection():
    detection = Detection((604, 892, 639, 949), "pussy", 0.31, "retry_rot90")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("detector:0.310", "near_pelvis:p1:0.180"),
        matched_persons=(0,),
        pelvis_distance_ratio=0.18,
    )
    result = AnatomyFilterResult(
        kept=(detection,),
        evidence=(evidence,),
        body_regions=_regions((102, 1, 1343, 1725), (675, 13, 1230, 563)),
        status="regions_only",
    )

    final = enhance_anatomy_result(result, (1386, 1728))

    assert final.kept == (detection,)
    assert final.suppressed == tuple()
