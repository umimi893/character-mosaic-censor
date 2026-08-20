from __future__ import annotations

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.body_geometry import apply_body_geometry_v2
from character_mosaic.types import CandidateEvidence, Detection, PosePoint


def _pose(person=0, x=0.0):
    coords = {
        "right_shoulder": (70, 80), "left_shoulder": (130, 80),
        "right_elbow": (55, 145), "left_elbow": (145, 145),
        "right_hip": (80, 220), "left_hip": (120, 220),
        "right_knee": (85, 320), "left_knee": (115, 320),
        "right_ankle": (85, 400), "left_ankle": (115, 400),
    }
    return tuple(PosePoint(px + x, py, 0.95, label, person) for label, (px, py) in coords.items())


def _result(box, *, matched=(0,), positive=(), points=None, score=0.7):
    detection = Detection(box, "pussy", score)
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=tuple(positive),
        matched_persons=tuple(matched),
        pelvis_distance_ratio=0.9,
    )
    return detection, AnatomyFilterResult(
        kept=(detection,), evidence=(evidence,), pose_points=points or _pose(), status="applied"
    )


def test_upper_back_candidate_is_suppressed():
    detection, result = _result((90, 120, 110, 150))
    final = apply_body_geometry_v2(result, (300, 450))
    assert final.kept == tuple()
    assert final.suppressed[0].reason == "inside_upper_back"


def test_armpit_candidate_is_suppressed_outside_torso_polygon():
    detection, result = _result((55, 98, 68, 112))
    final = apply_body_geometry_v2(result, (300, 450))
    assert final.kept == tuple()
    assert "armpit_v2" in final.suppressed[0].reason


def test_thigh_candidate_is_suppressed_along_pose_bone():
    detection, result = _result((78, 260, 96, 285))
    final = apply_body_geometry_v2(result, (300, 450))
    assert final.kept == tuple()
    assert final.suppressed[0].reason == "on_right_thigh"


def test_lower_leg_candidate_is_suppressed_along_pose_bone():
    detection, result = _result((78, 350, 96, 375))
    final = apply_body_geometry_v2(result, (300, 450))
    assert final.kept == tuple()
    assert final.suppressed[0].reason == "on_right_lower_leg"


def test_directional_groin_zone_protects_candidate():
    detection, result = _result((90, 226, 110, 246), positive=("near_pelvis:p0:0.25",))
    final = apply_body_geometry_v2(result, (300, 450))
    assert final.kept == (detection,)
    assert final.suppressed == tuple()
    assert any(signal.startswith("inside_groin_zone:p0") for signal in final.evidence[0].positive_signals)


def test_unmatched_candidate_fails_open():
    detection, result = _result((90, 120, 110, 150), matched=())
    final = apply_body_geometry_v2(result, (300, 450))
    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_other_person_pelvis_signal_wins_over_back_geometry():
    points = _pose(0) + _pose(1, x=220)
    detection, result = _result(
        (90, 120, 110, 150),
        matched=(0,),
        positive=("near_pelvis:p1:0.18",),
        points=points,
    )
    final = apply_body_geometry_v2(result, (600, 450))
    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_multi_person_disagreement_fails_open():
    points = _pose(0) + _pose(1, x=220)
    detection, result = _result((90, 120, 110, 150), matched=(0, 1), points=points)
    final = apply_body_geometry_v2(result, (600, 450))
    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_visual_map_contains_v2_regions():
    _, result = _result((200, 200, 220, 220))
    final = apply_body_geometry_v2(result, (300, 450))
    kinds = {region.kind for region in final.body_regions}
    assert "torso_geometry_v2" in kinds
    assert "right_armpit_geometry_v2" in kinds
    assert "right_thigh_geometry_v2" in kinds
    assert "right_lower_leg_geometry_v2" in kinds
