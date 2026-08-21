from __future__ import annotations

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.candidate_arbitration import apply_candidate_arbitration
from character_mosaic.pseudo_labels import classify_pseudo_label
from character_mosaic.types import BodyRegion, CandidateEvidence, Detection


def _scene(*evidence, extra_regions=()):
    detections = tuple(item.detection for item in evidence)
    regions = (
        BodyRegion((0, 0, 1344, 1728), "person", 0.95, 0, "detect_person"),
        BodyRegion((200, 0, 900, 600), "head", 0.92, 0, "detect_head"),
    ) + tuple(extra_regions)
    return AnatomyFilterResult(
        kept=detections,
        evidence=tuple(evidence),
        body_regions=regions,
        status="applied",
    )


def _evidence(detection, *, positive=None, pelvis=None, matched=(0,)):
    return CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=tuple(positive or (f"detector:{detection.score:.3f}",)),
        matched_persons=tuple(matched),
        pelvis_distance_ratio=pelvis,
    )


def test_two_tile_0496_extra_is_suppressed_by_full_supported_anchor():
    # 1,400-image regression: the real target was full+tile at 0.703 while a
    # remote butt/skin crease was independently repeated by two tiles at 0.496.
    anchor = Detection((967, 1379, 1073, 1474), "pussy", 0.7028928399085999, "tile_2x2_4of4+full")
    weak = Detection((719, 1215, 755, 1244), "pussy", 0.4957040548324585, "tile_2x2_3of4+tile_2x2_4of4")
    result = _scene(_evidence(anchor), _evidence(weak))

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor,)
    assert final.evidence[1].decision == "suppress"
    assert final.suppressed[0].reason == "weaker_aux_same_person_with_full_anchor"


def test_far_pelvis_ratio_does_not_protect_weak_aux_candidate():
    # Real armpit/torso regression: anatomy knew the candidate was not close to
    # the pelvis (0.568) but it still survived because there was no hard-negative
    # body classification.
    anchor = Detection((525, 1269, 612, 1400), "pussy", 0.7176398038864136, "tile_2x2_4of4+full+tile_2x2_3of4")
    weak = Detection((860, 670, 894, 806), "pussy", 0.3758683204650879, "tile_2x2_2of4")
    result = _scene(_evidence(anchor, pelvis=0.35, positive=("detector:0.718", "near_pelvis:p0:0.350")), _evidence(weak, pelvis=0.5681209321455934))

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor,)
    assert final.suppressed[0].detection == weak


def test_near_pelvis_positive_always_protects_weak_candidate():
    anchor = Detection((500, 1300, 600, 1400), "pussy", 0.72, "full+tile_2x2_4of4")
    weak = Detection((850, 700, 900, 780), "pussy", 0.34, "tile_2x2_2of4")
    weak_ev = _evidence(
        weak,
        positive=("detector:0.340", "near_pelvis:p0:0.320"),
        pelvis=0.32,
    )
    result = _scene(_evidence(anchor), weak_ev)

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor, weak)
    assert final.suppressed == tuple()


def test_geometry_groin_positive_always_protects_weak_candidate():
    anchor = Detection((500, 1300, 600, 1400), "pussy", 0.72, "full+tile_2x2_4of4")
    weak = Detection((850, 700, 900, 780), "pussy", 0.34, "tile_2x2_2of4")
    weak_ev = _evidence(
        weak,
        positive=("detector:0.340", "inside_groin_zone:p0"),
        pelvis=None,
    )
    result = _scene(_evidence(anchor), weak_ev)

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor, weak)
    assert final.suppressed == tuple()


def test_multi_person_scene_fails_open():
    anchor = Detection((500, 1300, 600, 1400), "pussy", 0.72, "full+tile_2x2_4of4")
    weak = Detection((850, 700, 900, 780), "pussy", 0.34, "tile_2x2_2of4")
    second_person = BodyRegion((700, 0, 1344, 1728), "person", 0.90, 1, "detect_person")
    result = _scene(_evidence(anchor), _evidence(weak), extra_regions=(second_person,))

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor, weak)
    assert final.suppressed == tuple()


def test_full_supported_secondary_candidate_is_not_arbitrated():
    anchor = Detection((500, 1300, 600, 1400), "pussy", 0.72, "full+tile_2x2_4of4")
    secondary = Detection((850, 700, 900, 780), "pussy", 0.34, "tile_2x2_2of4+full")
    result = _scene(_evidence(anchor), _evidence(secondary))

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor, secondary)
    assert final.suppressed == tuple()


def test_no_strong_anchor_means_fail_open():
    anchor = Detection((500, 1300, 600, 1400), "pussy", 0.59, "full+tile_2x2_4of4")
    weak = Detection((850, 700, 900, 780), "pussy", 0.34, "tile_2x2_2of4")
    result = _scene(_evidence(anchor), _evidence(weak))

    final = apply_candidate_arbitration(result)

    assert final.kept == (anchor, weak)
    assert final.suppressed == tuple()


def test_arbitration_runtime_negative_remains_silver():
    anchor = Detection((500, 1300, 600, 1400), "pussy", 0.72, "full+tile_2x2_4of4")
    weak = Detection((850, 700, 900, 780), "pussy", 0.34, "tile_2x2_2of4")
    final = apply_candidate_arbitration(_scene(_evidence(anchor), _evidence(weak)))
    weak_final = next(item for item in final.evidence if item.detection == weak)

    assert classify_pseudo_label(
        weak_final,
        "weaker_aux_same_person_with_full_anchor",
    ) == ("negative", "silver", "other")
