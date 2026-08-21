from __future__ import annotations

from character_mosaic.experience_store import classify_pseudo_label as classify_from_store
from character_mosaic.pseudo_labels import classify_pseudo_label as classify_canonical
from character_mosaic.types import CandidateEvidence, Detection


def test_experience_store_classifier_delegates_to_canonical_policy():
    detection = Detection((10, 10, 30, 30), "pussy", 0.6, "full")
    evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("review_without_pelvis",),
        matched_persons=(0,),
        pelvis_distance_ratio=1.0,
    )

    expected = classify_canonical(evidence, "review_without_pelvis")
    assert expected == ("negative", "silver", "review_only")
    assert classify_from_store(evidence, "review_without_pelvis") == expected


def test_upper_body_retry_fallback_never_seeds_gold_negative_memory():
    detection = Detection((604, 892, 639, 949), "pussy", 0.31, "retry_rot90")
    evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("upper_body_retry_without_lower_pose:p0:0.121",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )

    expected = classify_canonical(evidence, "upper_body_retry_without_lower_pose")
    assert expected == ("negative", "silver", "other")
    assert classify_from_store(evidence, "upper_body_retry_without_lower_pose") == expected


def test_upper_body_tile_fallback_never_seeds_gold_negative_memory():
    detection = Detection((394, 704, 442, 748), "pussy", 0.354, "tile_2x2_1of4")
    evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("upper_body_tile_without_lower_pose:p0:0.715",),
        matched_persons=(0,),
        pelvis_distance_ratio=None,
    )

    expected = classify_canonical(evidence, "upper_body_tile_without_lower_pose")
    assert expected == ("negative", "silver", "other")
    assert classify_from_store(evidence, "upper_body_tile_without_lower_pose") == expected
