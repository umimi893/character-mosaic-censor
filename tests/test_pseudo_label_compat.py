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
