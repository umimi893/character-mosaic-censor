from __future__ import annotations

from PIL import Image

from character_mosaic.semantic_probe import (
    candidate_context_crop,
    probe_evidence,
    summarize_wd14,
)
from character_mosaic.types import CandidateEvidence, Detection


def test_candidate_context_crop_expands_detector_box():
    image = Image.new("RGB", (1000, 800), "white")
    detection = Detection((450, 350, 500, 400), "pussy", 0.4, "tile_2x2_1of4")

    crop, crop_box = candidate_context_crop(image, detection, scale=3.5, min_side=256)

    assert crop.size == (256, 256)
    assert crop_box == (347, 247, 603, 503)


def test_candidate_context_crop_clamps_at_image_edges():
    image = Image.new("RGB", (300, 300), "white")
    detection = Detection((5, 10, 25, 30), "pussy", 0.3, "retry_rot90")

    crop, crop_box = candidate_context_crop(image, detection, min_side=256)

    assert crop_box[0] == 0
    assert crop_box[1] == 0
    assert crop_box[2] <= 300
    assert crop_box[3] <= 300
    assert crop.width > 0 and crop.height > 0


def test_summarize_wd14_keeps_relevant_scores_and_top_tags():
    rating, relevant, top = summarize_wd14(
        {"explicit": 0.8, "sensitive": 0.1},
        {"pussy": 0.72, "armpits": 0.04, "stomach": 0.55, "1girl": 0.99},
        top_k=2,
    )

    assert rating["explicit"] == 0.8
    assert relevant["pussy"] == 0.72
    assert relevant["armpits"] == 0.04
    assert relevant["stomach"] == 0.55
    assert top == (("1girl", 0.99), ("pussy", 0.72))


def test_probe_evidence_is_shadow_only_and_preserves_current_decision():
    image = Image.new("RGB", (640, 640), "white")
    detection = Detection((200, 260, 250, 320), "pussy", 0.36, "tile_2x2_3of4")
    evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        positive_signals=("detector:0.360", "near_pelvis:p0:0.115"),
        negative_signals=("inside_torso:p0:0.778",),
        matched_persons=(0,),
        pelvis_distance_ratio=0.115,
    )

    calls = []

    def fake_tagger(crop, **kwargs):
        calls.append((crop.size, kwargs))
        return (
            {"explicit": 0.77},
            {"pussy": 0.68, "armpits": 0.03, "stomach": 0.18},
        )

    probe = probe_evidence("sample.png", image, evidence, tagger=fake_tagger)

    assert probe.current_decision == "suppress"
    assert probe.genital_score == 0.68
    assert probe.armpit_score == 0.03
    assert probe.pelvis_distance_ratio == 0.115
    assert probe.negative_signals == ("inside_torso:p0:0.778",)
    assert calls
    assert calls[0][1]["general_threshold"] == 0.05
    assert calls[0][1]["character_threshold"] == 1.0
    assert calls[0][1]["fmt"] == ("rating", "general")
