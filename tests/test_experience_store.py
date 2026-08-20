from __future__ import annotations

from PIL import Image

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.experience_store import (
    ExperienceStore,
    candidate_fingerprint,
    classify_pseudo_label,
    fingerprint_hamming,
)
from character_mosaic.negative_memory import apply_negative_memory
from character_mosaic.types import CandidateEvidence, Detection


def _negative_evidence(box=(10, 10, 30, 30)):
    detection = Detection(box, "pussy", 0.5, "full")
    return CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("inside_upper_back:p0:1.000",),
        matched_persons=(0,),
        pelvis_distance_ratio=1.2,
    )


def test_fingerprint_is_stable_and_distance_zero():
    image = Image.new("RGB", (64, 64), (120, 90, 80))
    first = candidate_fingerprint(image)
    second = candidate_fingerprint(image.copy())
    assert first == second
    assert len(first) == 32
    assert fingerprint_hamming(first, second) == 0


def test_store_deduplicates_source_hash_and_counts_candidates(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    first = store.upsert_source(
        source_key="file://a.png", container_path="a.png", member_path=None,
        signature="1", sha256="abc", size_bytes=100, width=64, height=64,
        status="processed",
    )
    assert store.duplicate_source_id("abc", excluding_key="file://b.png") == first

    evidence = _negative_evidence()
    pseudo, tier, kind = classify_pseudo_label(evidence, "inside_upper_back")
    assert (pseudo, tier, kind) == ("negative", "gold", "back")
    store.record_candidate(
        first, evidence, pseudo_label=pseudo, quality_tier=tier,
        negative_kind=kind, suppression_reason="inside_upper_back",
        fingerprint="0123456789abcdef0123456789abcdef",
        crop_path=None, app_version="test",
    )
    stats = store.stats()
    assert stats["sources"] == 1
    assert stats["candidates"] == 1
    assert stats["candidate_negative_gold"] == 1


def test_ambiguous_keep_goes_to_quarantine():
    detection = Detection((1, 2, 3, 4), "pussy", 0.8)
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("near_pelvis:p0:0.2",),
        negative_signals=("inside_face:p1:0.8",),
        matched_persons=(0,),
    )
    assert classify_pseudo_label(evidence, None) == ("quarantine", "quarantine", None)


def test_repeated_gold_negative_can_veto_unprotected_candidate(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.4, "full")
    evidence = CandidateEvidence(detection=detection, decision="keep", matched_persons=(0,))
    crop = image.crop((11, 11, 49, 49))
    fingerprint = candidate_fingerprint(crop)

    for index in range(3):
        source_id = store.upsert_source(
            source_key=f"file://neg{index}.png", container_path=f"neg{index}.png", member_path=None,
            signature="1", sha256=f"hash{index}", size_bytes=100, width=80, height=80,
            status="processed",
        )
        neg = CandidateEvidence(detection=detection, decision="suppress", matched_persons=(0,))
        store.record_candidate(
            source_id, neg, pseudo_label="negative", quality_tier="gold", negative_kind="back",
            suppression_reason="inside_upper_back", fingerprint=fingerprint,
            crop_path=None, app_version="test",
        )

    result = AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied")
    final = apply_negative_memory(result, image, store)
    assert final.kept == tuple()
    assert final.suppressed[0].reason == "known_negative_memory"


def test_pelvis_signal_disables_negative_memory(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.4, "full")
    protected = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("near_pelvis:p0:0.1",),
        matched_persons=(0,),
    )
    result = AnatomyFilterResult(kept=(detection,), evidence=(protected,), status="applied")
    final = apply_negative_memory(result, image, store)
    assert final.kept == (detection,)
    assert final.suppressed == tuple()
