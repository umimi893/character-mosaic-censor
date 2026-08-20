from __future__ import annotations

from PIL import Image

from character_mosaic.anatomy_filter import AnatomyFilterResult
from character_mosaic.experience_recorder import record_process_experience
from character_mosaic.experience_store import (
    ExperienceStore,
    candidate_fingerprint,
    fingerprint_hamming,
)
from character_mosaic.negative_memory import apply_negative_memory
from character_mosaic.pseudo_labels import classify_pseudo_label
from character_mosaic.types import CandidateEvidence, Detection, ProcessResult


def _negative_evidence(box=(10, 10, 30, 30)):
    detection = Detection(box, "pussy", 0.5, "full")
    return CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("inside_upper_back:p0:1.000",),
        matched_persons=(0,),
        pelvis_distance_ratio=1.2,
    )


def _seed_gold_negatives(store, image, detection, count=8, *, distinct=True):
    # Match candidate_crop(..., context_ratio=1.8) for the uniform test image.
    crop = image.crop((11, 11, 49, 49))
    base = candidate_fingerprint(crop)
    base_value = int(base, 16)
    for index in range(count):
        # Change only low-order bits, preserving the indexed prefix while
        # creating distinct fingerprints one bit away from the query crop.
        fingerprint = (
            f"{base_value ^ (1 << index):032x}"
            if distinct
            else base
        )
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


def test_runtime_review_suppression_is_not_gold_training_data():
    detection = Detection((10, 10, 30, 30), "pussy", 0.6, "full")
    evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("review_without_pelvis",),
        matched_persons=(0,),
        pelvis_distance_ratio=1.0,
    )
    assert classify_pseudo_label(evidence, "review_without_pelvis") == (
        "negative",
        "silver",
        "review_only",
    )


def test_normal_rerun_replaces_stale_candidate_evidence(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 80), (150, 120, 110)).save(source)
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    detection = Detection((20, 20, 40, 40), "pussy", 0.80, "full")

    old_evidence = CandidateEvidence(
        detection=detection,
        decision="suppress",
        negative_signals=("inside_upper_back:p0:1.000",),
        matched_persons=(0,),
        pelvis_distance_ratio=1.1,
    )
    old_result = ProcessResult(
        source=source,
        output=None,
        detections=tuple(),
        review_required=False,
        anatomy_suppressed=(detection,),
        anatomy_suppression_reasons=("inside_upper_back",),
        candidate_evidence=(old_evidence,),
    )
    record_process_experience(source, old_result, store=store, save_crops=False)
    assert store.stats()["candidate_negative_gold"] == 1

    new_evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("near_pelvis:p0:0.20",),
        matched_persons=(0,),
        pelvis_distance_ratio=0.20,
    )
    new_result = ProcessResult(
        source=source,
        output=None,
        detections=(detection,),
        review_required=False,
        candidate_evidence=(new_evidence,),
    )
    record_process_experience(source, new_result, store=store, save_crops=False)

    with store.connect() as db:
        rows = db.execute(
            "SELECT pseudo_label,quality_tier FROM candidates"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["pseudo_label"] == "positive_seed"
    assert rows[0]["quality_tier"] == "silver"
    assert store.stats().get("candidate_negative_gold", 0) == 0


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


def test_repeated_distinct_gold_negatives_can_veto_unprotected_candidate(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.4, "full")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        matched_persons=(0,),
        pelvis_distance_ratio=1.0,
    )
    _seed_gold_negatives(store, image, detection, count=5, distinct=True)

    result = AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied")
    final = apply_negative_memory(result, image, store)
    assert final.kept == tuple()
    assert final.suppressed[0].reason == "known_negative_memory"


def test_duplicate_gold_fingerprints_do_not_manufacture_memory_confidence(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.3, "full")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        matched_persons=(0,),
        pelvis_distance_ratio=1.0,
    )
    _seed_gold_negatives(store, image, detection, count=12, distinct=False)
    final = apply_negative_memory(
        AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied"),
        image,
        store,
    )
    assert final.kept == (detection,)
    assert final.suppressed == tuple()


def test_unmatched_candidate_disables_negative_memory(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.3, "full")
    _seed_gold_negatives(store, image, detection, count=10)
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        matched_persons=tuple(),
        pelvis_distance_ratio=1.0,
    )
    final = apply_negative_memory(
        AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied"), image, store
    )
    assert final.kept == (detection,)


def test_high_confidence_candidate_disables_negative_memory(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.9, "full")
    _seed_gold_negatives(store, image, detection, count=10)
    evidence = CandidateEvidence(
        detection=detection, decision="keep", matched_persons=(0,), pelvis_distance_ratio=1.0
    )
    final = apply_negative_memory(
        AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied"), image, store
    )
    assert final.kept == (detection,)


def test_numeric_pelvis_proximity_disables_negative_memory(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.3, "full")
    _seed_gold_negatives(store, image, detection, count=10)
    evidence = CandidateEvidence(
        detection=detection, decision="keep", matched_persons=(0,), pelvis_distance_ratio=0.55
    )
    final = apply_negative_memory(
        AnatomyFilterResult(kept=(detection,), evidence=(evidence,), status="applied"), image, store
    )
    assert final.kept == (detection,)


def test_pelvis_signal_disables_negative_memory(tmp_path):
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    image = Image.new("RGB", (80, 80), (180, 160, 150))
    detection = Detection((20, 20, 40, 40), "pussy", 0.4, "full")
    _seed_gold_negatives(store, image, detection, count=10)
    protected = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=("near_pelvis:p0:0.1",),
        matched_persons=(0,),
        pelvis_distance_ratio=0.1,
    )
    result = AnatomyFilterResult(kept=(detection,), evidence=(protected,), status="applied")
    final = apply_negative_memory(result, image, store)
    assert final.kept == (detection,)
    assert final.suppressed == tuple()