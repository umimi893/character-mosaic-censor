from __future__ import annotations

from character_mosaic.experience_store import ExperienceStore
from character_mosaic.types import CandidateEvidence, Detection
from character_mosaic.verifier_store import VerifierStore


def _record(store: ExperienceStore, source_id: int, *, score: float = 0.36):
    detection = Detection((10, 20, 40, 60), "pussy", score, "tile_2x2_1of4")
    evidence = CandidateEvidence(
        detection=detection,
        decision="keep",
        positive_signals=(f"detector:{score:.3f}",),
        matched_persons=(0,),
    )
    store.record_candidate(
        source_id,
        evidence,
        pseudo_label="positive",
        quality_tier="silver",
        negative_kind=None,
        suppression_reason=None,
        fingerprint="0123456789abcdef0123456789abcdef",
        crop_path="candidate.webp",
        app_version="test",
    )


def test_human_label_survives_candidate_row_replacement(tmp_path):
    db_path = tmp_path / "experience.sqlite3"
    experience = ExperienceStore(db_path)
    source_id = experience.upsert_source(
        source_key="file://sample.png",
        container_path=str(tmp_path / "sample.png"),
        member_path=None,
        signature="v1",
        sha256="abc",
        size_bytes=123,
        width=100,
        height=100,
        status="processed",
    )
    _record(experience, source_id)

    verifier = VerifierStore(db_path)
    rows = verifier.candidates(decision="keep", only_unlabeled=True)
    assert len(rows) == 1
    verifier.set_label(rows[0], "negative")
    assert verifier.stats()["negative"] == 1

    # Normal runtime learning replaces candidate rows for the same source.
    # Human truth is keyed by fingerprint and must not disappear with that row.
    with experience.connect() as db:
        db.execute("DELETE FROM candidates WHERE source_id=?", (source_id,))
    _record(experience, source_id, score=0.34)

    relisted = verifier.candidates(decision="keep", only_unlabeled=False)
    assert len(relisted) == 1
    assert relisted[0].manual_label == "negative"
    assert verifier.candidates(decision="keep", only_unlabeled=True) == []
