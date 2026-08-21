from __future__ import annotations

from character_mosaic.experience_store import ExperienceStore
from character_mosaic.types import CandidateEvidence, Detection
from character_mosaic.verifier_store import VerifierStore, is_derived_source_path


def _record(
    store: ExperienceStore,
    source_id: int,
    *,
    score: float = 0.36,
    fingerprint: str = "0123456789abcdef0123456789abcdef",
):
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
        fingerprint=fingerprint,
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
    assert verifier.coverage_stats() == {"candidates": 1, "labeled": 0, "unlabeled": 1}

    verifier.set_label(rows[0], "negative")
    assert verifier.stats()["negative"] == 1
    assert verifier.coverage_stats() == {"candidates": 1, "labeled": 1, "unlabeled": 0}
    samples = verifier.labeled_samples(labels=("negative",))
    assert len(samples) == 1
    assert samples[0].fingerprint == rows[0].fingerprint
    assert samples[0].label == "negative"
    assert samples[0].box == rows[0].box

    # Normal runtime learning replaces candidate rows for the same source.
    # Human truth is keyed by fingerprint and must not disappear with that row.
    with experience.connect() as db:
        db.execute("DELETE FROM candidates WHERE source_id=?", (source_id,))
    _record(experience, source_id, score=0.34)

    relisted = verifier.candidates(decision="keep", only_unlabeled=False)
    assert len(relisted) == 1
    assert relisted[0].manual_label == "negative"
    assert verifier.candidates(decision="keep", only_unlabeled=True) == []
    assert verifier.labeled_samples(labels=("negative",))[0].detector_score == 0.36


def test_derived_censored_sources_are_hidden_and_excluded_from_training(tmp_path):
    db_path = tmp_path / "experience.sqlite3"
    experience = ExperienceStore(db_path)

    clean_id = experience.upsert_source(
        source_key="file://clean.png",
        container_path=str(tmp_path / "clean" / "sample.png"),
        member_path=None,
        signature="v1",
        sha256="clean",
        size_bytes=123,
        width=100,
        height=100,
        status="processed",
    )
    derived_id = experience.upsert_source(
        source_key="file://derived.png",
        container_path=str(tmp_path / "_censored" / "_manual_review" / "edit" / "sample.png"),
        member_path=None,
        signature="v1",
        sha256="derived",
        size_bytes=123,
        width=100,
        height=100,
        status="processed",
    )
    _record(
        experience,
        clean_id,
        fingerprint="11111111111111111111111111111111",
    )
    _record(
        experience,
        derived_id,
        fingerprint="22222222222222222222222222222222",
    )

    verifier = VerifierStore(db_path)
    clean_rows = verifier.candidates(decision="keep", only_unlabeled=True)
    assert [row.fingerprint for row in clean_rows] == ["11111111111111111111111111111111"]

    all_rows = verifier.candidates(
        decision="keep",
        only_unlabeled=True,
        exclude_derived=False,
    )
    assert {row.fingerprint for row in all_rows} == {
        "11111111111111111111111111111111",
        "22222222222222222222222222222222",
    }

    derived_row = next(
        row for row in all_rows
        if row.fingerprint == "22222222222222222222222222222222"
    )
    verifier.set_label(derived_row, "negative")
    assert verifier.labeled_samples(labels=("negative",)) == []
    assert len(verifier.labeled_samples(labels=("negative",), exclude_derived=False)) == 1
    assert verifier.stats()["negative"] == 0
    assert verifier.stats(exclude_derived=False)["negative"] == 1
    assert verifier.coverage_stats() == {"candidates": 1, "labeled": 0, "unlabeled": 1}


def test_derived_source_path_detection_handles_windows_and_review_dirs():
    assert is_derived_source_path(r"F:\\work\\_censored\\sample.png")
    assert is_derived_source_path(r"F:\\work\\_manual_review\\edit\\sample.png")
    assert is_derived_source_path(r"F:\\work\\review\\sample.png")
    assert is_derived_source_path(r"F:\\work\\auto_censored\\sample.png")
    assert not is_derived_source_path(r"F:\\work\\clean\\sample.png")
