from __future__ import annotations

import io
import zipfile

from PIL import Image

from character_mosaic.anatomy_filter import AnatomyFilterResult, AnatomySuppression
from character_mosaic.corpus_miner import CorpusMiner, CorpusMinerConfig
from character_mosaic.experience_store import ExperienceStore
from character_mosaic.types import CandidateEvidence, Detection


class FakeMiningDetector:
    def __init__(self):
        self.last_filter_result = AnatomyFilterResult(tuple())

    def detect(self, image, progress=None, stop_requested=None):
        detection = Detection((20, 20, 40, 40), "pussy", 0.55, "full")
        evidence = CandidateEvidence(
            detection=detection,
            decision="suppress",
            negative_signals=("inside_upper_back:p0:1.000",),
            matched_persons=(0,),
            pelvis_distance_ratio=1.2,
        )
        suppression = AnatomySuppression(detection, "inside_upper_back", 0, 0.0, 1.2)
        self.last_filter_result = AnatomyFilterResult(
            kept=tuple(), suppressed=(suppression,), evidence=(evidence,), status="applied"
        )
        return []


def _png_bytes(color=(120, 90, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (96, 96), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_miner_handles_mixed_files_zip_duplicates_and_corrupt_data(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    raw = _png_bytes()
    (root / "a.png").write_bytes(raw)
    (root / "broken.png").write_bytes(b"not an image")
    with zipfile.ZipFile(root / "archive.zip", "w") as archive:
        archive.writestr("same.png", raw)
        archive.writestr("different.png", _png_bytes((20, 30, 40)))
        archive.writestr("note.txt", "ignored")

    store = ExperienceStore(tmp_path / "experience.sqlite3")
    miner = CorpusMiner(
        CorpusMinerConfig(include_zip=True, idle_gpu_only=False, save_crops=False),
        store=store,
        detector=FakeMiningDetector(),
    )
    stats = miner.mine(root)

    assert stats.discovered == 4
    assert stats.processed == 3  # valid PNG + two valid ZIP members
    assert stats.duplicates == 1
    assert stats.skipped == 1  # corrupt PNG
    assert stats.gold_negative == 2  # duplicate contributes no candidate
    db_stats = store.stats()
    assert db_stats["candidate_negative_gold"] == 2


def test_miner_resumes_without_reprocessing_seen_material(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.png").write_bytes(_png_bytes())
    store = ExperienceStore(tmp_path / "experience.sqlite3")
    config = CorpusMinerConfig(include_zip=True, idle_gpu_only=False, save_crops=False)

    first = CorpusMiner(config, store=store, detector=FakeMiningDetector()).mine(root)
    second = CorpusMiner(config, store=store, detector=FakeMiningDetector()).mine(root)

    assert first.processed == 1
    assert first.gold_negative == 1
    assert second.processed == 0
    assert second.skipped == 1
    assert store.stats()["candidate_negative_gold"] == 1
