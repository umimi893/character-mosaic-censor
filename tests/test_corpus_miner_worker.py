from __future__ import annotations

from PySide6.QtCore import QCoreApplication

from character_mosaic.corpus_miner import CorpusMinerConfig
from character_mosaic.experience_store import MiningStats
from character_mosaic.workers import corpus_miner_worker as worker_module


class _FakeMiner:
    limits: list[int | None] = []

    def __init__(self, config, *, store):
        self.config = config
        self.store = store

    def mine(self, root, *, progress=None, stop_requested=None):
        del root, progress, stop_requested
        self.__class__.limits.append(self.config.max_images)
        amount = min(2, self.config.max_images) if self.config.max_images is not None else 2
        return MiningStats(discovered=amount, processed=amount)


def test_worker_applies_max_images_across_all_roots(tmp_path, monkeypatch):
    QCoreApplication.instance() or QCoreApplication([])
    _FakeMiner.limits = []
    monkeypatch.setattr(worker_module, "CorpusMiner", _FakeMiner)

    worker = worker_module.CorpusMinerWorker(
        [tmp_path / "a", tmp_path / "b", tmp_path / "c"],
        CorpusMinerConfig(max_images=3, idle_gpu_only=False),
        store_path=tmp_path / "experience.sqlite3",
    )
    emitted: list[dict] = []
    worker.finished.connect(emitted.append)
    worker.run()

    assert _FakeMiner.limits == [3, 1]
    assert emitted
    assert emitted[0]["processed"] == 3
