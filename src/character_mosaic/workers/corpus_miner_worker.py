from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..corpus_miner import CorpusMiner, CorpusMinerConfig
from ..experience_store import ExperienceStore


class CorpusMinerWorker(QObject):
    progress = Signal(object, str)
    root_started = Signal(str)
    root_finished = Signal(str, object)
    finished = Signal(object)
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        roots: list[Path],
        config: CorpusMinerConfig,
        store_path: Path | None = None,
    ):
        super().__init__()
        self.roots = [Path(root) for root in roots]
        self.config = config
        self.store_path = store_path
        self._stop = threading.Event()

    @Slot()
    def run(self) -> None:
        aggregate = {
            "discovered": 0,
            "processed": 0,
            "duplicates": 0,
            "skipped": 0,
            "errors": 0,
            "candidates": 0,
            "gold_negative": 0,
            "silver": 0,
            "quarantine": 0,
        }
        try:
            store = ExperienceStore(self.store_path) if self.store_path else ExperienceStore()
            miner = CorpusMiner(self.config, store=store)
            total_limit = self.config.max_images
            for root in self.roots:
                if self._stop.is_set():
                    break

                if total_limit is not None:
                    remaining = int(total_limit) - int(aggregate["processed"])
                    if remaining <= 0:
                        break
                    miner.config = replace(self.config, max_images=remaining)
                else:
                    miner.config = self.config

                self.root_started.emit(str(root))
                self.status.emit(f"採掘中: {root}")
                stats = miner.mine(
                    root,
                    progress=self.progress.emit,
                    stop_requested=self._stop.is_set,
                )
                self.root_finished.emit(str(root), stats)
                for key in aggregate:
                    aggregate[key] += int(getattr(stats, key))
            self.finished.emit(aggregate)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def request_stop(self) -> None:
        self._stop.set()
        self.status.emit("採掘停止を要求しました")