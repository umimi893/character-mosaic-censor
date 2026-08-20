from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from ..detector import get_runtime_info
from ..pipeline import BatchProcessor, JsonlRunLogger, PipelineConfig


class BatchWorker(QObject):
    preview = Signal(object)
    discovered = Signal(int)
    progress = Signal(int, int, str, object)
    runtime_ready = Signal(object)
    log_ready = Signal(str)
    finished = Signal(object, str, bool)
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        review_dir: Path | None,
        config: PipelineConfig,
        log_dir: Path,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.review_dir = review_dir
        self.config = config
        self.log_dir = log_dir
        self._stop = threading.Event()

    @Slot()
    def run(self) -> None:
        logger: JsonlRunLogger | None = None
        results = []
        log_path = self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
        try:
            self.status.emit("ランタイム確認中")
            self.runtime_ready.emit(get_runtime_info())
            self.status.emit("画像を走査中")
            processor = BatchProcessor(self.config)
            images = processor.discover_images(self.input_dir, self.output_dir, self.review_dir)
            self.discovered.emit(len(images))

            logger = JsonlRunLogger(log_path, self.config).open(total_images=len(images))
            self.log_ready.emit(str(log_path))
            if not images:
                logger.log_event("notice", message="処理対象画像がありません")

            def on_progress(index, total, src, result):
                self.progress.emit(index, total, str(src), result)

            results = processor.process_folder(
                self.input_dir,
                self.output_dir,
                self.review_dir,
                progress=on_progress,
                preview=self.preview.emit,
                stop_requested=self._stop.is_set,
                images=images,
                result_callback=logger.log_result,
            )
            stopped = self._stop.is_set() or any(r.cancelled for r in results)
            logger.finish(results, stopped=stopped)
            fatal = next((r for r in results if r.fatal_error), None)
            if fatal is not None:
                self.failed.emit(f"推論エンジンを継続できません: {fatal.error}\nログ: {log_path}")
            else:
                self.finished.emit(results, str(log_path), stopped)
        except Exception as exc:
            if logger is not None:
                try:
                    logger.log_event("run_error", error=f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if logger is not None:
                logger.close()

    def request_stop(self) -> None:
        """Thread-safe cooperative cancellation request."""
        self._stop.set()
        self.status.emit("停止要求を受け付けました")
