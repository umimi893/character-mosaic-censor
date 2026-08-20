from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .pipeline_config import PipelineConfig
from .types import ProcessResult


class JsonlRunLogger:
    """Crash-tolerant JSONL logger that flushes after each record."""

    def __init__(self, path: Path, config: PipelineConfig):
        self.path = path
        self.config = config
        self._file = None
        self._started = datetime.now()

    def open(self, total_images: int | None = None) -> "JsonlRunLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "type": "run_start",
                "timestamp": self._started.isoformat(),
                "total_images": total_images,
                "config": asdict(self.config),
            }
        )
        return self

    def log_result(self, result: ProcessResult) -> None:
        self._write(_result_to_log_item(result))

    def log_event(self, event_type: str, **fields) -> None:
        self._write({"type": event_type, "timestamp": datetime.now().isoformat(), **fields})

    def finish(self, results: Sequence[ProcessResult], stopped: bool) -> None:
        self._write(
            {
                "type": "run_end",
                "timestamp": datetime.now().isoformat(),
                "stopped": bool(stopped),
                "processed": len(results),
                "detections": sum(len(r.detections) for r in results),
                "images_with_detection": sum(1 for r in results if r.detections),
                "review": sum(1 for r in results if r.review_required),
                "count_mismatch": sum(1 for r in results if r.count_mismatch),
                "skipped": sum(1 for r in results if r.skipped),
                "cancelled": sum(1 for r in results if r.cancelled),
                "errors": sum(1 for r in results if r.error),
                "fatal_errors": sum(1 for r in results if r.fatal_error),
                "anatomy_suppressed": sum(len(r.anatomy_suppressed) for r in results),
                "elapsed_seconds": round((datetime.now() - self._started).total_seconds(), 3),
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "JsonlRunLogger":
        return self if self._file is not None else self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write(self, item: dict) -> None:
        if self._file is None:
            raise RuntimeError("logger is not open")
        self._file.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            pass


def write_jsonl_log(results: Iterable[ProcessResult], path: Path, config: PipelineConfig) -> None:
    results = list(results)
    logger = JsonlRunLogger(path, config).open(total_images=len(results))
    try:
        for result in results:
            logger.log_result(result)
        logger.finish(results, stopped=any(r.cancelled for r in results))
    finally:
        logger.close()


def _result_to_log_item(r: ProcessResult) -> dict:
    return {
        "type": "image",
        "source": str(r.source),
        "output": str(r.output) if r.output else None,
        "review_path": str(r.review_path) if r.review_path else None,
        "review_required": r.review_required,
        "count_mismatch": r.count_mismatch,
        "manual_review_path": str(r.manual_review_path) if r.manual_review_path else None,
        "skipped": r.skipped,
        "cancelled": r.cancelled,
        "elapsed_seconds": round(r.elapsed_seconds, 6),
        "error": r.error,
        "fatal_error": r.fatal_error,
        "detections": [asdict(d) for d in r.detections],
        "censor_boxes": [list(box) for box in r.censor_boxes],
        "anatomy_filter_status": r.anatomy_filter_status,
        "anatomy_suppressed": [asdict(d) for d in r.anatomy_suppressed],
        "anatomy_suppression_reasons": list(r.anatomy_suppression_reasons),
    }
