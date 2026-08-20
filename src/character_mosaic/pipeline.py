"""Public batch-processing API.

Implementation is split into focused modules so GUI, CLI, review persistence,
and file I/O can evolve independently while keeping the historical import
surface stable.
"""

from __future__ import annotations

from dataclasses import replace

from .i18n import t
from .pipeline_config import PipelineConfig
from .pipeline_logging import JsonlRunLogger, write_jsonl_log
from .pipeline_processor import BatchProcessor as _BatchProcessor
from .pipeline_processor import discover_images, validate_processing_paths
from .pipeline_review import write_review_html


class BatchProcessor(_BatchProcessor):
    """Batch processor with an optional GUI-friendly count-review policy.

    The historical behavior treats any detection count different from the
    configured person count as a mismatch. When ``review_only_over_count`` is
    enabled, the person count is instead a maximum plausible detection count.
    This keeps zero detections and partially visible targets out of the manual
    count-mismatch bucket while still quarantining obvious over-detections.
    """

    def process_file(
        self,
        source,
        output,
        review_copy=None,
        manual_review_copy=None,
        manual_review_annotated=None,
        preview=None,
        stop_requested=None,
    ):
        if not self.config.review_only_over_count:
            return super().process_file(
                source,
                output,
                review_copy,
                manual_review_copy,
                manual_review_annotated,
                preview,
                stop_requested,
            )

        expected = self.config.expected_person_count

        def preview_proxy(frame):
            if preview is None:
                return
            count = len(frame.detections)
            if frame.stage == "detected" and count <= expected:
                if count == 0:
                    status = t(
                        self.config.language,
                        "対象未検出（正常扱い）",
                        "No target detected (treated as normal)",
                    )
                else:
                    status = t(
                        self.config.language,
                        f"検出完了: {count}件",
                        f"Detection complete: {count}",
                    )
                frame = replace(frame, status=status)
            elif frame.stage == "censored" and count == 0:
                frame = replace(
                    frame,
                    status=t(
                        self.config.language,
                        "対象未検出: 元画像をそのまま保存",
                        "No target detected: original image copied unchanged",
                    ),
                )
            preview(frame)

        result = super().process_file(
            source,
            output,
            review_copy,
            manual_review_copy,
            manual_review_annotated,
            preview_proxy if preview is not None else None,
            stop_requested,
        )

        if result.error or result.cancelled or result.skipped or result.fatal_error:
            return result

        over_detected = len(result.detections) > expected
        if result.count_mismatch and not over_detected:
            for path in (manual_review_copy, manual_review_annotated):
                if path is not None:
                    path.unlink(missing_ok=True)
            return replace(result, count_mismatch=False, manual_review_path=None)
        return result


__all__ = [
    "BatchProcessor",
    "PipelineConfig",
    "JsonlRunLogger",
    "discover_images",
    "validate_processing_paths",
    "write_jsonl_log",
    "write_review_html",
]
