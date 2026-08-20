"""Public batch-processing API.

Implementation is split into focused modules so GUI, CLI, review persistence,
and file I/O can evolve independently while keeping the historical import
surface stable.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from .anatomy_filter import AnatomyFilterConfig
from .body_reasoning import BodyReasoningDetector
from .i18n import t
from .pipeline_config import PipelineConfig
from .pipeline_logging import JsonlRunLogger, write_jsonl_log
from .pipeline_processor import BatchProcessor as _BatchProcessor
from .pipeline_processor import discover_images, validate_processing_paths
from .pipeline_review import write_review_html


class BatchProcessor(_BatchProcessor):
    """Batch processor with public UX/safety policy layers.

    The default detector is wrapped by the final body-region reasoning layer.
    Pelvis evidence protects close-contact scenes, while strong face/head,
    knee/armpit, and torso/back evidence can remove obvious false positives.
    """

    def __init__(self, config: PipelineConfig | None = None, detector=None):
        super().__init__(config=config, detector=detector)
        if detector is None and not isinstance(self.detector, BodyReasoningDetector):
            self.detector = BodyReasoningDetector(
                self.detector,
                AnatomyFilterConfig(enabled=self.config.anatomy_filter),
            )

    def _reset_anatomy_diagnostics(self) -> None:
        reset = getattr(self.detector, "reset_filter_state", None)
        if callable(reset):
            reset()

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
        self._reset_anatomy_diagnostics()
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
                status = (
                    t(
                        self.config.language,
                        "対象未検出（正常扱い）",
                        "No target detected (treated as normal)",
                    )
                    if count == 0
                    else t(
                        self.config.language,
                        f"検出完了: {count}件",
                        f"Detection complete: {count}",
                    )
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
        manual_bundle = _manual_review_bundle_paths(manual_review_copy)
        if over_detected:
            if manual_bundle is not None:
                edit_path, bbox_path, auto_path = manual_bundle
                _move_if_exists(manual_review_copy, edit_path)
                _move_if_exists(manual_review_annotated, bbox_path)
                if result.output is not None and result.output.exists():
                    auto_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(result.output, auto_path)
                return replace(result, manual_review_path=edit_path)
            return result

        # Re-running with overwrite enabled must also clean stale manual-review
        # artifacts when the latest result no longer needs them.
        if manual_bundle is not None:
            for path in manual_bundle:
                path.unlink(missing_ok=True)
        if result.count_mismatch:
            for path in (manual_review_copy, manual_review_annotated):
                if path is not None:
                    path.unlink(missing_ok=True)
            return replace(result, count_mismatch=False, manual_review_path=None)
        return result


def _manual_review_bundle_paths(manual_review_copy: Path | None) -> tuple[Path, Path, Path] | None:
    if manual_review_copy is None:
        return None
    original_path = Path(manual_review_copy)
    original_root = next(
        (
            parent
            for parent in original_path.parents
            if parent.name == "original" and parent.parent.name == "_manual_review"
        ),
        None,
    )
    if original_root is None:
        return None
    relative = original_path.relative_to(original_root)
    root = original_root.parent
    return (
        root / "edit" / relative,
        root / "reference_bbox" / relative,
        root / "auto_censored" / relative,
    )


def _move_if_exists(source: Path | None, destination: Path) -> None:
    if source is None:
        return
    source = Path(source)
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    source.replace(destination)


__all__ = [
    "BatchProcessor",
    "PipelineConfig",
    "JsonlRunLogger",
    "discover_images",
    "validate_processing_paths",
    "write_jsonl_log",
    "write_review_html",
]
