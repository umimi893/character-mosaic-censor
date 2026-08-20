from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image

from .detector import AnimeCensorDetector, DetectorConfig
from .image_ops import apply_mosaic, draw_review_overlay, expand_box, normalize_image
from .pipeline_config import PipelineConfig
from .pipeline_review import (
    _load_review_manifest,
    _update_review_manifest_for_result,
    _write_review_manifest,
    write_review_html,
)
from .pipeline_storage import (
    _assert_directory_writable,
    _copy_file_atomic,
    _iter_images,
    _make_preview_image,
    _save_image_atomic,
)
from .types import Detection, PreviewFrame, ProcessResult

PreviewCallback = Callable[[PreviewFrame], None]

class BatchProcessor:
    def __init__(self, config: PipelineConfig | None = None, detector=None):
        self.config = config or PipelineConfig()
        self.config.validate()
        detector_cfg = DetectorConfig(
            detection_threshold=self.config.detection_threshold,
            model_level=self.config.model_level,
            model_version=self.config.model_version,
            iou_threshold=self.config.model_iou_threshold,
            merge_iou_threshold=self.config.merge_iou_threshold,
            merge_ios_threshold=self.config.merge_ios_threshold,
            tile_large_images=self.config.tile_large_images,
            tile_trigger_px=self.config.tile_trigger_px,
            tile_grid_3_trigger_px=self.config.tile_grid_3_trigger_px,
            tile_overlap=self.config.tile_overlap,
            flip_tta=self.config.flip_tta,
            female_only=self.config.female_only,
        )
        self.detector = detector or AnimeCensorDetector(detector_cfg)

    def discover_images(self, input_dir: Path, output_dir: Path, review_dir: Path | None = None) -> list[Path]:
        input_dir, output_dir, review_dir = validate_processing_paths(
            input_dir, output_dir, review_dir if self.config.review_enabled else None
        )
        excluded = {output_dir}
        if review_dir:
            excluded.add(review_dir)
        return list(_iter_images(input_dir, recursive=self.config.recursive, excluded_roots=excluded))

    def process_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        review_dir: Path | None = None,
        progress: Callable[[int, int, Path, ProcessResult | None], None] | None = None,
        preview: PreviewCallback | None = None,
        stop_requested: Callable[[], bool] | None = None,
        images: Sequence[Path] | None = None,
        result_callback: Callable[[ProcessResult], None] | None = None,
    ) -> list[ProcessResult]:
        input_dir, output_dir, review_dir = validate_processing_paths(
            input_dir, output_dir, review_dir if self.config.review_enabled else None
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if review_dir:
            review_dir.mkdir(parents=True, exist_ok=True)
        _assert_directory_writable(output_dir)
        if review_dir:
            _assert_directory_writable(review_dir)

        if images is None:
            excluded = {output_dir}
            if review_dir:
                excluded.add(review_dir)
            images = list(_iter_images(input_dir, recursive=self.config.recursive, excluded_roots=excluded))
        else:
            images = list(images)

        results: list[ProcessResult] = []
        review_manifest = _load_review_manifest(review_dir) if review_dir else None
        for index, src in enumerate(images, start=1):
            if stop_requested and stop_requested():
                break
            relative = src.resolve().relative_to(input_dir)
            dst = output_dir / relative
            review_path = (review_dir / relative) if review_dir else None
            result = self.process_file(
                src,
                dst,
                review_path,
                preview=preview,
                stop_requested=stop_requested,
            )
            results.append(result)
            if review_dir is not None and review_manifest is not None:
                if _update_review_manifest_for_result(review_manifest, result, review_dir, input_dir):
                    _write_review_manifest(review_dir, review_manifest)
            if result_callback:
                result_callback(result)
            if progress:
                progress(index, len(images), src, result)
            if result.cancelled or result.fatal_error:
                break

        if review_dir and self.config.generate_review_html:
            write_review_html(results, review_dir, input_dir=input_dir)
        return results

    def process_file(
        self,
        source: Path,
        output: Path,
        review_copy: Path | None = None,
        preview: PreviewCallback | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> ProcessResult:
        started = time.perf_counter()
        if output.exists() and not self.config.overwrite:
            return ProcessResult(source, output, tuple(), False, skipped=True, elapsed_seconds=time.perf_counter() - started)

        try:
            with Image.open(source) as opened:
                image = normalize_image(opened.copy())
            coordinate_size = image.size
            detector_preview = _make_preview_image(image, self.config.preview_max_side)

            if preview:
                preview(
                    PreviewFrame(
                        "original",
                        source,
                        detector_preview.copy(),
                        status=f"画像を読み込みました  {image.width}×{image.height}",
                        coordinate_size=coordinate_size,
                    )
                )

            last_preview_at = 0.0

            def detector_progress(pass_name: str, interim: tuple[Detection, ...]) -> None:
                nonlocal last_preview_at
                if not preview:
                    return
                now = time.monotonic()
                # Keep the UI reactive on fast GPUs without queuing dozens of
                # large preview signals. Always allow at least ~8 fps.
                if now - last_preview_at < 0.12:
                    return
                last_preview_at = now
                boxes = tuple(self._expanded_boxes(interim, coordinate_size))
                preview(
                    PreviewFrame(
                        "detecting",
                        source,
                        detector_preview.copy(),
                        detections=interim,
                        censor_boxes=boxes,
                        status=f"AI解析中: {pass_name}",
                        coordinate_size=coordinate_size,
                    )
                )

            try:
                detections = self._detect(image, detector_progress, stop_requested)
            except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
                return ProcessResult(
                    source,
                    None,
                    tuple(),
                    True,
                    error=f"{type(exc).__name__}: {exc}",
                    fatal_error=True,
                    elapsed_seconds=time.perf_counter() - started,
                )
            boxes = self._expanded_boxes(detections, coordinate_size)
            if stop_requested and stop_requested():
                if preview:
                    preview(
                        PreviewFrame(
                            "detected",
                            source,
                            detector_preview.copy(),
                            detections=tuple(detections),
                            censor_boxes=tuple(boxes),
                            status="停止: この画像は保存しません",
                            coordinate_size=coordinate_size,
                        )
                    )
                return ProcessResult(
                    source,
                    None,
                    tuple(detections),
                    False,
                    censor_boxes=tuple(boxes),
                    cancelled=True,
                    elapsed_seconds=time.perf_counter() - started,
                )
            if preview:
                preview(
                    PreviewFrame(
                        "detected",
                        source,
                        detector_preview.copy(),
                        detections=tuple(detections),
                        censor_boxes=tuple(boxes),
                        status=f"検出完了: {len(detections)}件",
                        coordinate_size=coordinate_size,
                    )
                )

            review_required = any(d.score < self.config.auto_threshold for d in detections)
            if not detections and self.config.copy_no_detection_to_review:
                review_required = True

            output.parent.mkdir(parents=True, exist_ok=True)
            if detections:
                censored = apply_mosaic(image, boxes, block_size=self.config.block_size, mode=self.config.mode)
                _save_image_atomic(censored, output, source.suffix.lower(), jpeg_quality=self.config.jpeg_quality)
            else:
                censored = image.copy()
                _copy_file_atomic(source, output)

            if preview:
                preview(
                    PreviewFrame(
                        "censored",
                        source,
                        _make_preview_image(censored, self.config.preview_max_side),
                        detections=tuple(detections),
                        censor_boxes=tuple(boxes),
                        status="モザイク適用後",
                        coordinate_size=coordinate_size,
                    )
                )

            saved_review: Path | None = None
            if review_copy and self.config.review_enabled and not review_required and review_copy.exists():
                # A successful re-run can promote a previously low-confidence
                # image out of Review; do not leave a stale annotated copy.
                review_copy.unlink(missing_ok=True)
            if review_copy and self.config.review_enabled and review_required and self.config.copy_low_confidence_to_review:
                review_copy.parent.mkdir(parents=True, exist_ok=True)
                annotated = draw_review_overlay(censored, detections, boxes, no_detection=not detections)
                _save_image_atomic(
                    annotated,
                    review_copy,
                    source.suffix.lower(),
                    jpeg_quality=self.config.jpeg_quality,
                )
                saved_review = review_copy

            return ProcessResult(
                source,
                output,
                tuple(detections),
                review_required,
                censor_boxes=tuple(boxes),
                review_path=saved_review,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            return ProcessResult(
                source,
                None,
                tuple(),
                True,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - started,
            )

    def _detect(self, image: Image.Image, progress_cb, stop_requested=None) -> list[Detection]:
        method = self.detector.detect
        parameters = inspect.signature(method).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        kwargs = {}
        if "progress" in parameters or accepts_kwargs:
            kwargs["progress"] = progress_cb
        if "stop_requested" in parameters or accepts_kwargs:
            kwargs["stop_requested"] = stop_requested
        return list(method(image, **kwargs))

    def _expanded_boxes(
        self,
        detections: Iterable[Detection],
        image_size: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        return [
            expand_box(
                d.box,
                image_size,
                padding_px=self.config.padding_px,
                padding_ratio=self.config.padding_ratio,
            )
            for d in detections
        ]
def validate_processing_paths(
    input_dir: Path,
    output_dir: Path,
    review_dir: Path | None,
) -> tuple[Path, Path, Path | None]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    review_dir = review_dir.expanduser().resolve() if review_dir else None

    if not input_dir.is_dir():
        raise ValueError("有効な入力フォルダを指定してください。")
    if input_dir == output_dir:
        raise ValueError("入力フォルダと出力フォルダは別にしてください。")
    if output_dir in input_dir.parents:
        raise ValueError("出力フォルダを入力フォルダの親には設定できません。入力画像が走査対象から外れます。")
    if review_dir is not None:
        if review_dir == input_dir:
            raise ValueError("Reviewフォルダを入力フォルダそのものには設定できません。")
        if review_dir == output_dir:
            raise ValueError("Reviewフォルダと出力フォルダは別にしてください。")
        if review_dir in input_dir.parents:
            raise ValueError("Reviewフォルダを入力フォルダの親には設定できません。入力画像が走査対象から外れます。")
    return input_dir, output_dir, review_dir


def discover_images(
    input_dir: Path,
    output_dir: Path,
    review_dir: Path | None,
    recursive: bool = True,
) -> list[Path]:
    input_dir, output_dir, review_dir = validate_processing_paths(input_dir, output_dir, review_dir)
    excluded = {output_dir}
    if review_dir:
        excluded.add(review_dir)
    return list(_iter_images(input_dir, recursive=recursive, excluded_roots=excluded))
