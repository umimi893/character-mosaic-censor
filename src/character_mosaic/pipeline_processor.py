from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image

from .detector import AnimeCensorDetector, DetectorConfig
from .image_ops import apply_mosaic, draw_review_overlay, expand_box, normalize_image
from .i18n import t
from .pipeline_config import PipelineConfig
from .pipeline_review import _load_review_manifest, _update_review_manifest_for_result, _write_review_manifest, write_review_html
from .pipeline_storage import _assert_directory_writable, _copy_file_atomic, _iter_images, _make_preview_image, _save_image_atomic
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
        input_dir, output_dir, review_dir = validate_processing_paths(input_dir, output_dir, review_dir if self.config.review_enabled else None)
        excluded = {output_dir}
        if review_dir:
            excluded.add(review_dir)
        return list(_iter_images(input_dir, recursive=self.config.recursive, excluded_roots=excluded))

    def process_folder(self, input_dir: Path, output_dir: Path, review_dir: Path | None = None,
                       progress: Callable[[int, int, Path, ProcessResult | None], None] | None = None,
                       preview: PreviewCallback | None = None,
                       stop_requested: Callable[[], bool] | None = None,
                       images: Sequence[Path] | None = None,
                       result_callback: Callable[[ProcessResult], None] | None = None) -> list[ProcessResult]:
        input_dir, output_dir, review_dir = validate_processing_paths(input_dir, output_dir, review_dir if self.config.review_enabled else None)
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
            manual_root = output_dir / "_manual_review"
            result = self.process_file(
                src, dst, review_path,
                manual_review_copy=manual_root / "original" / relative,
                manual_review_annotated=manual_root / "annotated" / relative,
                preview=preview, stop_requested=stop_requested,
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

    def process_file(self, source: Path, output: Path, review_copy: Path | None = None,
                     manual_review_copy: Path | None = None, manual_review_annotated: Path | None = None,
                     preview: PreviewCallback | None = None,
                     stop_requested: Callable[[], bool] | None = None) -> ProcessResult:
        started = time.perf_counter()
        if output.exists() and not self.config.overwrite:
            return ProcessResult(source, output, tuple(), False, skipped=True, elapsed_seconds=time.perf_counter() - started)
        try:
            with Image.open(source) as opened:
                image = normalize_image(opened.copy())
            coordinate_size = image.size
            detector_preview = _make_preview_image(image, self.config.preview_max_side)
            if preview:
                preview(PreviewFrame("original", source, detector_preview.copy(),
                    status=t(self.config.language, f"画像を読み込みました  {image.width}×{image.height}", f"Image loaded  {image.width}×{image.height}"),
                    coordinate_size=coordinate_size))

            last_preview_at = 0.0
            def detector_progress(pass_name: str, interim: tuple[Detection, ...]) -> None:
                nonlocal last_preview_at
                if not preview:
                    return
                now = time.monotonic()
                if now - last_preview_at < 0.12:
                    return
                last_preview_at = now
                preview(PreviewFrame("detecting", source, detector_preview.copy(), detections=interim,
                    censor_boxes=tuple(self._expanded_boxes(interim, coordinate_size)),
                    status=t(self.config.language, f"AI解析中: {pass_name}", f"Analyzing: {pass_name}"),
                    coordinate_size=coordinate_size))

            try:
                detections = self._detect(image, detector_progress, stop_requested)
            except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
                return ProcessResult(source, None, tuple(), True, error=f"{type(exc).__name__}: {exc}", fatal_error=True,
                                     elapsed_seconds=time.perf_counter() - started)

            analysis = self._analysis_snapshot()
            boxes = self._expanded_boxes(detections, coordinate_size)
            count_mismatch = len(detections) != self.config.expected_person_count
            preview_fields = self._analysis_preview_fields(analysis)
            result_fields = self._analysis_result_fields(analysis)

            if stop_requested and stop_requested():
                if preview:
                    preview(PreviewFrame("detected", source, detector_preview.copy(), detections=tuple(detections),
                        censor_boxes=tuple(boxes), status=t(self.config.language, "停止: この画像は保存しません", "Stopped: this incomplete image was not saved"),
                        coordinate_size=coordinate_size, **preview_fields))
                return ProcessResult(source, None, tuple(detections), False, censor_boxes=tuple(boxes), cancelled=True,
                                     elapsed_seconds=time.perf_counter() - started, **result_fields)

            if preview and analysis is not None:
                suppressed = len(getattr(analysis, "suppressed", ()))
                reviews = sum(1 for e in getattr(analysis, "evidence", ()) if e.decision == "review")
                preview(PreviewFrame("analysis", source, detector_preview.copy(), detections=tuple(detections), censor_boxes=tuple(boxes),
                    status=t(self.config.language, f"人体解析: 除外 {suppressed} / Review候補 {reviews}", f"Body analysis: suppressed {suppressed} / review candidates {reviews}"),
                    coordinate_size=coordinate_size, **preview_fields))

            if preview:
                status = (t(self.config.language, f"検出完了: {len(detections)}件", f"Detection complete: {len(detections)}")
                          if not count_mismatch else
                          t(self.config.language, f"要手動確認: 人数 {self.config.expected_person_count} / 検出 {len(detections)}", f"Manual review: expected {self.config.expected_person_count} / detected {len(detections)}"))
                preview(PreviewFrame("detected", source, detector_preview.copy(), detections=tuple(detections), censor_boxes=tuple(boxes),
                                     status=status, coordinate_size=coordinate_size, **preview_fields))

            review_required = any(d.score < self.config.auto_threshold for d in detections)
            detector_review = getattr(self.detector, "requires_review", False)
            if callable(detector_review):
                detector_review = detector_review()
            review_required = review_required or bool(detector_review)
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
                preview(PreviewFrame("censored", source, _make_preview_image(censored, self.config.preview_max_side),
                    detections=tuple(detections), censor_boxes=tuple(boxes), status=t(self.config.language, "モザイク適用後", "Censor effect applied"),
                    coordinate_size=coordinate_size, **preview_fields))

            saved_review: Path | None = None
            if review_copy and self.config.review_enabled and not review_required and review_copy.exists():
                review_copy.unlink(missing_ok=True)
            if review_copy and self.config.review_enabled and review_required and self.config.copy_low_confidence_to_review:
                review_copy.parent.mkdir(parents=True, exist_ok=True)
                annotated = draw_review_overlay(censored, detections, boxes, no_detection=not detections)
                _save_image_atomic(annotated, review_copy, source.suffix.lower(), jpeg_quality=self.config.jpeg_quality)
                saved_review = review_copy

            saved_manual: Path | None = None
            if manual_review_copy and count_mismatch:
                manual_review_copy.parent.mkdir(parents=True, exist_ok=True)
                _copy_file_atomic(source, manual_review_copy)
                saved_manual = manual_review_copy
                if manual_review_annotated:
                    manual_review_annotated.parent.mkdir(parents=True, exist_ok=True)
                    annotated = draw_review_overlay(image, detections, boxes, no_detection=not detections)
                    _save_image_atomic(annotated, manual_review_annotated, source.suffix.lower(), jpeg_quality=self.config.jpeg_quality)
            elif manual_review_copy:
                manual_review_copy.unlink(missing_ok=True)
                if manual_review_annotated:
                    manual_review_annotated.unlink(missing_ok=True)

            return ProcessResult(source, output, tuple(detections), review_required, censor_boxes=tuple(boxes), review_path=saved_review,
                                 count_mismatch=count_mismatch, manual_review_path=saved_manual,
                                 elapsed_seconds=time.perf_counter() - started, **result_fields)
        except Exception as exc:
            return ProcessResult(source, None, tuple(), True, error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)

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

    def _analysis_snapshot(self):
        return getattr(self.detector, "last_filter_result", None)

    @staticmethod
    def _analysis_preview_fields(analysis) -> dict:
        if analysis is None:
            return {}
        return {"body_regions": tuple(getattr(analysis, "body_regions", ())), "pose_points": tuple(getattr(analysis, "pose_points", ())),
                "pose_edges": tuple(getattr(analysis, "pose_edges", ())), "candidate_evidence": tuple(getattr(analysis, "evidence", ())),
                "analysis_status": str(getattr(analysis, "status", ""))}

    @staticmethod
    def _analysis_result_fields(analysis) -> dict:
        if analysis is None:
            return {}
        suppressed = tuple(getattr(analysis, "suppressed", ()))
        return {"anatomy_suppressed": tuple(item.detection for item in suppressed),
                "anatomy_suppression_reasons": tuple(item.log_reason for item in suppressed),
                "anatomy_filter_status": str(getattr(analysis, "status", "")),
                "body_regions": tuple(getattr(analysis, "body_regions", ())), "pose_points": tuple(getattr(analysis, "pose_points", ())),
                "pose_edges": tuple(getattr(analysis, "pose_edges", ())), "candidate_evidence": tuple(getattr(analysis, "evidence", ()))}

    def _expanded_boxes(self, detections: Iterable[Detection], image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        return [expand_box(d.box, image_size, padding_px=self.config.padding_px, padding_ratio=self.config.padding_ratio) for d in detections]


def validate_processing_paths(input_dir: Path, output_dir: Path, review_dir: Path | None) -> tuple[Path, Path, Path | None]:
    input_dir = input_dir.expanduser().resolve(); output_dir = output_dir.expanduser().resolve(); review_dir = review_dir.expanduser().resolve() if review_dir else None
    if not input_dir.is_dir(): raise ValueError("有効な入力フォルダを指定してください。")
    if input_dir == output_dir: raise ValueError("入力フォルダと出力フォルダは別にしてください。")
    if output_dir in input_dir.parents: raise ValueError("出力フォルダを入力フォルダの親には設定できません。入力画像が走査対象から外れます。")
    if review_dir is not None:
        if review_dir == input_dir: raise ValueError("Reviewフォルダを入力フォルダそのものには設定できません。")
        if review_dir == output_dir: raise ValueError("Reviewフォルダと出力フォルダは別にしてください。")
        if review_dir in input_dir.parents: raise ValueError("Reviewフォルダを入力フォルダの親には設定できません。入力画像が走査対象から外れます。")
    return input_dir, output_dir, review_dir


def discover_images(input_dir: Path, output_dir: Path, review_dir: Path | None, recursive: bool = True) -> list[Path]:
    input_dir, output_dir, review_dir = validate_processing_paths(input_dir, output_dir, review_dir)
    excluded = {output_dir}
    if review_dir: excluded.add(review_dir)
    return list(_iter_images(input_dir, recursive=recursive, excluded_roots=excluded))
