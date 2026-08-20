from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox

from ..pipeline import validate_processing_paths
from ..types import ProcessResult
from ..workers.batch_worker import BatchWorker
from .main_window import MainWindow


_DEGRADED_ANALYSIS_PREFIXES = (
    "partial:",
    "partial_disabled:",
    "unavailable:",
    "failed:",
    "disabled_after_failure",
)


class EnhancedMainWindow(MainWindow):
    """Main window UX additions layered over the stable worker implementation."""

    def __init__(self):
        super().__init__()
        open_input = getattr(self.controls, "open_input_requested", None)
        if open_input is not None:
            open_input.connect(self.open_input)
        file_dropped = getattr(self.preview, "file_dropped", None)
        if file_dropped is not None:
            file_dropped.connect(self.process_dropped_file)
        self._relabel_manual_review_button()

    def _on_language_changed(self, language: str) -> None:
        super()._on_language_changed(language)
        self._relabel_manual_review_button()

    def _relabel_manual_review_button(self) -> None:
        button = getattr(self.controls, "open_manual_review_button", None)
        if button is None:
            return
        button.setText(self._t("手動修正用画像を開く", "Open images for manual correction"))
        button.setToolTip(
            self._t(
                "BBoxもモザイクも入っていない元画像を開きます。reference_bboxは検出位置確認用、auto_censoredは自動処理結果の比較用です。",
                "Opens untouched originals with no BBox or censoring. reference_bbox is for detector reference and auto_censored is for comparison.",
            )
        )

    def open_input(self) -> None:
        text = self.controls.input_edit.text().strip()
        if text:
            self._open_local_path(Path(text))

    def open_manual_review(self) -> None:
        self.controls.ensure_output_default()
        text = self.controls.output_edit.text().strip()
        if text:
            self._open_local_path(Path(text) / "_manual_review" / "edit")

    def process_dropped_file(self, path_text: str) -> None:
        """Analyze one dropped image with the exact same processing pipeline."""

        if self._thread is not None:
            self.statusBar().showMessage(
                self._t(
                    "処理中は単体画像を追加できません。現在の処理終了後にもう一度ドロップしてください。",
                    "A single image cannot be added while processing is active. Drop it again after the current run finishes.",
                ),
                5000,
            )
            return

        source = Path(path_text).expanduser()
        if not source.is_file():
            return

        config = self.controls.config()
        output_dir = self._single_image_output_dir(source)
        review_dir = self._single_image_review_dir(output_dir, config.review_enabled)

        try:
            config.validate()
            input_dir, output_dir, review_dir = validate_processing_paths(
                source.parent,
                output_dir,
                review_dir if config.review_enabled else None,
            )
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, self._t("設定エラー", "Settings error"), str(exc))
            return

        self._save_settings()
        log_dir = output_dir.parent / "logs"
        self.preview.clear()
        self.controls.set_running(True)
        self.controls.set_summary(
            self._t(
                f"単体画像を処理中: {source.name}",
                f"Processing single image: {source.name}",
            )
        )
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.counter_label.setText("0 / 1")
        self.detect_label.setText("Detected: -")
        self.statusBar().showMessage(
            self._t(
                f"ドラッグ＆ドロップ画像を解析します → {output_dir}",
                f"Analyzing dropped image → {output_dir}",
            )
        )

        thread = QThread(self)
        worker = BatchWorker(
            input_dir,
            output_dir,
            review_dir,
            config,
            log_dir,
            images=[source],
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.preview.connect(self._on_preview)
        worker.discovered.connect(self._on_discovered)
        worker.progress.connect(self._on_progress)
        worker.runtime_ready.connect(self._on_runtime)
        worker.log_ready.connect(self._on_log_ready)
        worker.status.connect(self.statusBar().showMessage)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _single_image_output_dir(self, source: Path) -> Path:
        fixed_getter = getattr(self.controls, "_output_is_fixed", None)
        fixed = bool(fixed_getter()) if callable(fixed_getter) else False
        current = self.controls.output_edit.text().strip()
        if fixed and current:
            return Path(current).expanduser()
        return source.parent / "_censored"

    def _single_image_review_dir(self, output_dir: Path, enabled: bool) -> Path | None:
        if not enabled:
            return None
        current = self.controls.review_edit.text().strip()
        if bool(getattr(self.controls, "_review_custom", False)) and current:
            return Path(current).expanduser()
        return output_dir.parent / "review"

    def _on_progress(self, index: int, total: int, src: str, result: ProcessResult) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(index)
        self.counter_label.setText(f"{index} / {total}")
        if result.error:
            self.detect_label.setText(f"ERROR: {Path(src).name}")
            self.statusBar().showMessage(result.error)
            return
        if result.cancelled:
            self.detect_label.setText(f"CANCEL: {Path(src).name}")
            self.statusBar().showMessage(
                self._t(
                    "停止しました — 処理途中の画像は保存していません",
                    "Stopped — the incomplete image was not saved",
                )
            )
            return
        if result.skipped:
            self.detect_label.setText(f"SKIP: {Path(src).name}")
            return

        suppressed = len(result.anatomy_suppressed)
        evidence_review = sum(1 for item in result.candidate_evidence if item.decision == "review")
        filtered = (
            self._t(
                f" / 人体解析で除外 {suppressed}",
                f" / body-filtered {suppressed}",
            )
            if suppressed
            else ""
        )
        body_review = (
            self._t(
                f" / 人体解析Review {evidence_review}",
                f" / body-review {evidence_review}",
            )
            if evidence_review
            else ""
        )
        status = result.anatomy_filter_status
        degraded = status.startswith(_DEGRADED_ANALYSIS_PREFIXES)
        degraded_text = (
            self._t(" / 人体解析一部無効", " / body analysis degraded")
            if degraded
            else ""
        )

        if result.detections:
            top = max(result.detections, key=lambda d: d.score)
            review = " / Review" if result.review_required else ""
            over = self._t(" / 検出過多", " / Over-detected") if result.count_mismatch else ""
            self.detect_label.setText(
                f"Detected: {len(result.detections)} / {top.label} {top.score:.2f}"
                f"{review}{over}{filtered}{body_review}{degraded_text}"
            )
        else:
            review = " / Review" if result.review_required else ""
            self.detect_label.setText(
                self._t(
                    f"対象未検出{review}{filtered}{body_review}{degraded_text}",
                    f"No target detected{review}{filtered}{body_review}{degraded_text}",
                )
            )
        if degraded:
            self.statusBar().showMessage(
                self._t(
                    f"人体解析の一部を利用できません ({status})。安全側で通常検出を維持しています。",
                    f"Part of body analysis is unavailable ({status}); normal detection is kept fail-open.",
                )
            )

    def _on_finished(self, results: list[ProcessResult], log_path: str, stopped: bool) -> None:
        self._last_log_path = Path(log_path)
        errors = sum(1 for r in results if r.error)
        reviews = sum(1 for r in results if r.review_required)
        over = sum(1 for r in results if r.count_mismatch)
        detected = sum(
            1
            for r in results
            if r.detections and not r.error and not r.cancelled and not r.skipped
        )
        no_target = sum(
            1
            for r in results
            if not r.detections and not r.error and not r.cancelled and not r.skipped
        )
        skipped = sum(1 for r in results if r.skipped)
        suppressed = sum(len(r.anatomy_suppressed) for r in results)
        body_review = sum(
            sum(1 for item in r.candidate_evidence if item.decision == "review")
            for r in results
        )
        degraded = sum(
            1
            for r in results
            if r.anatomy_filter_status.startswith(_DEGRADED_ANALYSIS_PREFIXES)
        )
        state = self._t("停止", "Stopped") if stopped else self._t("完了", "Complete")
        summary = self._t(
            f"{state}: {len(results)}件 / 検出あり {detected}件 / 対象未検出 {no_target}件 / "
            f"人体解析で除外 {suppressed}件 / 人体解析Review {body_review}件 / "
            f"人体解析縮退 {degraded}件 / 検出過多 {over}件 / Review {reviews}件 / "
            f"Skip {skipped}件 / エラー {errors}件\nログ: {log_path}",
            f"{state}: {len(results)} image(s) / target detected {detected} / no target detected {no_target} / "
            f"body-filtered {suppressed} / body-review {body_review} / body-analysis degraded {degraded} / "
            f"over-detected {over} / Review {reviews} / skipped {skipped} / errors {errors}\nLog: {log_path}",
        )
        self.controls.set_summary(summary)
        self.statusBar().showMessage(summary.replace("\n", "  "))
        self.controls.set_running(False)
