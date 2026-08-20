from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QThread, QUrl, Qt
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
)

from ..detector import RuntimeInfo
from ..i18n import normalize_language, t
from ..pipeline import validate_processing_paths
from ..types import PreviewFrame, ProcessResult
from ..workers.batch_worker import BatchWorker
from .control_panel import ControlPanel
from .preview_widget import PreviewWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Character Mosaic Censor")
        self.resize(1320, 820)
        self.setMinimumSize(1040, 680)
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._pending_close = False
        self._settings = QSettings("CharacterMosaicCensor", "CharacterMosaicCensor")
        self._language = normalize_language(str(self._settings.value("ui/language", "ja")))
        self._last_log_path: Path | None = None

        self.preview = PreviewWidget(self._language)
        self.controls = ControlPanel(self._language)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.preview)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([930, 390])
        self.setCentralWidget(splitter)
        self._splitter = splitter

        self.counter_label = QLabel("0 / 0")
        self.detect_label = QLabel("Detected: -")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedWidth(260)
        status = QStatusBar()
        status.addWidget(self.counter_label)
        status.addWidget(self.detect_label, 1)
        status.addPermanentWidget(self.progress)
        self.setStatusBar(status)
        self.statusBar().showMessage(self._t("待機中", "Ready"))

        self.controls.language_changed.connect(self._on_language_changed)
        self.controls.start_requested.connect(self.start_processing)
        self.controls.stop_requested.connect(self.stop_processing)
        self.controls.browse_input_requested.connect(self.choose_input)
        self.controls.browse_output_requested.connect(self.choose_output)
        self.controls.browse_review_requested.connect(self.choose_review)
        self.controls.open_output_requested.connect(self.open_output)
        self.controls.open_review_requested.connect(self.open_review)
        self.controls.open_manual_review_requested.connect(self.open_manual_review)
        self.controls.open_logs_requested.connect(self.open_logs)

        self._restore_settings()

    def _t(self, ja: str, en: str) -> str:
        return t(self._language, ja, en)

    def _on_language_changed(self, language: str) -> None:
        self._language = normalize_language(language)
        self.preview.set_language(self._language)
        self._settings.setValue("ui/language", self._language)
        self._settings.sync()
        self.statusBar().showMessage(self._t("表示言語を日本語に変更しました", "Display language changed to English"))

    def _restore_settings(self) -> None:
        self.controls.load_settings(self._settings)
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self._settings.value("window/splitter")
        if splitter_state is not None:
            self._splitter.restoreState(splitter_state)

    def _save_settings(self) -> None:
        self.controls.save_settings(self._settings)
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/splitter", self._splitter.saveState())
        self._settings.sync()

    def choose_input(self) -> None:
        start = self.controls.input_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, self._t("入力フォルダ", "Input folder"), start)
        if path:
            self.controls.set_input_path(path)

    def choose_output(self) -> None:
        start = self.controls.output_edit.text().strip() or self.controls.input_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, self._t("出力フォルダ", "Output folder"), start)
        if path:
            self.controls.set_output_path(path)

    def choose_review(self) -> None:
        start = self.controls.review_edit.text().strip() or self.controls.output_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, self._t("Reviewフォルダ", "Review folder"), start)
        if path:
            self.controls.set_review_path(path)

    def open_output(self) -> None:
        text = self.controls.output_edit.text().strip()
        if not text:
            return
        self._open_local_path(Path(text))

    def open_review(self) -> None:
        review = self.controls.review_path()
        if review is None:
            return
        index = review / "index.html"
        self._open_local_path(index if index.exists() else review)

    def open_manual_review(self) -> None:
        self.controls.ensure_output_default()
        text = self.controls.output_edit.text().strip()
        if text:
            self._open_local_path(Path(text) / "_manual_review")

    def open_logs(self) -> None:
        if self._last_log_path is not None:
            self._open_local_path(self._last_log_path)
            return
        output_text = self.controls.output_edit.text().strip()
        if output_text:
            self._open_local_path(Path(output_text).expanduser().parent / "logs")

    def _on_log_ready(self, log_path: str) -> None:
        self._last_log_path = Path(log_path)

    @staticmethod
    def _open_local_path(path: Path) -> None:
        target = path if path.exists() else path.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def start_processing(self) -> None:
        if self._thread is not None:
            return
        if not self.controls.input_edit.text().strip():
            QMessageBox.critical(
                self,
                self._t("設定エラー", "Settings error"),
                self._t("入力フォルダを指定してください。", "Choose an input folder."),
            )
            return
        self.controls.ensure_output_default()
        if not self.controls.output_edit.text().strip():
            QMessageBox.critical(
                self,
                self._t("設定エラー", "Settings error"),
                self._t("出力フォルダを指定してください。", "Choose an output folder."),
            )
            return

        input_dir = self.controls.input_path()
        output_dir = self.controls.output_path()
        review_dir = self.controls.review_path()
        config = self.controls.config()

        try:
            config.validate()
            input_dir, output_dir, review_dir = validate_processing_paths(input_dir, output_dir, review_dir)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, self._t("設定エラー", "Settings error"), str(exc))
            return

        self._save_settings()
        log_dir = output_dir.parent / "logs"
        self.preview.clear()
        self.controls.set_running(True)
        self.controls.set_summary(self._t("処理中…", "Processing…"))
        self.progress.setRange(0, 0)  # indeterminate while scanning
        self.counter_label.setText(self._t("走査中", "Scanning"))
        self.detect_label.setText("Detected: -")
        self.statusBar().showMessage(self._t("準備中", "Preparing"))

        thread = QThread(self)
        worker = BatchWorker(input_dir, output_dir, review_dir, config, log_dir)
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

    def stop_processing(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        self.controls.stop_button.setEnabled(False)
        self.statusBar().showMessage(self._t(
            "停止要求中 — 現在の推論パス終了後に停止します",
            "Stop requested — processing will stop after the current inference pass",
        ))

    def _on_discovered(self, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        self.counter_label.setText(f"0 / {total}")
        if total == 0:
            self.statusBar().showMessage(self._t("処理対象画像がありません", "No images found"))
        else:
            self.statusBar().showMessage(self._t(f"{total}枚を処理します", f"Processing {total} image(s)"))

    def _on_preview(self, frame: PreviewFrame) -> None:
        self.preview.set_frame(frame)
        if frame.status:
            self.statusBar().showMessage(frame.status)

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
            self.statusBar().showMessage(self._t(
                "停止しました — 処理途中の画像は保存していません",
                "Stopped — the incomplete image was not saved",
            ))
            return
        if result.skipped:
            self.detect_label.setText(f"SKIP: {Path(src).name}")
            return
        if result.detections:
            top = max(result.detections, key=lambda d: d.score)
            review = " / Review" if result.review_required else ""
            manual = self._t(" / 人数不一致", " / Count mismatch") if result.count_mismatch else ""
            self.detect_label.setText(f"Detected: {top.label} {top.score:.2f}{review}{manual}")
        else:
            manual = self._t(" / 手動確認へ隔離", " / Quarantined") if result.count_mismatch else ""
            self.detect_label.setText(f"Detected: 0{manual}")

    def _on_runtime(self, info: RuntimeInfo) -> None:
        text = info.display_text
        if info.onnxruntime_version:
            text += f"\nONNX Runtime {info.onnxruntime_version}"
        if info.python_version:
            text += f" / Python {info.python_version}"
        if info.available_providers:
            text += "\nProviders: " + ", ".join(info.available_providers)
        if not info.cuda_available:
            text += self._t(
                "\n⚠ CUDAExecutionProvider がありません。CPU推論になります。",
                "\n⚠ CUDAExecutionProvider is unavailable. Inference will use the CPU.",
            )
        elif not info.using_cuda:
            text += self._t(
                "\n⚠ CUDAは利用可能ですが選択されていません。ONNX_MODE等を確認してください。",
                "\n⚠ CUDA is available but not selected. Check ONNX_MODE and the runtime configuration.",
            )
        self.controls.set_runtime_text(text)

    def _on_finished(self, results: list[ProcessResult], log_path: str, stopped: bool) -> None:
        self._last_log_path = Path(log_path)
        errors = sum(1 for r in results if r.error)
        reviews = sum(1 for r in results if r.review_required)
        mismatches = sum(1 for r in results if r.count_mismatch)
        detected = sum(1 for r in results if r.detections)
        skipped = sum(1 for r in results if r.skipped)
        state = self._t("停止", "Stopped") if stopped else self._t("完了", "Complete")
        summary = self._t(
            f"{state}: {len(results)}件 / 検出 {detected}件 / 人数不一致 {mismatches}件 / Review {reviews}件 / Skip {skipped}件 / エラー {errors}件\nログ: {log_path}",
            f"{state}: {len(results)} image(s) / detected {detected} / count mismatch {mismatches} / Review {reviews} / skipped {skipped} / errors {errors}\nLog: {log_path}",
        )
        self.controls.set_summary(summary)
        self.statusBar().showMessage(summary.replace("\n", "  "))
        self.controls.set_running(False)

    def _on_failed(self, message: str) -> None:
        self.controls.set_running(False)
        self.controls.set_summary(self._t(f"エラー: {message}", f"Error: {message}"))
        self.statusBar().showMessage(self._t("処理エラー", "Processing error"))
        QMessageBox.critical(self, self._t("処理エラー", "Processing error"), message)

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.controls.set_running(False)
        if self._pending_close:
            self._pending_close = False
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._worker is None:
            self._save_settings()
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            self._t("処理中", "Processing"),
            self._t(
                "処理中です。停止要求を出して、処理が止まってから終了しますか？",
                "Processing is active. Request a stop and close after the current pass finishes?",
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._pending_close = True
            self.stop_processing()
        event.ignore()


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Character Mosaic Censor")
    app.setOrganizationName("CharacterMosaicCensor")
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


_STYLE = """
QMainWindow { background: #11161d; }
QWidget { color: #dfe7ef; font-size: 13px; }
QScrollArea { background: #151b23; }
QGroupBox {
    border: 1px solid #303a46;
    border-radius: 9px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
    background: #171e27;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #dce6f0; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0f141b; border: 1px solid #34404e; border-radius: 6px; padding: 6px; min-height: 22px;
}
QPushButton {
    background: #252f3b; border: 1px solid #3a4756; border-radius: 7px; padding: 7px 10px;
}
QPushButton:hover { background: #2d3947; }
QPushButton:disabled { color: #687380; background: #1b222a; }
QPushButton#startButton { background: #176b4e; border-color: #228465; font-weight: 700; }
QPushButton#startButton:hover { background: #1c7a5a; }
QPushButton#stopButton { background: #69343b; border-color: #84454f; font-weight: 700; }
QPushButton[previewMode="true"]:checked { background: #24597d; border-color: #3476a3; }
QLabel#panelTitle { font-size: 21px; font-weight: 750; }
QLabel#panelSubtitle { color: #8d9aa8; margin-bottom: 3px; }
QLabel#runtimeLabel, QLabel#summaryLabel, QLabel#previewStatus, QLabel#dialogNote { color: #9eabb8; }
QLabel#previewFile { font-weight: 600; }
QProgressBar { border: 1px solid #34404e; border-radius: 5px; text-align: center; background: #0f141b; }
QProgressBar::chunk { background: #2c8d68; border-radius: 4px; }
QStatusBar { background: #0d1218; border-top: 1px solid #28313c; }
QSplitter::handle { background: #27303b; width: 1px; }
"""
