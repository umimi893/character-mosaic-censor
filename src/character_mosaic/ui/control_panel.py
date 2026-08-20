from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import PipelineConfig
from .settings_dialog import SettingsDialog


class ControlPanel(QScrollArea):
    start_requested = Signal()
    stop_requested = Signal()
    browse_input_requested = Signal()
    browse_output_requested = Signal()
    browse_review_requested = Signal()
    open_output_requested = Signal()
    open_review_requested = Signal()
    open_logs_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumWidth(350)
        self.setMaximumWidth(450)
        self._review_custom = False
        self._running = False
        defaults = PipelineConfig()
        self._advanced = {
            "tile_trigger_px": defaults.tile_trigger_px,
            "tile_grid_3_trigger_px": defaults.tile_grid_3_trigger_px,
            "tile_overlap": defaults.tile_overlap,
            "model_level": defaults.model_level,
            "model_version": defaults.model_version,
            "model_iou_threshold": defaults.model_iou_threshold,
            "merge_iou_threshold": defaults.merge_iou_threshold,
            "merge_ios_threshold": defaults.merge_ios_threshold,
            "copy_no_detection_to_review": defaults.copy_no_detection_to_review,
            "preview_max_side": defaults.preview_max_side,
            "jpeg_quality": defaults.jpeg_quality,
        }

        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        title = QLabel("Character Mosaic Censor")
        title.setObjectName("panelTitle")
        subtitle = QLabel("ローカルAI 自動モザイク")
        subtitle.setObjectName("panelSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        io_group = QGroupBox("処理フォルダ")
        io_layout = QVBoxLayout(io_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.review_edit = QLineEdit()
        self.input_edit.setPlaceholderText("入力画像フォルダ")
        self.output_edit.setPlaceholderText("出力フォルダ")
        self.review_edit.setPlaceholderText("Reviewフォルダ")
        input_row, self.input_browse = self._path_row("入力", self.input_edit, self.browse_input_requested.emit)
        output_row, self.output_browse = self._path_row("出力", self.output_edit, self.browse_output_requested.emit)
        review_row, self.review_browse = self._path_row("Review", self.review_edit, self._browse_review)
        io_layout.addLayout(input_row)
        io_layout.addLayout(output_row)
        io_layout.addLayout(review_row)
        self.output_edit.textChanged.connect(self._update_review_default)
        self.review_edit.textEdited.connect(self._mark_review_custom)
        layout.addWidget(io_group)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("▶  実行")
        self.start_button.setObjectName("startButton")
        self.start_button.setMinimumHeight(44)
        self.stop_button = QPushButton("■  停止")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setMinimumHeight(44)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.stop_button, 1)
        layout.addLayout(buttons)

        detect_group = QGroupBox("検出設定")
        detect_form = QFormLayout(detect_group)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.95)
        self.confidence.setSingleStep(0.01)
        self.confidence.setDecimals(2)
        self.confidence.setValue(defaults.detection_threshold)
        self.review_threshold = QDoubleSpinBox()
        self.review_threshold.setRange(0.01, 0.99)
        self.review_threshold.setSingleStep(0.01)
        self.review_threshold.setDecimals(2)
        self.review_threshold.setValue(defaults.auto_threshold)
        self.padding_ratio = QSpinBox()
        self.padding_ratio.setRange(0, 100)
        self.padding_ratio.setSuffix(" %")
        self.padding_ratio.setValue(round(defaults.padding_ratio * 100))
        self.padding_px = QSpinBox()
        self.padding_px.setRange(0, 300)
        self.padding_px.setSuffix(" px")
        self.padding_px.setValue(defaults.padding_px)
        detect_form.addRow("Confidence", self.confidence)
        detect_form.addRow("Review threshold", self.review_threshold)
        detect_form.addRow("Padding", self.padding_ratio)
        detect_form.addRow("固定余白", self.padding_px)
        layout.addWidget(detect_group)

        mosaic_group = QGroupBox("モザイク設定")
        mosaic_form = QFormLayout(mosaic_group)
        self.mode = QComboBox()
        self.mode.addItem("Mosaic", "mosaic")
        self.mode.addItem("Blur", "blur")
        self.mode.addItem("Black", "black")
        self.strength = QSpinBox()
        self.strength.setRange(2, 128)
        self.strength.setValue(defaults.block_size)
        mosaic_form.addRow("Mode", self.mode)
        mosaic_form.addRow("Strength", self.strength)
        layout.addWidget(mosaic_group)

        options_group = QGroupBox("見逃し対策")
        options_layout = QVBoxLayout(options_group)
        self.tile_detection = QCheckBox("タイル検出")
        self.tile_detection.setChecked(defaults.tile_large_images)
        self.flip_tta = QCheckBox("左右反転TTA")
        self.flip_tta.setChecked(defaults.flip_tta)
        self.review_save = QCheckBox("Review保存 + HTML一覧")
        self.review_save.setChecked(defaults.review_enabled)
        self.recursive = QCheckBox("サブフォルダも処理")
        self.recursive.setChecked(defaults.recursive)
        self.overwrite = QCheckBox("既存出力を上書き")
        self.overwrite.setChecked(defaults.overwrite)
        self.review_save.toggled.connect(self._sync_review_enabled)
        for widget in (self.tile_detection, self.flip_tta, self.review_save, self.recursive, self.overwrite):
            options_layout.addWidget(widget)
        self.advanced_button = QPushButton("詳細設定…")
        self.advanced_button.clicked.connect(self._open_advanced)
        options_layout.addWidget(self.advanced_button)
        layout.addWidget(options_group)

        runtime_group = QGroupBox("GPU / Runtime")
        runtime_layout = QVBoxLayout(runtime_group)
        self.runtime_label = QLabel("実行開始時に確認します")
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setObjectName("runtimeLabel")
        runtime_layout.addWidget(self.runtime_label)
        layout.addWidget(runtime_group)

        self.summary_label = QLabel("待機中")
        self.summary_label.setObjectName("summaryLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        open_row = QHBoxLayout()
        self.open_output_button = QPushButton("出力を開く")
        self.open_review_button = QPushButton("Reviewを開く")
        self.open_logs_button = QPushButton("ログを開く")
        self.open_output_button.clicked.connect(self.open_output_requested.emit)
        self.open_review_button.clicked.connect(self.open_review_requested.emit)
        self.open_logs_button.clicked.connect(self.open_logs_requested.emit)
        open_row.addWidget(self.open_output_button)
        open_row.addWidget(self.open_review_button)
        open_row.addWidget(self.open_logs_button)
        layout.addLayout(open_row)
        layout.addStretch(1)

    def _path_row(self, label: str, edit: QLineEdit, callback) -> tuple[QHBoxLayout, QPushButton]:
        row = QHBoxLayout()
        row.setSpacing(6)
        prefix = QLabel(label)
        prefix.setFixedWidth(48)
        button = QPushButton("参照")
        button.setFixedWidth(54)
        button.clicked.connect(callback)
        row.addWidget(prefix)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row, button

    def _browse_review(self) -> None:
        self._review_custom = True
        self.browse_review_requested.emit()

    def _mark_review_custom(self, _text: str) -> None:
        self._review_custom = True

    def _update_review_default(self, text: str) -> None:
        if self._review_custom or not text.strip():
            return
        out = Path(text.strip())
        self.review_edit.setText(str(out.parent / "review"))

    def _sync_review_enabled(self, enabled: bool) -> None:
        usable = enabled and not self._running
        self.review_edit.setEnabled(usable)
        self.review_browse.setEnabled(usable)
        self.open_review_button.setEnabled(enabled)

    def _open_advanced(self) -> None:
        config = self.config()
        dialog = SettingsDialog(config, self)
        if dialog.exec():
            config = dialog.apply_to(config)
            self._advanced.update(
                {
                    "tile_trigger_px": config.tile_trigger_px,
                    "tile_grid_3_trigger_px": config.tile_grid_3_trigger_px,
                    "tile_overlap": config.tile_overlap,
                    "model_level": config.model_level,
                    "model_version": config.model_version,
                    "model_iou_threshold": config.model_iou_threshold,
                    "merge_iou_threshold": config.merge_iou_threshold,
                    "merge_ios_threshold": config.merge_ios_threshold,
                    "copy_no_detection_to_review": config.copy_no_detection_to_review,
                    "preview_max_side": config.preview_max_side,
                    "jpeg_quality": config.jpeg_quality,
                }
            )

    def set_input_path(self, path: str) -> None:
        self.input_edit.setText(path)
        if not self.output_edit.text().strip():
            self.output_edit.setText(str(Path(path) / "_censored"))

    def set_output_path(self, path: str) -> None:
        self.output_edit.setText(path)

    def set_review_path(self, path: str) -> None:
        self._review_custom = True
        self.review_edit.setText(path)

    def input_path(self) -> Path:
        return Path(self.input_edit.text().strip())

    def output_path(self) -> Path:
        return Path(self.output_edit.text().strip())

    def review_path(self) -> Path | None:
        if not self.review_save.isChecked() or not self.review_edit.text().strip():
            return None
        return Path(self.review_edit.text().strip())

    def config(self) -> PipelineConfig:
        return PipelineConfig(
            detection_threshold=self.confidence.value(),
            auto_threshold=self.review_threshold.value(),
            padding_px=self.padding_px.value(),
            padding_ratio=self.padding_ratio.value() / 100.0,
            block_size=self.strength.value(),
            mode=str(self.mode.currentData()),
            recursive=self.recursive.isChecked(),
            overwrite=self.overwrite.isChecked(),
            tile_large_images=self.tile_detection.isChecked(),
            tile_trigger_px=int(self._advanced["tile_trigger_px"]),
            tile_grid_3_trigger_px=int(self._advanced["tile_grid_3_trigger_px"]),
            tile_overlap=float(self._advanced["tile_overlap"]),
            flip_tta=self.flip_tta.isChecked(),
            female_only=True,
            model_level=str(self._advanced["model_level"]),
            model_version=str(self._advanced["model_version"]),
            model_iou_threshold=float(self._advanced["model_iou_threshold"]),
            merge_iou_threshold=float(self._advanced["merge_iou_threshold"]),
            merge_ios_threshold=float(self._advanced["merge_ios_threshold"]),
            review_enabled=self.review_save.isChecked(),
            copy_low_confidence_to_review=True,
            copy_no_detection_to_review=bool(self._advanced["copy_no_detection_to_review"]),
            generate_review_html=self.review_save.isChecked(),
            preview_max_side=int(self._advanced["preview_max_side"]),
            jpeg_quality=int(self._advanced["jpeg_quality"]),
        )

    def set_running(self, running: bool) -> None:
        self._running = running
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        for widget in (
            self.input_edit,
            self.output_edit,
            self.review_edit,
            self.confidence,
            self.review_threshold,
            self.padding_ratio,
            self.padding_px,
            self.mode,
            self.strength,
            self.tile_detection,
            self.flip_tta,
            self.review_save,
            self.recursive,
            self.overwrite,
            self.advanced_button,
            self.input_browse,
            self.output_browse,
        ):
            widget.setEnabled(not running)
        self._sync_review_enabled(self.review_save.isChecked())

    def set_runtime_text(self, text: str) -> None:
        self.runtime_label.setText(text)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def save_settings(self, settings: QSettings) -> None:
        settings.setValue("paths/input", self.input_edit.text())
        settings.setValue("paths/output", self.output_edit.text())
        settings.setValue("paths/review", self.review_edit.text())
        settings.setValue("paths/review_custom", self._review_custom)
        settings.setValue("detect/confidence", self.confidence.value())
        settings.setValue("detect/review_threshold", self.review_threshold.value())
        settings.setValue("detect/padding_ratio", self.padding_ratio.value())
        settings.setValue("detect/padding_px", self.padding_px.value())
        settings.setValue("mosaic/mode", self.mode.currentData())
        settings.setValue("mosaic/strength", self.strength.value())
        settings.setValue("options/tile", self.tile_detection.isChecked())
        settings.setValue("options/flip", self.flip_tta.isChecked())
        settings.setValue("options/review", self.review_save.isChecked())
        settings.setValue("options/recursive", self.recursive.isChecked())
        settings.setValue("options/overwrite", self.overwrite.isChecked())
        for key, value in self._advanced.items():
            settings.setValue(f"advanced/{key}", value)
        settings.sync()

    def load_settings(self, settings: QSettings) -> None:
        self.input_edit.setText(str(settings.value("paths/input", "")))
        self.output_edit.setText(str(settings.value("paths/output", "")))
        self.review_edit.setText(str(settings.value("paths/review", "")))
        self._review_custom = _as_bool(settings.value("paths/review_custom", False))
        self.confidence.setValue(float(settings.value("detect/confidence", self.confidence.value())))
        self.review_threshold.setValue(float(settings.value("detect/review_threshold", self.review_threshold.value())))
        self.padding_ratio.setValue(int(settings.value("detect/padding_ratio", self.padding_ratio.value())))
        self.padding_px.setValue(int(settings.value("detect/padding_px", self.padding_px.value())))
        mode = str(settings.value("mosaic/mode", self.mode.currentData()))
        idx = self.mode.findData(mode)
        if idx >= 0:
            self.mode.setCurrentIndex(idx)
        self.strength.setValue(int(settings.value("mosaic/strength", self.strength.value())))
        self.tile_detection.setChecked(_as_bool(settings.value("options/tile", self.tile_detection.isChecked())))
        self.flip_tta.setChecked(_as_bool(settings.value("options/flip", self.flip_tta.isChecked())))
        self.review_save.setChecked(_as_bool(settings.value("options/review", self.review_save.isChecked())))
        self.recursive.setChecked(_as_bool(settings.value("options/recursive", self.recursive.isChecked())))
        self.overwrite.setChecked(_as_bool(settings.value("options/overwrite", self.overwrite.isChecked())))

        defaults = PipelineConfig()
        int_keys = {"tile_trigger_px", "tile_grid_3_trigger_px", "preview_max_side", "jpeg_quality"}
        float_keys = {"tile_overlap", "model_iou_threshold", "merge_iou_threshold", "merge_ios_threshold"}
        bool_keys = {"copy_no_detection_to_review"}
        for key in self._advanced:
            default = getattr(defaults, key)
            raw = settings.value(f"advanced/{key}", default)
            if key in int_keys:
                value = int(raw)
            elif key in float_keys:
                value = float(raw)
            elif key in bool_keys:
                value = _as_bool(raw)
            else:
                value = str(raw)
            self._advanced[key] = value
        self._sync_review_enabled(self.review_save.isChecked())


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
