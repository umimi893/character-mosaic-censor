from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import PipelineConfig


class SettingsDialog(QDialog):
    """Less frequently changed safety/accuracy settings."""

    def __init__(self, config: PipelineConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("詳細設定")
        self.setModal(True)
        self.setMinimumWidth(430)

        layout = QVBoxLayout(self)
        note = QLabel(
            "通常は変更不要です。RTX 5090 では Standard(s) + タイル検出を推奨します。"
        )
        note.setWordWrap(True)
        note.setObjectName("dialogNote")
        layout.addWidget(note)

        detection_group = QGroupBox("検出器")
        detection_form = QFormLayout(detection_group)
        self.model_level = QComboBox()
        self.model_level.addItem("Standard (s) — 精度優先", "s")
        self.model_level.addItem("Nano (n) — 速度優先", "n")
        self.model_iou = _float_spin(0.10, 0.95, config.model_iou_threshold, 0.05, 2)
        self.merge_iou = _float_spin(0.10, 0.95, config.merge_iou_threshold, 0.05, 2)
        self.merge_ios = _float_spin(0.10, 1.00, config.merge_ios_threshold, 0.05, 2)
        detection_form.addRow("Model", self.model_level)
        detection_form.addRow("Model NMS IoU", self.model_iou)
        detection_form.addRow("TTA/Tile Merge IoU", self.merge_iou)
        detection_form.addRow("Nested Merge IoS", self.merge_ios)
        layout.addWidget(detection_group)

        tile_group = QGroupBox("タイル")
        tile_form = QFormLayout(tile_group)
        self.tile_trigger = QSpinBox()
        self.tile_trigger.setRange(512, 10000)
        self.tile_trigger.setSuffix(" px")
        self.tile_3_trigger = QSpinBox()
        self.tile_3_trigger.setRange(512, 20000)
        self.tile_3_trigger.setSuffix(" px")
        self.tile_overlap = QSpinBox()
        self.tile_overlap.setRange(0, 40)
        self.tile_overlap.setSuffix(" %")
        tile_form.addRow("2×2開始", self.tile_trigger)
        tile_form.addRow("3×3開始", self.tile_3_trigger)
        tile_form.addRow("Overlap", self.tile_overlap)
        layout.addWidget(tile_group)

        operation_group = QGroupBox("運用")
        operation_form = QFormLayout(operation_group)
        self.review_no_detection = QCheckBox("未検出画像もReviewへ保存")
        self.preview_max_side = QSpinBox()
        self.preview_max_side.setRange(640, 4096)
        self.preview_max_side.setSingleStep(160)
        self.preview_max_side.setSuffix(" px")
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(70, 100)
        self.jpeg_quality.setSuffix(" %")
        operation_form.addRow("見逃し監査", self.review_no_detection)
        operation_form.addRow("Preview最大辺", self.preview_max_side)
        operation_form.addRow("JPEG/WebP品質", self.jpeg_quality)
        layout.addWidget(operation_group)

        reset = QPushButton("既定値に戻す")
        reset.clicked.connect(self._reset_defaults)
        layout.addWidget(reset)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load(config)

    def _load(self, config: PipelineConfig) -> None:
        index = self.model_level.findData(config.model_level)
        self.model_level.setCurrentIndex(max(0, index))
        self.model_iou.setValue(config.model_iou_threshold)
        self.merge_iou.setValue(config.merge_iou_threshold)
        self.merge_ios.setValue(config.merge_ios_threshold)
        self.tile_trigger.setValue(config.tile_trigger_px)
        self.tile_3_trigger.setValue(config.tile_grid_3_trigger_px)
        self.tile_overlap.setValue(round(config.tile_overlap * 100))
        self.review_no_detection.setChecked(config.copy_no_detection_to_review)
        self.preview_max_side.setValue(config.preview_max_side)
        self.jpeg_quality.setValue(config.jpeg_quality)

    def _reset_defaults(self) -> None:
        self._load(PipelineConfig())

    def _accept_if_valid(self) -> None:
        if self.tile_3_trigger.value() < self.tile_trigger.value():
            self.tile_3_trigger.setValue(self.tile_trigger.value())
        self.accept()

    def apply_to(self, config: PipelineConfig) -> PipelineConfig:
        config.model_level = str(self.model_level.currentData())
        config.model_iou_threshold = self.model_iou.value()
        config.merge_iou_threshold = self.merge_iou.value()
        config.merge_ios_threshold = self.merge_ios.value()
        config.tile_trigger_px = self.tile_trigger.value()
        config.tile_grid_3_trigger_px = self.tile_3_trigger.value()
        config.tile_overlap = self.tile_overlap.value() / 100.0
        config.copy_no_detection_to_review = self.review_no_detection.isChecked()
        config.preview_max_side = self.preview_max_side.value()
        config.jpeg_quality = self.jpeg_quality.value()
        return config


def _float_spin(low: float, high: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin
