from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import PipelineConfig
from .control_panel import ControlPanel
from .settings_dialog import SettingsDialog


class _WheelValueGuard(QObject):
    """Keep mouse-wheel scrolling from silently changing settings controls."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() != QEvent.Type.Wheel:
            return False
        if not isinstance(watched, (QAbstractSpinBox, QComboBox)):
            return False

        self._forward_to_scroll_area(watched, event)
        event.accept()
        return True

    @staticmethod
    def _forward_to_scroll_area(watched: QWidget, event: QEvent) -> None:
        parent = watched.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                bar = parent.verticalScrollBar()
                pixel_delta = event.pixelDelta().y()
                if pixel_delta:
                    bar.setValue(bar.value() - pixel_delta)
                else:
                    steps = event.angleDelta().y() / 120.0
                    distance = max(20, bar.singleStep() * 3)
                    bar.setValue(bar.value() - round(steps * distance))
                return
            parent = parent.parentWidget()


def _install_wheel_guard(root: QWidget) -> _WheelValueGuard:
    guard = _WheelValueGuard(root)
    for widget in root.findChildren(QAbstractSpinBox):
        widget.installEventFilter(guard)
    for widget in root.findChildren(QComboBox):
        widget.installEventFilter(guard)
    return guard


class EnhancedControlPanel(ControlPanel):
    """Control panel with explicit persistence, reset, and wheel-safe editing."""

    def __init__(
        self,
        language: str = "ja",
        parent: QWidget | None = None,
        settings: QSettings | None = None,
    ):
        super().__init__(language, parent)
        self._settings_store = settings if settings is not None else QSettings(
            "CharacterMosaicCensor", "CharacterMosaicCensor"
        )

        self.settings_group = QGroupBox()
        settings_layout = QVBoxLayout(self.settings_group)
        self.settings_help = QLabel()
        self.settings_help.setWordWrap(True)
        self.settings_help.setObjectName("dialogNote")
        settings_layout.addWidget(self.settings_help)

        button_row = QHBoxLayout()
        self.save_settings_button = QPushButton()
        self.reset_settings_button = QPushButton()
        self.save_settings_button.clicked.connect(self._save_now)
        self.reset_settings_button.clicked.connect(self.reset_defaults)
        button_row.addWidget(self.save_settings_button)
        button_row.addWidget(self.reset_settings_button)
        settings_layout.addLayout(button_row)

        layout = self.widget().layout()
        layout.insertWidget(max(0, layout.count() - 1), self.settings_group)

        self._wheel_guard = _install_wheel_guard(self)
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self._save_now)
        self.retranslate()

    def retranslate(self) -> None:
        super().retranslate()
        if not hasattr(self, "settings_group"):
            return
        self.settings_group.setTitle(self._t("設定管理", "Settings"))
        self.settings_help.setText(
            self._t(
                "マウスホイールでは設定値は変わりません。Tabで項目移動、↑↓または直接入力で数値を変更できます。",
                "The mouse wheel no longer changes values. Use Tab to move, then Up/Down or direct typing to edit numbers.",
            )
        )
        self.save_settings_button.setText(self._t("設定を保存", "Save settings"))
        self.save_settings_button.setToolTip(self._t("Ctrl+Sでも保存できます。", "You can also save with Ctrl+S."))
        self.reset_settings_button.setText(self._t("初期設定に戻す", "Reset defaults"))
        self.reset_settings_button.setToolTip(
            self._t(
                "入力・出力フォルダと言語は残し、検出・モザイク・詳細設定だけ既定値へ戻します。",
                "Keeps folders and language, while restoring detection, censor, and advanced settings to defaults.",
            )
        )

    def _save_now(self) -> None:
        self.save_settings(self._settings_store)
        self._notify(self._t("設定を保存しました", "Settings saved"))

    def reset_defaults(self, checked: bool = False, *, confirm: bool = True) -> bool:
        del checked
        if confirm:
            answer = QMessageBox.question(
                self,
                self._t("初期設定に戻す", "Reset defaults"),
                self._t(
                    "検出・モザイク・詳細設定を初期値に戻します。入力・出力フォルダと言語はそのまま残します。よろしいですか？",
                    "Reset detection, censor, and advanced settings to defaults? Input/output folders and language will be kept.",
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False

        defaults = PipelineConfig(language=self._language)
        self.person_count.setValue(defaults.expected_person_count)
        self.confidence.setValue(defaults.detection_threshold)
        self.review_threshold.setValue(defaults.auto_threshold)
        self.padding_ratio.setValue(round(defaults.padding_ratio * 100))
        self.padding_px.setValue(defaults.padding_px)
        mode_index = self.mode.findData(defaults.mode)
        self.mode.setCurrentIndex(max(0, mode_index))
        self.strength.setValue(defaults.block_size)
        self.tile_detection.setChecked(defaults.tile_large_images)
        self.flip_tta.setChecked(defaults.flip_tta)
        self.review_save.setChecked(defaults.review_enabled)
        self.recursive.setChecked(defaults.recursive)
        self.overwrite.setChecked(defaults.overwrite)
        for key in self._advanced:
            self._advanced[key] = getattr(defaults, key)
        self._sync_review_enabled(self.review_save.isChecked())
        self._save_now()
        self._notify(self._t("初期設定に戻しました", "Defaults restored"))
        return True

    def _open_advanced(self) -> None:
        config = self.config()
        dialog = SettingsDialog(config, self._language, self)
        dialog._wheel_value_guard = _install_wheel_guard(dialog)  # type: ignore[attr-defined]
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

    def set_running(self, running: bool) -> None:
        super().set_running(running)
        if hasattr(self, "save_settings_button"):
            self.save_settings_button.setEnabled(not running)
            self.reset_settings_button.setEnabled(not running)
            self._save_shortcut.setEnabled(not running)

    def _notify(self, text: str) -> None:
        window = self.window()
        status_bar_getter = getattr(window, "statusBar", None)
        if callable(status_bar_getter):
            status_bar = status_bar_getter()
            if status_bar is not None:
                status_bar.showMessage(text, 3000)
                return
        self.set_summary(text)
