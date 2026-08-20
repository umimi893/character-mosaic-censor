from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import PipelineConfig
from .control_panel import ControlPanel
from .settings_dialog import SettingsDialog


class _WheelValueGuard(QObject):
    """Make the mouse wheel scroll only, never focus or change value controls."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() != QEvent.Type.Wheel:
            return False

        if isinstance(watched, (QAbstractSpinBox, QComboBox)):
            self._forward_to_scroll_area(watched, event)
            event.accept()
            return True

        if bool(watched.property("_wheel_selection_blocked")):
            event.accept()
            return True

        return False

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
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(guard)

    for widget in root.findChildren(QComboBox):
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(guard)

        view = widget.view()
        view.setProperty("_wheel_selection_blocked", True)
        view.installEventFilter(guard)
        viewport = view.viewport()
        viewport.setProperty("_wheel_selection_blocked", True)
        viewport.installEventFilter(guard)

    return guard


class _PathDropGuard(QObject):
    """Accept Explorer folder/image drops on path fields."""

    def __init__(self, panel: "EnhancedControlPanel"):
        super().__init__(panel)
        self.panel = panel

    @staticmethod
    def path_from_event(event: QEvent) -> str | None:
        mime_getter = getattr(event, "mimeData", None)
        if not callable(mime_getter):
            return None
        mime = mime_getter()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile()).expanduser()
            if path.is_dir():
                return str(path)
            if path.is_file():
                return str(path.parent)
        return None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if not isinstance(watched, QLineEdit):
            return False
        if event.type() in {QEvent.Type.DragEnter, QEvent.Type.DragMove}:
            if self.path_from_event(event):
                event.acceptProposedAction()
                return True
            return False
        if event.type() == QEvent.Type.Drop:
            path = self.path_from_event(event)
            if not path:
                return False
            if watched is self.panel.review_edit:
                self.panel._review_custom = True
            watched.setText(path)
            event.acceptProposedAction()
            return True
        return False


class EnhancedControlPanel(ControlPanel):
    """Control panel with persistence, safe editing, and folder UX helpers."""

    open_input_requested = Signal()

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

        self._install_path_actions()

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

    def _install_path_actions(self) -> None:
        io_layout = self.io_group.layout()
        specs = (
            (0, self.input_edit, self.open_input_requested.emit),
            (1, self.output_edit, self.open_output_requested.emit),
            (2, self.review_edit, self.open_review_requested.emit),
        )
        self._path_open_buttons: list[QPushButton] = []
        for row_index, edit, callback in specs:
            row_item = io_layout.itemAt(row_index)
            row = row_item.layout() if row_item is not None else None
            button = QPushButton()
            button.setFixedWidth(54)
            button.clicked.connect(callback)
            if row is not None:
                row.addWidget(button)
            self._path_open_buttons.append(button)
            edit.textChanged.connect(self._sync_path_open_buttons)

        self.input_open_button, self.output_open_inline_button, self.review_open_inline_button = self._path_open_buttons

        self.open_output_button.hide()
        self.open_review_button.hide()

        self._path_drop_guard = _PathDropGuard(self)
        for edit in (self.input_edit, self.output_edit, self.review_edit):
            edit.setAcceptDrops(True)
            edit.installEventFilter(self._path_drop_guard)
        self.review_save.toggled.connect(self._sync_path_open_buttons)
        self._sync_path_open_buttons()

    def _sync_path_open_buttons(self, *_args) -> None:
        if not hasattr(self, "_path_open_buttons"):
            return
        self.input_open_button.setEnabled(bool(self.input_edit.text().strip()))
        self.output_open_inline_button.setEnabled(bool(self.output_edit.text().strip()))
        self.review_open_inline_button.setEnabled(
            self.review_save.isChecked() and bool(self.review_edit.text().strip())
        )

    def retranslate(self) -> None:
        super().retranslate()
        if not hasattr(self, "settings_group"):
            return

        self.person_count_label.setText(self._t("画像内の最大人数", "Maximum people in image"))
        self.person_count.setToolTip(
            self._t(
                "この人数を超えて対象が検出された画像だけを『検出過多』として手動確認へ回します。0件は対象未検出として正常扱いします。",
                "Only images with more detections than this count are quarantined as over-detections. Zero detections are treated as no target detected, not as a failure.",
            )
        )
        self.open_manual_review_button.setText(
            self._t("検出過多・誤検出候補を開く", "Open over-detections / false-positive candidates")
        )

        for button in self._path_open_buttons:
            button.setText(self._t("開く", "Open"))
        self.input_open_button.setToolTip(self._t("入力フォルダを開きます。", "Open the input folder."))
        self.output_open_inline_button.setToolTip(self._t("出力フォルダを開きます。", "Open the output folder."))
        self.review_open_inline_button.setToolTip(self._t("Reviewフォルダを開きます。", "Open the Review folder."))
        drop_tip = self._t(
            "Explorerからフォルダ、または画像ファイルをドラッグ&ドロップできます。画像を落とした場合は、その画像のフォルダを設定します。",
            "Drag and drop a folder or image file from Explorer. Dropping an image selects its containing folder.",
        )
        for edit in (self.input_edit, self.output_edit, self.review_edit):
            edit.setToolTip(drop_tip)

        self.settings_group.setTitle(self._t("設定管理", "Settings"))
        self.settings_help.setText(
            self._t(
                "マウスホイールはスクロール専用です。設定欄は左クリックまたはTabで選択し、↑↓または直接入力で変更できます。",
                "The mouse wheel is scroll-only. Select a setting with left-click or Tab, then use Up/Down or direct typing to edit it.",
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

    def config(self) -> PipelineConfig:
        config = super().config()
        config.review_only_over_count = True
        return config

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
        self._sync_path_open_buttons()
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
        if hasattr(self, "_path_open_buttons"):
            self._sync_path_open_buttons()

    def _notify(self, text: str) -> None:
        window = self.window()
        status_bar_getter = getattr(window, "statusBar", None)
        if callable(status_bar_getter):
            status_bar = status_bar_getter()
            if status_bar is not None:
                status_bar.showMessage(text, 3000)
                return
        self.set_summary(text)
