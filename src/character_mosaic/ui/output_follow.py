from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QCheckBox, QPushButton, QWidget

from ..output_paths import default_output_for_input, output_differs_from_default
from ..pipeline_config import PipelineConfig
from .settings_safety import EnhancedControlPanel


_OVERWRITE_DEFAULT_MIGRATION = "migrations/overwrite_default_v130"
_GUI_LEARNING_DEFAULT = True


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OutputFollowControlPanel(EnhancedControlPanel):
    """Control panel with output-follow and automatic learning UX."""

    learning_requested = Signal()
    _output_fixed_state = False

    def __init__(
        self,
        language: str = "ja",
        parent: QWidget | None = None,
        settings: QSettings | None = None,
    ):
        super().__init__(language, parent, settings)

        # Fresh GUI sessions should be immediately rerunnable even before a
        # QSettings load has occurred.
        self.overwrite.setChecked(True)

        self._output_fixed_state = False
        self.output_fixed = QCheckBox()
        io_layout = self.io_group.layout()
        # Input row, output row, [lock], review row.
        io_layout.insertWidget(2, self.output_fixed)

        # Learning capture is intentionally separate from the corpus miner:
        # normal processing can quietly accumulate candidate evidence while the
        # miner is launched explicitly for large legacy folders/ZIP archives.
        self.learning_capture = QCheckBox()
        self.learning_capture.setChecked(_GUI_LEARNING_DEFAULT)
        self.learning_button = QPushButton()
        options_layout = self.options_group.layout()
        insert_at = max(0, options_layout.count() - 1)
        options_layout.insertWidget(insert_at, self.learning_capture)
        options_layout.insertWidget(insert_at + 1, self.learning_button)
        self.learning_button.clicked.connect(self.learning_requested.emit)

        self.output_edit.textEdited.connect(self._fix_output_after_manual_edit)
        self.output_fixed.toggled.connect(self._on_output_fixed_toggled)
        self.retranslate()

    def retranslate(self) -> None:
        super().retranslate()
        checkbox = getattr(self, "output_fixed", None)
        if checkbox is None:
            return
        checkbox.setText(self._t("出力先を固定", "Lock output folder"))
        checkbox.setToolTip(
            self._t(
                "OFFでは入力フォルダを変更すると出力も自動で <入力>\\_censored に切り替わります。"
                "出力先を手入力または参照で選ぶと自動的に固定されます。",
                "When off, changing the input folder automatically changes the output to <input>\\_censored. "
                "Typing or browsing for an output folder automatically locks it.",
            )
        )
        self.overwrite.setToolTip(
            self._t(
                "ONなら同じ画像をもう一度実行したとき、出力・Review・手動確認用データを最新結果へ更新します。ログは履歴として別ファイルに残ります。",
                "When enabled, rerunning the same image refreshes output, Review, and manual-review data. Logs remain as separate history files.",
            )
        )
        learning_capture = getattr(self, "learning_capture", None)
        if learning_capture is not None:
            learning_capture.setText(self._t(
                "処理しながら誤検出候補を自動記憶",
                "Remember candidate evidence while processing",
            ))
            learning_capture.setToolTip(self._t(
                "元画像はコピーせず、候補cropと判定根拠だけをローカルSQLiteへ蓄積します。通常処理の失敗原因にはなりません。",
                "Stores only compact candidate crops and evidence in a local SQLite database. Original images are not copied, and learning failures never fail normal processing.",
            ))
            self.learning_button.setText(self._t(
                "過去画像フォルダを自動採掘…",
                "Mine legacy image folders…",
            ))
            self.learning_button.setToolTip(self._t(
                "PNG/JPEG/WebPとZIPが混在したフォルダを、破損・重複を飛ばしながらバックグラウンド解析します。",
                "Background-mine folders containing mixed PNG/JPEG/WebP/ZIP files while skipping corrupt and duplicate data.",
            ))

    def config(self) -> PipelineConfig:
        config = super().config()
        config.learning_enabled = bool(getattr(self, "learning_capture", None) and self.learning_capture.isChecked())
        return config

    def _output_is_fixed(self) -> bool:
        return bool(getattr(self, "_output_fixed_state", False))

    def _set_output_fixed_state(self, fixed: bool) -> None:
        fixed = bool(fixed)
        self._output_fixed_state = fixed
        checkbox = getattr(self, "output_fixed", None)
        if checkbox is None or checkbox.isChecked() == fixed:
            return
        checkbox.blockSignals(True)
        checkbox.setChecked(fixed)
        checkbox.blockSignals(False)

    def _update_output_default(self, text: str) -> None:
        if self._output_is_fixed():
            return
        target = default_output_for_input(text)
        if self.output_edit.text() != target:
            self.output_edit.setText(target)

    def _fix_output_after_manual_edit(self, _text: str) -> None:
        self._set_output_fixed_state(True)

    def _on_output_fixed_toggled(self, fixed: bool) -> None:
        self._output_fixed_state = bool(fixed)
        if not fixed:
            self._update_output_default(self.input_edit.text())

    def set_output_path(self, path: str) -> None:
        # Choosing a folder with the Browse dialog is an explicit custom output.
        self._set_output_fixed_state(True)
        super().set_output_path(path)

    def save_settings(self, settings: QSettings) -> None:
        super().save_settings(settings)
        settings.setValue("paths/output_fixed", self._output_is_fixed())
        settings.setValue(_OVERWRITE_DEFAULT_MIGRATION, True)
        if hasattr(self, "learning_capture"):
            settings.setValue("learning/capture_enabled", self.learning_capture.isChecked())
        settings.sync()

    def load_settings(self, settings: QSettings) -> None:
        super().load_settings(settings)
        raw_fixed = settings.value("paths/output_fixed", None)
        if raw_fixed is None:
            # Migration from v1.0.0 and older: preserve a previously customized
            # output folder, but keep ordinary <input>/_censored values following.
            fixed = output_differs_from_default(
                self.input_edit.text(),
                self.output_edit.text(),
            )
        else:
            fixed = _as_bool(raw_fixed)

        self._set_output_fixed_state(fixed)
        if not fixed:
            self._update_output_default(self.input_edit.text())

        # Previous versions wrote overwrite=False to settings automatically even
        # when the user never chose it. Migrate that inherited default exactly
        # once; subsequent explicit OFF choices are preserved.
        if not _as_bool(settings.value(_OVERWRITE_DEFAULT_MIGRATION, False)):
            self.overwrite.setChecked(True)
            settings.setValue("options/overwrite", True)
            settings.setValue(_OVERWRITE_DEFAULT_MIGRATION, True)

        if hasattr(self, "learning_capture"):
            self.learning_capture.setChecked(
                _as_bool(settings.value("learning/capture_enabled", _GUI_LEARNING_DEFAULT))
            )
        settings.sync()

    def reset_defaults(self, checked: bool = False, *, confirm: bool = True) -> bool:
        changed = super().reset_defaults(checked, confirm=confirm)
        if changed:
            self.overwrite.setChecked(True)
            if hasattr(self, "learning_capture"):
                self.learning_capture.setChecked(_GUI_LEARNING_DEFAULT)
            self.save_settings(self._settings_store)
        return changed

    def set_running(self, running: bool) -> None:
        super().set_running(running)
        checkbox = getattr(self, "output_fixed", None)
        if checkbox is not None:
            checkbox.setEnabled(not running)
        if hasattr(self, "learning_button"):
            self.learning_button.setEnabled(not running)
