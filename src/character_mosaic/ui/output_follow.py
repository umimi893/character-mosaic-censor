from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QWidget

from ..output_paths import default_output_for_input, output_differs_from_default
from .settings_safety import EnhancedControlPanel


_OVERWRITE_DEFAULT_MIGRATION = "migrations/overwrite_default_v130"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OutputFollowControlPanel(EnhancedControlPanel):
    """Control panel whose default output follows the selected input folder.

    The output path follows ``<input>/_censored`` unless the user explicitly
    fixes it. Manually editing or browsing for an output folder automatically
    enables the fixed state.

    GUI reruns default to overwriting existing generated results.  A one-time
    v1.3 settings migration turns this on for installations that previously
    persisted the old false default; after that, an explicit user choice is
    preserved normally.
    """

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
        # when the user never chose it.  Migrate that inherited default exactly
        # once; subsequent explicit OFF choices are preserved.
        if not _as_bool(settings.value(_OVERWRITE_DEFAULT_MIGRATION, False)):
            self.overwrite.setChecked(True)
            settings.setValue("options/overwrite", True)
            settings.setValue(_OVERWRITE_DEFAULT_MIGRATION, True)
            settings.sync()

    def reset_defaults(self, checked: bool = False, *, confirm: bool = True) -> bool:
        changed = super().reset_defaults(checked, confirm=confirm)
        if changed:
            self.overwrite.setChecked(True)
            self.save_settings(self._settings_store)
        return changed

    def set_running(self, running: bool) -> None:
        super().set_running(running)
        checkbox = getattr(self, "output_fixed", None)
        if checkbox is not None:
            checkbox.setEnabled(not running)
