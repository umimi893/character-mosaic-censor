from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QCheckBox, QWidget

from ..output_paths import default_output_for_input
from .settings_safety import EnhancedControlPanel


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OutputFollowControlPanel(EnhancedControlPanel):
    """Control panel whose default output follows the selected input folder.

    The output path follows ``<input>/_censored`` unless the user explicitly
    fixes it. Manually editing or browsing for an output folder automatically
    enables the fixed state.
    """

    def __init__(
        self,
        language: str = "ja",
        parent: QWidget | None = None,
        settings: QSettings | None = None,
    ):
        super().__init__(language, parent, settings)

        self.output_fixed = QCheckBox()
        io_layout = self.io_group.layout()
        # Input row, output row, [lock], review row.
        io_layout.insertWidget(2, self.output_fixed)

        self.output_edit.textEdited.connect(self._fix_output_after_manual_edit)
        self.output_fixed.toggled.connect(self._on_output_fixed_toggled)
        self.retranslate()

    def retranslate(self) -> None:
        super().retranslate()
        if not hasattr(self, "output_fixed"):
            return
        self.output_fixed.setText(self._t("出力先を固定", "Lock output folder"))
        self.output_fixed.setToolTip(
            self._t(
                "OFFでは入力フォルダを変更すると出力も自動で <入力>\\_censored に切り替わります。"
                "出力先を手入力または参照で選ぶと自動的に固定されます。",
                "When off, changing the input folder automatically changes the output to <input>\\_censored. "
                "Typing or browsing for an output folder automatically locks it.",
            )
        )

    def _output_is_fixed(self) -> bool:
        checkbox = getattr(self, "output_fixed", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _update_output_default(self, text: str) -> None:
        if self._output_is_fixed():
            return
        target = default_output_for_input(text)
        if self.output_edit.text() != target:
            self.output_edit.setText(target)

    def _fix_output_after_manual_edit(self, _text: str) -> None:
        if not self.output_fixed.isChecked():
            self.output_fixed.setChecked(True)

    def _on_output_fixed_toggled(self, fixed: bool) -> None:
        if not fixed:
            self._update_output_default(self.input_edit.text())

    def set_output_path(self, path: str) -> None:
        # Choosing a folder with the Browse dialog is an explicit custom output.
        self.output_fixed.setChecked(True)
        super().set_output_path(path)

    def save_settings(self, settings: QSettings) -> None:
        super().save_settings(settings)
        settings.setValue("paths/output_fixed", self.output_fixed.isChecked())
        settings.sync()

    def load_settings(self, settings: QSettings) -> None:
        super().load_settings(settings)
        fixed = _as_bool(settings.value("paths/output_fixed", False))
        self.output_fixed.blockSignals(True)
        self.output_fixed.setChecked(fixed)
        self.output_fixed.blockSignals(False)
        if not fixed:
            self._update_output_default(self.input_edit.text())

    def set_running(self, running: bool) -> None:
        super().set_running(running)
        self.output_fixed.setEnabled(not running)
