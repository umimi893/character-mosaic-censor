import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QSettings, Qt
from PySide6.QtWidgets import QApplication

from character_mosaic.pipeline import PipelineConfig
from character_mosaic.ui.control_panel import ControlPanel
from character_mosaic.ui.settings_dialog import SettingsDialog
from character_mosaic.ui.settings_safety import EnhancedControlPanel, _install_wheel_guard


def _app():
    return QApplication.instance() or QApplication([])


class _FakeWheelEvent:
    def __init__(self, angle_y: int):
        self._angle_y = angle_y
        self.accepted = False

    def type(self):
        return QEvent.Type.Wheel

    def pixelDelta(self):
        return QPoint(0, 0)

    def angleDelta(self):
        return QPoint(0, self._angle_y)

    def accept(self):
        self.accepted = True


def test_wheel_guard_keeps_numeric_value_and_scrolls_panel_without_focusing():
    _app()
    panel = EnhancedControlPanel("ja")
    panel.verticalScrollBar().setRange(0, 1000)
    panel.verticalScrollBar().setValue(500)
    panel.confidence.clearFocus()
    before_value = panel.confidence.value()

    event = _FakeWheelEvent(-120)
    consumed = panel._wheel_guard.eventFilter(panel.confidence, event)

    assert consumed is True
    assert event.accepted is True
    assert panel.confidence.value() == before_value
    assert panel.confidence.hasFocus() is False
    assert panel.verticalScrollBar().value() > 500


def test_value_controls_accept_click_or_tab_focus_not_wheel_focus():
    _app()
    panel = EnhancedControlPanel("ja")

    assert panel.confidence.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert panel.person_count.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert panel.mode.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert panel.language_combo.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_combo_popup_wheel_is_consumed_so_highlight_does_not_move():
    _app()
    panel = EnhancedControlPanel("ja")
    event = _FakeWheelEvent(-120)

    consumed = panel._wheel_guard.eventFilter(panel.mode.view().viewport(), event)

    assert consumed is True
    assert event.accepted is True


def test_advanced_settings_use_same_click_or_tab_focus_policy():
    _app()
    dialog = SettingsDialog(PipelineConfig(), "ja")
    guard = _install_wheel_guard(dialog)
    dialog._test_guard = guard

    assert dialog.tile_trigger.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert dialog.model_iou.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert dialog.model_level.focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_reset_defaults_preserves_paths_and_language_and_persists(tmp_path):
    _app()
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    panel = EnhancedControlPanel("en", settings=settings)
    panel.set_input_path("C:/images")
    panel.set_output_path("C:/out")
    panel.set_review_path("C:/review")
    panel.confidence.setValue(0.55)
    panel.person_count.setValue(4)
    panel.strength.setValue(64)
    panel.tile_detection.setChecked(False)

    assert panel.reset_defaults(confirm=False) is True

    defaults = PipelineConfig(language="en")
    assert panel.language() == "en"
    assert panel.input_edit.text() == "C:/images"
    assert panel.output_edit.text() == "C:/out"
    assert panel.review_edit.text() == "C:/review"
    assert panel.confidence.value() == defaults.detection_threshold
    assert panel.person_count.value() == defaults.expected_person_count
    assert panel.strength.value() == defaults.block_size
    assert panel.tile_detection.isChecked() == defaults.tile_large_images

    restored = ControlPanel("ja")
    restored.load_settings(settings)
    assert restored.language() == "en"
    assert restored.input_edit.text() == "C:/images"
    assert restored.output_edit.text() == "C:/out"
    assert restored.confidence.value() == defaults.detection_threshold


def test_manual_save_button_and_keyboard_guidance_are_localized(tmp_path):
    _app()
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    panel = EnhancedControlPanel("ja", settings=settings)
    panel.confidence.setValue(0.44)

    panel._save_now()

    assert float(settings.value("detect/confidence")) == 0.44
    assert panel.save_settings_button.text() == "設定を保存"
    assert panel.reset_settings_button.text() == "初期設定に戻す"
    assert "左クリック" in panel.settings_help.text()
    assert "↑↓" in panel.settings_help.text()

    panel.set_language("en")
    assert panel.save_settings_button.text() == "Save settings"
    assert panel.reset_settings_button.text() == "Reset defaults"
