import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QTabWidget

from character_mosaic.pipeline import PipelineConfig
from character_mosaic.ui.control_panel import ControlPanel
from character_mosaic.ui.preview_widget import PreviewWidget
from character_mosaic.ui.settings_dialog import SettingsDialog


def _app():
    return QApplication.instance() or QApplication([])


def test_control_panel_switches_between_japanese_and_english():
    _app()
    panel = ControlPanel("ja")
    assert panel.advanced_button.text() == "詳細設定…"
    assert panel.detect_group.title() == "検出設定"

    panel.set_language("en")

    assert panel.advanced_button.text() == "Advanced settings…"
    assert panel.detect_group.title() == "Detection"
    assert panel.config().language == "en"


def test_language_setting_is_persisted(tmp_path):
    _app()
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    first = ControlPanel("ja")
    first.set_language("en")
    first.save_settings(settings)

    second = ControlPanel("ja")
    second.load_settings(settings)

    assert second.language() == "en"
    assert second.language_combo.currentData() == "en"


def test_advanced_settings_have_clear_language_specific_tabs():
    _app()
    english = SettingsDialog(PipelineConfig(language="en"), "en")
    tabs = english.findChild(QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Basic", "Large images", "Duplicate merging"]

    japanese = SettingsDialog(PipelineConfig(language="ja"), "ja")
    tabs = japanese.findChild(QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["基本", "大きな画像", "重複検出の調整"]


def test_preview_labels_switch_language():
    _app()
    preview = PreviewWidget("ja")
    assert preview.original_btn.text() == "元画像"
    assert preview.analysis_btn.text() == "人体解析"

    preview.set_language("en")

    assert preview.original_btn.text() == "Original"
    assert preview.analysis_btn.text() == "Body analysis"
    assert preview.box_toggle.text() == "Show detection areas"
