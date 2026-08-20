from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from character_mosaic.ui.output_follow import OutputFollowControlPanel  # noqa: E402


def test_output_follow_panel_constructs_without_missing_widget(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)

    panel = OutputFollowControlPanel(settings=settings)

    assert panel.output_fixed is not None
    assert panel._output_is_fixed() is False
    panel.close()
    app.processEvents()


def test_output_lock_survives_settings_roundtrip(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    first = OutputFollowControlPanel(settings=settings)
    first.set_input_path(str(tmp_path / "input_a"))
    first.set_output_path(str(tmp_path / "shared_output"))
    first.save_settings(settings)
    first.close()

    restored_settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    second = OutputFollowControlPanel(settings=restored_settings)
    second.load_settings(restored_settings)

    assert second._output_is_fixed() is True
    assert second.output_edit.text() == str(tmp_path / "shared_output")
    second.close()
    app.processEvents()
