from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from character_mosaic.ui.preview_drop import (  # noqa: E402
    DropPreviewWidget,
    _first_supported_local_image,
)


def test_drop_helper_accepts_one_local_image(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-decoded-in-this-test")
    url = QUrl.fromLocalFile(str(image))

    assert _first_supported_local_image([url]) == image


def test_drop_helper_rejects_non_image_file(tmp_path):
    text = tmp_path / "sample.txt"
    text.write_text("x", encoding="utf-8")
    url = QUrl.fromLocalFile(str(text))

    assert _first_supported_local_image([url]) is None


def test_drop_preview_exposes_single_image_signal_and_hint():
    app = QApplication.instance() or QApplication([])
    preview = DropPreviewWidget("ja")

    assert preview.file_dropped is not None
    assert "ドラッグ" in preview.drop_hint.text()
    preview.set_language("en")
    assert "Drag one image" in preview.drop_hint.text()

    preview.close()
    app.processEvents()
