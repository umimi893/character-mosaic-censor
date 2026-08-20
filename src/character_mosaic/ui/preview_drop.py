from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QWidget

from . import preview_widget as preview_module
from .preview_widget import PreviewWidget


_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_BASE_REGION_STYLE = preview_module._region_style
_BASE_HUMANIZE_SIGNAL = preview_module._humanize_signal


def _enhanced_region_style(kind):
    if "torso_back_zone" in kind:
        return QColor(122, 108, 255, 225), True
    return _BASE_REGION_STYLE(kind)


def _enhanced_humanize_signal(signal, language):
    prefix, sep, rest = signal.partition(":")
    if language != "en":
        mapping = {
            "inside_torso_back": "胴体・背中BBox内",
            "review_without_pelvis": "骨盤根拠なしの誤検出",
        }
        if prefix in mapping:
            return f"{mapping[prefix]}:{rest}" if sep else mapping[prefix]
    return _BASE_HUMANIZE_SIGNAL(signal, language)


# The base canvas resolves these helpers at paint/evidence-render time, so this
# keeps the visualization extension small while preserving the stable preview
# implementation.
preview_module._region_style = _enhanced_region_style
preview_module._humanize_signal = _enhanced_humanize_signal


class DropPreviewWidget(PreviewWidget):
    """Preview widget that accepts one Explorer image for immediate analysis."""

    file_dropped = Signal(str)

    def __init__(self, language: str = "ja", parent: QWidget | None = None):
        super().__init__(language, parent)
        self.setAcceptDrops(True)
        self.canvas.setAcceptDrops(False)

        self.drop_hint = QLabel()
        self.drop_hint.setObjectName("dropHint")
        self.drop_hint.setWordWrap(True)
        self.layout().insertWidget(1, self.drop_hint)
        self._set_drop_hint(False)

    def set_language(self, language):
        super().set_language(language)
        if hasattr(self, "drop_hint"):
            self._set_drop_hint(False)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        path = _first_supported_local_image(event.mimeData().urls())
        if path is None:
            event.ignore()
            return
        self._set_drop_hint(True)
        event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if _first_supported_local_image(event.mimeData().urls()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._set_drop_hint(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        path = _first_supported_local_image(event.mimeData().urls())
        self._set_drop_hint(False)
        if path is None:
            event.ignore()
            return
        self.file_dropped.emit(str(path))
        event.acceptProposedAction()

    def _set_drop_hint(self, active: bool) -> None:
        if active:
            self.drop_hint.setText(
                self._t(
                    "ここにドロップすると、この画像1枚だけ現在の設定で解析します",
                    "Drop here to analyze only this image with the current settings",
                )
            )
            return
        self.drop_hint.setText(
            self._t(
                "画像1枚をここへドラッグ＆ドロップすると、フォルダ一括処理をせず単体テストできます。人体解析: 緑=骨盤保護 / 紫=頭 / 桃=顔 / 黄=目 / 青紫=胴体・背中",
                "Drag one image here for a single-image test without running the whole folder. Body analysis: green=pelvis / purple=head / pink=face / yellow=eyes / violet=torso-back",
            )
        )


def _first_supported_local_image(urls) -> Path | None:
    for url in urls:
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile()).expanduser()
        if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES:
            return path
    return None
