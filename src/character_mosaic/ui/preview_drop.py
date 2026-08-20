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
_BASE_DRAW_BODY_MAP = preview_module.ImageCanvas._draw_body_map


def _enhanced_region_style(kind):
    if "torso_back_zone" in kind or "torso_geometry_v2" in kind:
        return QColor(122, 108, 255, 225), True
    if "armpit_geometry_v2" in kind:
        return QColor(255, 150, 70, 225), True
    if "thigh_geometry_v2" in kind:
        return QColor(70, 210, 200, 220), True
    if "lower_leg_geometry_v2" in kind:
        return QColor(70, 180, 220, 220), True
    return _BASE_REGION_STYLE(kind)


def _enhanced_humanize_signal(signal, language):
    prefix, sep, rest = signal.partition(":")
    if language != "en":
        mapping = {
            "inside_torso_back": "胴体・背中BBox内",
            "review_without_pelvis": "骨盤根拠なしの誤検出",
            "inside_groin_zone": "股間候補領域内",
            "inside_upper_back": "背中・肩甲骨領域内",
            "inside_torso": "胴体領域内",
            "near_right_armpit_v2": "右わき領域内",
            "near_left_armpit_v2": "左わき領域内",
            "on_right_thigh": "右太もも上",
            "on_left_thigh": "左太もも上",
            "on_right_lower_leg": "右脚上",
            "on_left_lower_leg": "左脚上",
            "negative_memory": "過去の高信頼誤検出と近似一致",
        }
        if prefix in mapping:
            return f"{mapping[prefix]}:{rest}" if sep else mapping[prefix]
    return _BASE_HUMANIZE_SIGNAL(signal, language)


def _enhanced_draw_body_map(canvas, painter, left, top, sx, sy, scale):
    _BASE_DRAW_BODY_MAP(canvas, painter, left, top, sx, sy, scale)
    state = getattr(canvas, "_state", None)
    if state is None:
        return
    labels = {
        "torso_back_zone": "torso/back",
        "torso_geometry_v2": "torso v2",
        "right_armpit_geometry_v2": "R armpit",
        "left_armpit_geometry_v2": "L armpit",
        "right_thigh_geometry_v2": "R thigh",
        "left_thigh_geometry_v2": "L thigh",
        "right_lower_leg_geometry_v2": "R leg",
        "left_lower_leg_geometry_v2": "L leg",
    }
    for region in state.body_regions:
        label = labels.get(region.kind)
        if label is None:
            continue
        color, _dashed = _enhanced_region_style(region.kind)
        rect = canvas._map_box(region.box, left, top, sx, sy)
        person = f" p{region.person_index}" if region.person_index >= 0 else ""
        canvas._draw_label(painter, rect, f"{label}{person}", color, compact=True)


# The base canvas resolves these helpers at paint/evidence-render time. Patch
# only the small presentation hooks so the stable preview implementation stays
# shared with the normal folder workflow.
preview_module._region_style = _enhanced_region_style
preview_module._humanize_signal = _enhanced_humanize_signal
preview_module.ImageCanvas._draw_body_map = _enhanced_draw_body_map


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
                "画像1枚をここへドラッグ＆ドロップすると単体テストできます。人体解析では骨格に加えて胴体・背中・わき・太もも・脚の除外領域も確認できます。",
                "Drag one image here for a single-image test. Body analysis also shows torso/back, armpit, thigh, and leg exclusion regions derived from the pose.",
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
