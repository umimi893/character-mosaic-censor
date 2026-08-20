from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..types import Detection, PreviewFrame


@dataclass
class _CanvasState:
    image: QImage
    detections: tuple[Detection, ...]
    censor_boxes: tuple[tuple[int, int, int, int], ...]
    status: str
    coordinate_size: tuple[int, int]


class ImageCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._state: _CanvasState | None = None
        self._show_boxes = True
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_state(self, state: _CanvasState | None, show_boxes: bool) -> None:
        self._state = state
        self._show_boxes = show_boxes
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#090c11"))
        if self._state is None or self._state.image.isNull():
            painter.setPen(QColor("#73808f"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "処理を開始するとここに画像が表示されます")
            return

        image = self._state.image
        coord_w, coord_h = self._state.coordinate_size
        avail = self.rect().adjusted(12, 12, -12, -12)
        scale = min(avail.width() / image.width(), avail.height() / image.height())
        draw_w = image.width() * scale
        draw_h = image.height() * scale
        box_scale_x = draw_w / max(1, coord_w)
        box_scale_y = draw_h / max(1, coord_h)
        left = avail.left() + (avail.width() - draw_w) / 2
        top = avail.top() + (avail.height() - draw_h) / 2
        target = QRectF(left, top, draw_w, draw_h)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, image)

        if not self._show_boxes:
            return

        # Expanded censor area first, then the model's raw detector box.
        censor_pen = QPen(QColor(255, 171, 46, 215))
        censor_pen.setWidthF(max(1.5, 2.0 * min(1.5, scale)))
        censor_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(censor_pen)
        for box in self._state.censor_boxes:
            painter.drawRect(self._map_box(box, left, top, box_scale_x, box_scale_y))

        det_pen = QPen(QColor(58, 224, 139, 245))
        det_pen.setWidthF(max(2.0, 2.5 * min(1.5, scale)))
        det_pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(det_pen)
        for det in self._state.detections:
            rect = self._map_box(det.box, left, top, box_scale_x, box_scale_y)
            painter.drawRect(rect)
            self._draw_label(painter, rect, f"{det.label} {det.score:.2f}")

    @staticmethod
    def _map_box(
        box: tuple[int, int, int, int], left: float, top: float, scale_x: float, scale_y: float
    ) -> QRectF:
        x0, y0, x1, y1 = box
        return QRectF(
            left + x0 * scale_x,
            top + y0 * scale_y,
            max(1.0, (x1 - x0) * scale_x),
            max(1.0, (y1 - y0) * scale_y),
        )

    @staticmethod
    def _draw_label(painter: QPainter, rect: QRectF, text: str) -> None:
        metrics = QFontMetrics(painter.font())
        text_rect = metrics.boundingRect(text)
        width = text_rect.width() + 12
        height = text_rect.height() + 8
        x = rect.left()
        y = max(0.0, rect.top() - height)
        label_rect = QRectF(x, y, width, height)
        painter.fillRect(label_rect, QColor(7, 14, 18, 220))
        painter.setPen(QColor("#f4fff9"))
        painter.drawText(QPointF(x + 6, y + height - 5), text)


class PreviewWidget(QWidget):
    """Large monitor showing original, detection and censored stages."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._states: dict[str, _CanvasState | None] = {
            "original": None,
            "detected": None,
            "censored": None,
        }
        self._current_mode = "original"

        self.canvas = ImageCanvas(self)
        self.file_label = QLabel("待機中")
        self.file_label.setObjectName("previewFile")
        self.status_label = QLabel("画像を選択して実行してください")
        self.status_label.setObjectName("previewStatus")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.original_btn = self._mode_button("元画像", "original")
        self.detected_btn = self._mode_button("検出BBox", "detected")
        self.censored_btn = self._mode_button("モザイク後", "censored")
        self.original_btn.setChecked(True)
        self.box_toggle = QCheckBox("BBox表示")
        self.box_toggle.setChecked(True)
        self.box_toggle.toggled.connect(self._refresh)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for button in (self.original_btn, self.detected_btn, self.censored_btn):
            self.mode_group.addButton(button)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.original_btn)
        top.addWidget(self.detected_btn)
        top.addWidget(self.censored_btn)
        top.addSpacing(8)
        top.addWidget(self.box_toggle)
        top.addStretch(1)

        info = QHBoxLayout()
        info.setContentsMargins(2, 0, 2, 0)
        info.addWidget(self.file_label, 1)
        info.addWidget(self.status_label, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(10)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(info)

    def _mode_button(self, text: str, mode: str) -> QPushButton:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setProperty("previewMode", True)
        button.clicked.connect(lambda checked=False, m=mode: self._set_mode(m))
        return button

    def set_frame(self, frame: PreviewFrame) -> None:
        qimage = _pil_to_qimage(frame.image)
        state = _CanvasState(
            qimage,
            frame.detections,
            frame.censor_boxes,
            frame.status,
            frame.coordinate_size or frame.image.size,
        )
        self.file_label.setText(frame.source.name)
        self.file_label.setToolTip(str(frame.source))

        if frame.stage == "original":
            # New source image: do not keep the prior file's processed states.
            self._states = {"original": state, "detected": None, "censored": None}
            self._activate_mode("original")
        elif frame.stage in {"detecting", "detected"}:
            self._states["detected"] = state
            self._activate_mode("detected")
        elif frame.stage == "censored":
            self._states["censored"] = state
            self._activate_mode("censored")
        else:
            self._states["detected"] = state

        self.status_label.setText(frame.status or frame.stage)
        self._refresh()

    def clear(self) -> None:
        self._states = {"original": None, "detected": None, "censored": None}
        self.file_label.setText("待機中")
        self.status_label.setText("画像を選択して実行してください")
        self._activate_mode("original")
        self._refresh()

    def _activate_mode(self, mode: str) -> None:
        self._current_mode = mode
        buttons = {
            "original": self.original_btn,
            "detected": self.detected_btn,
            "censored": self.censored_btn,
        }
        buttons[mode].setChecked(True)

    def _set_mode(self, mode: str) -> None:
        self._current_mode = mode
        self._refresh()

    def _refresh(self) -> None:
        state = self._states.get(self._current_mode)
        if state is None:
            # Graceful fallback when a requested stage has not happened yet.
            for fallback in ("censored", "detected", "original"):
                candidate = self._states.get(fallback)
                if candidate is not None:
                    state = candidate
                    break
        self.canvas.set_state(state, self.box_toggle.isChecked())


def _pil_to_qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    # Detach from the temporary bytes object before returning across event turns.
    return qimage.copy()
