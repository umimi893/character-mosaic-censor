from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from ..i18n import normalize_language, t
from ..types import BodyRegion, CandidateEvidence, Detection, PoseEdge, PosePoint, PreviewFrame


@dataclass
class _CanvasState:
    image: QImage
    detections: tuple[Detection, ...]
    censor_boxes: tuple[tuple[int, int, int, int], ...]
    status: str
    coordinate_size: tuple[int, int]
    body_regions: tuple[BodyRegion, ...] = tuple()
    pose_points: tuple[PosePoint, ...] = tuple()
    pose_edges: tuple[PoseEdge, ...] = tuple()
    candidate_evidence: tuple[CandidateEvidence, ...] = tuple()
    analysis_status: str = ""


class ImageCanvas(QWidget):
    candidate_selected = Signal(int)

    def __init__(self, language: str = "ja", parent: QWidget | None = None):
        super().__init__(parent)
        self._language = normalize_language(language)
        self._state: _CanvasState | None = None
        self._show_boxes = True
        self._show_body_map = False
        self._candidate_rects: list[tuple[int, QRectF]] = []
        self._selected_candidate = -1
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_state(self, state: _CanvasState | None, show_boxes: bool, show_body_map: bool = False) -> None:
        self._state = state; self._show_boxes = show_boxes; self._show_body_map = show_body_map
        if state is None or self._selected_candidate >= len(state.candidate_evidence):
            self._selected_candidate = -1
        self.update()

    def set_selected_candidate(self, index: int) -> None:
        self._selected_candidate = index; self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("#090c11")); self._candidate_rects = []
        if self._state is None or self._state.image.isNull():
            painter.setPen(QColor("#73808f")); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                t(self._language, "処理を開始するとここに画像が表示されます", "Images appear here after processing starts")); return
        image = self._state.image; coord_w, coord_h = self._state.coordinate_size
        avail = self.rect().adjusted(12, 12, -12, -12)
        scale = min(avail.width() / image.width(), avail.height() / image.height())
        draw_w, draw_h = image.width() * scale, image.height() * scale
        sx, sy = draw_w / max(1, coord_w), draw_h / max(1, coord_h)
        left = avail.left() + (avail.width() - draw_w) / 2; top = avail.top() + (avail.height() - draw_h) / 2
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(left, top, draw_w, draw_h), image)
        if self._show_body_map:
            self._draw_body_map(painter, left, top, sx, sy, scale); self._draw_candidate_evidence(painter, left, top, sx, sy, scale); return
        if not self._show_boxes: return
        pen = QPen(QColor(255, 171, 46, 215)); pen.setWidthF(max(1.5, 2.0 * min(1.5, scale))); pen.setStyle(Qt.PenStyle.DashLine); painter.setPen(pen)
        for box in self._state.censor_boxes: painter.drawEllipse(self._map_box(box, left, top, sx, sy))
        pen = QPen(QColor(58, 224, 139, 245)); pen.setWidthF(max(2.0, 2.5 * min(1.5, scale))); painter.setPen(pen)
        for det in self._state.detections:
            rect = self._map_box(det.box, left, top, sx, sy); painter.drawRect(rect); self._draw_label(painter, rect, f"{det.label} {det.score:.2f}", QColor("#3ae08b"))

    def _draw_body_map(self, painter, left, top, sx, sy, scale) -> None:
        if self._state is None: return
        pen = QPen(QColor(65, 190, 230, 180)); pen.setWidthF(max(1.2, 1.8 * min(1.5, scale))); painter.setPen(pen)
        for edge in self._state.pose_edges:
            painter.drawLine(QPointF(left + edge.start[0] * sx, top + edge.start[1] * sy), QPointF(left + edge.end[0] * sx, top + edge.end[1] * sy))
        for region in self._state.body_regions:
            color, dashed = _region_style(region.kind); pen = QPen(color); pen.setWidthF(max(1.2, 1.8 * min(1.5, scale)))
            if dashed: pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen); rect = self._map_box(region.box, left, top, sx, sy); painter.drawRect(rect)
            if region.kind in {"person", "head", "face", "eye", "pelvis_safe"}:
                person = f" p{region.person_index}" if region.person_index >= 0 else ""
                self._draw_label(painter, rect, f"{region.kind}{person} {region.score:.2f}", color, compact=True)
        painter.setPen(QPen(QColor(121, 213, 255, 230))); painter.setBrush(QColor(121, 213, 255, 180))
        for point in self._state.pose_points:
            x, y = left + point.x * sx, top + point.y * sy
            radius = 4.0 if point.label in {"right_hip", "left_hip", "right_knee", "left_knee"} else 2.5
            painter.drawEllipse(QPointF(x, y), radius, radius)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_candidate_evidence(self, painter, left, top, sx, sy, scale) -> None:
        if self._state is None: return
        for index, evidence in enumerate(self._state.candidate_evidence):
            color = _decision_color(evidence.decision); pen = QPen(color)
            pen.setWidthF(max(2.0, (3.5 if index == self._selected_candidate else 2.4) * min(1.5, scale)))
            if evidence.decision == "suppress": pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen); rect = self._map_box(evidence.detection.box, left, top, sx, sy); painter.drawRect(rect)
            self._candidate_rects.append((index, rect)); self._draw_label(painter, rect, f"#{index + 1} {evidence.decision.upper()} {evidence.detection.score:.2f}", color)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        point = event.position(); hit = -1
        for index, rect in reversed(self._candidate_rects):
            if rect.adjusted(-5, -5, 5, 5).contains(point): hit = index; break
        if hit >= 0:
            self._selected_candidate = hit; self.candidate_selected.emit(hit); self.update()
        super().mousePressEvent(event)

    def set_language(self, language: str) -> None: self._language = normalize_language(language); self.update()

    @staticmethod
    def _map_box(box, left, top, sx, sy) -> QRectF:
        x0, y0, x1, y1 = box
        return QRectF(left + x0 * sx, top + y0 * sy, max(1.0, (x1 - x0) * sx), max(1.0, (y1 - y0) * sy))

    @staticmethod
    def _draw_label(painter, rect, text, accent, compact=False) -> None:
        metrics = QFontMetrics(painter.font()); text_rect = metrics.boundingRect(text); px = 5 if compact else 6; py = 3 if compact else 4
        width, height = text_rect.width() + px * 2, text_rect.height() + py * 2; x, y = rect.left(), max(0.0, rect.top() - height)
        painter.fillRect(QRectF(x, y, width, height), QColor(7, 14, 18, 225)); painter.setPen(accent); painter.drawText(QPointF(x + px, y + height - py - 1), text)


class PreviewWidget(QWidget):
    """Monitor with original/detection/body-analysis/censored views."""

    def __init__(self, language: str = "ja", parent: QWidget | None = None):
        super().__init__(parent); self._language = normalize_language(language)
        self._states = {"original": None, "detected": None, "analysis": None, "censored": None}; self._current_mode = "original"; self._selected_candidate = -1
        self.canvas = ImageCanvas(self._language, self); self.canvas.candidate_selected.connect(self._on_candidate_selected)
        self.file_label = QLabel(); self.file_label.setObjectName("previewFile")
        self.status_label = QLabel(); self.status_label.setObjectName("previewStatus"); self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.evidence_label = QLabel(); self.evidence_label.setObjectName("analysisEvidence"); self.evidence_label.setWordWrap(True); self.evidence_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); self.evidence_label.setMinimumHeight(42)
        self.original_btn = self._mode_button("", "original"); self.detected_btn = self._mode_button("", "detected"); self.analysis_btn = self._mode_button("", "analysis"); self.censored_btn = self._mode_button("", "censored"); self.original_btn.setChecked(True)
        self.box_toggle = QCheckBox(); self.box_toggle.setChecked(True); self.box_toggle.toggled.connect(self._refresh)
        self.mode_group = QButtonGroup(self); self.mode_group.setExclusive(True)
        for button in (self.original_btn, self.detected_btn, self.analysis_btn, self.censored_btn): self.mode_group.addButton(button)
        top = QHBoxLayout(); top.setContentsMargins(0,0,0,0)
        for button in (self.original_btn, self.detected_btn, self.analysis_btn, self.censored_btn): top.addWidget(button)
        top.addSpacing(8); top.addWidget(self.box_toggle); top.addStretch(1)
        info = QHBoxLayout(); info.setContentsMargins(2,0,2,0); info.addWidget(self.file_label,1); info.addWidget(self.status_label,2)
        layout = QVBoxLayout(self); layout.setContentsMargins(12,12,12,10); layout.setSpacing(10); layout.addLayout(top); layout.addWidget(self.canvas,1); layout.addWidget(self.evidence_label); layout.addLayout(info)
        self.set_language(self._language)

    def _t(self, ja, en): return t(self._language, ja, en)
    def set_language(self, language):
        self._language = normalize_language(language); self.canvas.set_language(self._language)
        self.original_btn.setText(self._t("元画像","Original")); self.detected_btn.setText(self._t("検出範囲","Detections")); self.analysis_btn.setText(self._t("人体解析","Body analysis")); self.censored_btn.setText(self._t("処理後","Censored")); self.box_toggle.setText(self._t("検出範囲を表示","Show detection areas"))
        if not any(self._states.values()):
            self.file_label.setText(self._t("待機中","Ready")); self.status_label.setText(self._t("入力フォルダを選んで実行してください","Choose an input folder and start processing")); self.evidence_label.setText(self._t("人体解析結果はここに表示されます","Body-analysis evidence appears here"))

    def _mode_button(self, text, mode):
        button = QPushButton(text); button.setCheckable(True); button.setProperty("previewMode", True); button.clicked.connect(lambda checked=False, m=mode: self._set_mode(m)); return button

    def set_frame(self, frame: PreviewFrame) -> None:
        state = _CanvasState(_pil_to_qimage(frame.image), frame.detections, frame.censor_boxes, frame.status, frame.coordinate_size or frame.image.size, frame.body_regions, frame.pose_points, frame.pose_edges, frame.candidate_evidence, frame.analysis_status)
        self.file_label.setText(frame.source.name); self.file_label.setToolTip(str(frame.source))
        if frame.stage == "original": self._states = {"original":state,"detected":None,"analysis":None,"censored":None}; self._selected_candidate=-1; self._activate_mode("original")
        elif frame.stage == "analysis": self._states["analysis"] = state; self._selected_candidate=-1; self._activate_mode("analysis")
        elif frame.stage in {"detecting","detected"}:
            self._states["detected"] = state
            if frame.stage == "detected" and self._states["analysis"] is None and frame.candidate_evidence: self._states["analysis"] = state
            self._activate_mode("detected")
        elif frame.stage == "censored": self._states["censored"] = state; self._activate_mode("censored")
        else: self._states["detected"] = state
        self.status_label.setText(frame.status or frame.stage); self._refresh_evidence(); self._refresh()

    def clear(self):
        self._states={"original":None,"detected":None,"analysis":None,"censored":None}; self._selected_candidate=-1
        self.file_label.setText(self._t("待機中","Ready")); self.status_label.setText(self._t("入力フォルダを選んで実行してください","Choose an input folder and start processing")); self.evidence_label.setText(self._t("人体解析結果はここに表示されます","Body-analysis evidence appears here")); self._activate_mode("original"); self._refresh()

    def _activate_mode(self, mode):
        self._current_mode=mode; {"original":self.original_btn,"detected":self.detected_btn,"analysis":self.analysis_btn,"censored":self.censored_btn}[mode].setChecked(True)
    def _set_mode(self, mode): self._current_mode=mode; self._refresh_evidence(); self._refresh()
    def _on_candidate_selected(self, index): self._selected_candidate=index; self.canvas.set_selected_candidate(index); self._refresh_evidence()

    def _refresh(self):
        state=self._states.get(self._current_mode)
        if state is None:
            for fallback in ("analysis","censored","detected","original"):
                if self._states.get(fallback) is not None: state=self._states[fallback]; break
        self.canvas.set_selected_candidate(self._selected_candidate); self.canvas.set_state(state, self.box_toggle.isChecked(), show_body_map=self._current_mode=="analysis")

    def _refresh_evidence(self):
        state=self._states.get("analysis")
        if state is None or not state.candidate_evidence:
            self.evidence_label.setText(self._t(f"人体解析: {state.analysis_status}", f"Body analysis: {state.analysis_status}") if state is not None and state.analysis_status else self._t("人体解析データなし","No body-analysis data")); return
        if self._selected_candidate < 0 or self._selected_candidate >= len(state.candidate_evidence):
            keep=sum(e.decision=="keep" for e in state.candidate_evidence); review=sum(e.decision=="review" for e in state.candidate_evidence); suppress=sum(e.decision=="suppress" for e in state.candidate_evidence)
            self.evidence_label.setText(self._t(f"候補をクリックすると判定理由を表示します。KEEP {keep} / REVIEW {review} / SUPPRESS {suppress} / 状態 {state.analysis_status}", f"Click a candidate to inspect evidence. KEEP {keep} / REVIEW {review} / SUPPRESS {suppress} / status {state.analysis_status}")); return
        evidence=state.candidate_evidence[self._selected_candidate]; pos=" / ".join(_humanize_signal(s,self._language) for s in evidence.positive_signals) or "-"; neg=" / ".join(_humanize_signal(s,self._language) for s in evidence.negative_signals) or "-"
        self.evidence_label.setText(self._t(f"候補 #{self._selected_candidate+1}  判定: {evidence.decision.upper()}  |  保持材料: {pos}  |  除外材料: {neg}", f"Candidate #{self._selected_candidate+1}  decision: {evidence.decision.upper()}  |  keep evidence: {pos}  |  negative evidence: {neg}"))


def _decision_color(decision):
    if decision=="suppress": return QColor(244,92,92,245)
    if decision=="review": return QColor(255,184,77,245)
    return QColor(58,224,139,245)

def _region_style(kind):
    if kind=="person": return QColor(93,165,255,190),True
    if kind=="head": return QColor(190,120,255,210),False
    if kind=="face": return QColor(255,120,205,210),False
    if kind=="eye": return QColor(255,230,90,230),False
    if kind=="pelvis_safe": return QColor(71,232,155,220),True
    if "knee_zone" in kind: return QColor(255,126,76,215),True
    if "armpit_zone" in kind: return QColor(255,156,86,215),True
    return QColor(120,190,225,180),True

def _humanize_signal(signal, language):
    if language=="en": return signal
    mapping={"detector":"検出信頼度","near_pelvis":"骨盤付近","inside_head":"頭BBox内","inside_face":"顔BBox内","inside_eye":"目BBox内","near_right_knee":"右膝付近","near_left_knee":"左膝付近","near_right_armpit":"右腋付近","near_left_armpit":"左腋付近"}
    prefix,sep,rest=signal.partition(":"); label=mapping.get(prefix,prefix); return f"{label}:{rest}" if sep else label

def _pil_to_qimage(image: Image.Image) -> QImage:
    rgba=image.convert("RGBA"); data=rgba.tobytes("raw","RGBA"); return QImage(data,rgba.width,rgba.height,rgba.width*4,QImage.Format.Format_RGBA8888).copy()
