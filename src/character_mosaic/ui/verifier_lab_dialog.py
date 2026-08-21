from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..semantic_probe import candidate_context_crop
from ..types import Detection
from ..verifier_store import VerifierCandidate, VerifierStore


_LAB_CANDIDATE_LIMIT = 300


class VerifierLabDialog(QDialog):
    """Human labeling UI for candidate-level verifier ground truth."""

    def __init__(self, language: str = "ja", parent=None):
        super().__init__(parent)
        self._language = language
        self.store = VerifierStore()
        self.rows: list[VerifierCandidate] = []
        self.index = 0

        self.setWindowTitle(self._t("Verifier Lab", "Verifier Lab"))
        self.resize(900, 820)
        self.setMinimumSize(700, 620)

        layout = QVBoxLayout(self)

        intro = QLabel(self._t(
            "候補を見て『本物』『誤検出』『保留』を付けます。処理済みの _censored / review 系画像は"
            "自動除外し、1回の表示は最大300件です。ラベルは候補fingerprintに保存されます。",
            "Label each candidate as target, false positive, or uncertain. Derived _censored/review outputs are "
            "excluded automatically and each session shows at most 300 candidates. Labels are stored by fingerprint.",
        ))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        filters = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem(self._t("KEEP候補", "KEEP candidates"), "keep")
        self.filter_combo.addItem(self._t("SUPPRESS候補", "SUPPRESS candidates"), "suppress")
        self.filter_combo.addItem(self._t("全候補", "All candidates"), "all")
        self.unlabeled_only = QCheckBox(self._t("未ラベルだけ", "Unlabeled only"))
        self.unlabeled_only.setChecked(True)
        self.reload_button = QPushButton(self._t("再読込", "Reload"))
        self.reload_button.clicked.connect(self.reload)
        filters.addWidget(self.filter_combo)
        filters.addWidget(self.unlabeled_only)
        filters.addStretch(1)
        filters.addWidget(self.reload_button)
        layout.addLayout(filters)

        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(420)
        self.image_label.setStyleSheet("QLabel { background: #161616; border: 1px solid #444; }")
        layout.addWidget(self.image_label, 1)

        self.meta_label = QLabel()
        self.meta_label.setWordWrap(True)
        self.meta_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.meta_label)

        nav = QHBoxLayout()
        self.prev_button = QPushButton(self._t("← 前", "← Previous"))
        self.skip_button = QPushButton(self._t("スキップ →", "Skip →"))
        self.next_button = QPushButton(self._t("次 →", "Next →"))
        self.prev_button.clicked.connect(self.previous)
        self.skip_button.clicked.connect(self.next)
        self.next_button.clicked.connect(self.next)
        nav.addWidget(self.prev_button)
        nav.addStretch(1)
        nav.addWidget(self.skip_button)
        nav.addWidget(self.next_button)
        layout.addLayout(nav)

        labels = QHBoxLayout()
        self.positive_button = QPushButton(self._t("✓ 本物 (P)", "✓ Target (P)"))
        self.negative_button = QPushButton(self._t("✕ 誤検出 (N)", "✕ False positive (N)"))
        self.uncertain_button = QPushButton(self._t("? 保留 (U)", "? Uncertain (U)"))
        self.positive_button.clicked.connect(lambda: self.label_current("positive"))
        self.negative_button.clicked.connect(lambda: self.label_current("negative"))
        self.uncertain_button.clicked.connect(lambda: self.label_current("uncertain"))
        labels.addWidget(self.positive_button)
        labels.addWidget(self.negative_button)
        labels.addWidget(self.uncertain_button)
        layout.addLayout(labels)

        QShortcut(QKeySequence("P"), self, activated=lambda: self.label_current("positive"))
        QShortcut(QKeySequence("N"), self, activated=lambda: self.label_current("negative"))
        QShortcut(QKeySequence("U"), self, activated=lambda: self.label_current("uncertain"))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self.next)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self.previous)

        self.reload()

    def _t(self, ja: str, en: str) -> str:
        return ja if self._language == "ja" else en

    def reload(self) -> None:
        try:
            self.rows = self.store.candidates(
                decision=str(self.filter_combo.currentData()),
                only_unlabeled=self.unlabeled_only.isChecked(),
                limit=_LAB_CANDIDATE_LIMIT,
                exclude_derived=True,
            )
            self.index = 0
            self._show_current()
        except Exception as exc:
            QMessageBox.warning(self, self._t("読込エラー", "Load error"), str(exc))

    def label_current(self, label: str) -> None:
        row = self.current_row()
        if row is None:
            return
        try:
            self.store.set_label(row, label)
        except Exception as exc:
            QMessageBox.warning(self, self._t("保存エラー", "Save error"), str(exc))
            return

        if self.unlabeled_only.isChecked():
            self.rows.pop(self.index)
            if self.index >= len(self.rows):
                self.index = max(0, len(self.rows) - 1)
        else:
            self.index = min(self.index + 1, max(0, len(self.rows) - 1))
        self._show_current()

    def current_row(self) -> VerifierCandidate | None:
        if not self.rows or not (0 <= self.index < len(self.rows)):
            return None
        return self.rows[self.index]

    def previous(self) -> None:
        if self.rows:
            self.index = max(0, self.index - 1)
            self._show_current()

    def next(self) -> None:
        if self.rows:
            self.index = min(len(self.rows) - 1, self.index + 1)
            self._show_current()

    def _show_current(self) -> None:
        counts = self.store.stats(exclude_derived=True)
        self.stats_label.setText(self._t(
            f"クリーン教師ラベル: 本物 {counts['positive']:,} / 誤検出 {counts['negative']:,} / "
            f"保留 {counts['uncertain']:,} / 合計 {counts['total']:,}　|　表示候補 {len(self.rows):,}/{_LAB_CANDIDATE_LIMIT}",
            f"Clean ground truth: target {counts['positive']:,} / false positive {counts['negative']:,} / "
            f"uncertain {counts['uncertain']:,} / total {counts['total']:,} | visible {len(self.rows):,}/{_LAB_CANDIDATE_LIMIT}",
        ))

        row = self.current_row()
        enabled = row is not None
        for button in (
            self.prev_button, self.skip_button, self.next_button,
            self.positive_button, self.negative_button, self.uncertain_button,
        ):
            button.setEnabled(enabled)
        if row is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(self._t("候補がありません", "No candidates"))
            self.meta_label.setText("")
            return

        image = self._candidate_preview(row)
        if image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(self._t("画像を読めません", "Cannot load image"))
        else:
            qimage = ImageQt.ImageQt(image.convert("RGBA"))
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(
                max(100, self.image_label.width() - 12),
                max(100, self.image_label.height() - 12),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setText("")
            self.image_label.setPixmap(pixmap)

        pelvis = "-" if row.pelvis_distance is None else f"{row.pelvis_distance:.3f}"
        label_text = row.manual_label or "-"
        self.meta_label.setText(
            f"{self.index + 1} / {len(self.rows)}\n"
            f"source: {row.source_path}\n"
            f"box: {row.box}  detector={row.detector_score:.3f}  source={row.detector_source}\n"
            f"current={row.final_decision}  pseudo={row.pseudo_label}/{row.quality_tier}  "
            f"pelvis={pelvis}  manual={label_text}\n"
            f"suppression={row.suppression_reason or '-'}\n"
            f"positive={row.positive_signals}\nnegative={row.negative_signals}"
        )

    def _candidate_preview(self, row: VerifierCandidate) -> Image.Image | None:
        source = Path(row.source_path)
        try:
            if source.is_file():
                with Image.open(source) as opened:
                    original = opened.copy().convert("RGB")
                detection = Detection(row.box, "pussy", row.detector_score, row.detector_source)
                crop, crop_box = candidate_context_crop(
                    original,
                    detection,
                    scale=3.8,
                    min_side=320,
                )
                draw = ImageDraw.Draw(crop)
                offset_x, offset_y = crop_box[0], crop_box[1]
                local = (
                    row.box[0] - offset_x,
                    row.box[1] - offset_y,
                    row.box[2] - offset_x,
                    row.box[3] - offset_y,
                )
                line = max(2, round(max(crop.size) / 160))
                draw.rectangle(local, outline=(0, 255, 120), width=line)
                return crop
        except Exception:
            pass

        if row.crop_path:
            try:
                path = Path(row.crop_path)
                if path.is_file():
                    with Image.open(path) as opened:
                        return opened.copy().convert("RGB")
            except Exception:
                pass
        return None

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self.current_row() is not None:
            self._show_current()
