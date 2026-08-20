from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..i18n import normalize_language, t
from ..pipeline import PipelineConfig


class SettingsDialog(QDialog):
    """Advanced settings grouped by user intent with inline guidance."""

    def __init__(self, config: PipelineConfig, language: str = "ja", parent: QWidget | None = None):
        super().__init__(parent)
        self.language = normalize_language(language)
        self.setWindowTitle(self._t("詳細設定", "Advanced settings"))
        self.setModal(True)
        self.setMinimumWidth(620)
        self.resize(680, 620)

        layout = QVBoxLayout(self)
        note = QLabel(
            self._t(
                "通常は「基本」タブだけ確認すれば十分です。RTX 5090では精度優先モデルと大画像の分割検出を推奨します。",
                "For normal use, the Basic tab is enough. On an RTX 5090, use the accuracy model and keep large-image tiling enabled.",
            )
        )
        note.setWordWrap(True)
        note.setObjectName("dialogNote")
        layout.addWidget(note)

        tabs = QTabWidget()
        tabs.addTab(self._build_basic_tab(), self._t("基本", "Basic"))
        tabs.addTab(self._build_tile_tab(), self._t("大きな画像", "Large images"))
        tabs.addTab(self._build_merge_tab(), self._t("重複検出の調整", "Duplicate merging"))
        layout.addWidget(tabs, 1)

        reset = QPushButton(self._t("すべて既定値に戻す", "Reset all to defaults"))
        reset.clicked.connect(self._reset_defaults)
        layout.addWidget(reset)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self._t("保存", "Save"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self._t("キャンセル", "Cancel"))
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._load(config)

    def _t(self, ja: str, en: str) -> str:
        return t(self.language, ja, en)

    def _build_basic_tab(self) -> QWidget:
        tab = QWidget()
        form = _form(tab)
        self.model_level = QComboBox()
        self.model_level.addItem(self._t("精度優先 Standard (s)", "Accuracy first: Standard (s)"), "s")
        self.model_level.addItem(self._t("速度優先 Nano (n)", "Speed first: Nano (n)"), "n")
        form.addRow(
            self._t("検出モデル", "Detection model"),
            _with_help(self.model_level, self._t(
                "Standardは見逃しを減らしたい場合の推奨値です。Nanoは速い代わりに精度が下がります。",
                "Standard is recommended for recall. Nano is faster but can miss more targets.",
            )),
        )
        self.review_no_detection = QCheckBox(self._t(
            "検出0件の画像も通常Reviewに保存",
            "Also save zero-detection images to normal Review",
        ))
        form.addRow(
            self._t("未検出画像", "Zero detections"),
            _with_help(self.review_no_detection, self._t(
                "人数不一致の隔離とは別に、Review HTMLにも追加します。枚数が多くなることがあります。",
                "In addition to count-mismatch quarantine, add these images to the Review HTML. This can create many review items.",
            )),
        )
        self.preview_max_side = QSpinBox()
        self.preview_max_side.setRange(640, 4096)
        self.preview_max_side.setSingleStep(160)
        self.preview_max_side.setSuffix(" px")
        form.addRow(
            self._t("画面プレビューの大きさ", "On-screen preview size"),
            _with_help(self.preview_max_side, self._t(
                "検出や保存画像の精度には影響しません。画面表示の軽さだけが変わります。",
                "This does not affect detection or saved-image quality; it only changes UI memory and rendering cost.",
            )),
        )
        self.jpeg_quality = QSpinBox()
        self.jpeg_quality.setRange(70, 100)
        self.jpeg_quality.setSuffix(" %")
        form.addRow(
            self._t("JPEG/WebP保存品質", "JPEG/WebP quality"),
            _with_help(self.jpeg_quality, self._t(
                "高いほど画質とファイルサイズが上がります。PNGには影響しません。",
                "Higher values improve quality and increase file size. PNG files are unaffected.",
            )),
        )
        return tab

    def _build_tile_tab(self) -> QWidget:
        tab = QWidget()
        form = _form(tab)
        intro = QLabel(self._t(
            "大きな画像を重なりのある小領域に分け、小さく写った対象を見つけやすくします。数値を下げると分割検出が増え、処理時間も増えます。",
            "Large images are split into overlapping crops so small targets are easier to detect. Lower thresholds run more tiled inference and take longer.",
        ))
        intro.setWordWrap(True)
        form.addRow(intro)
        self.tile_trigger = QSpinBox()
        self.tile_trigger.setRange(512, 10000)
        self.tile_trigger.setSuffix(" px")
        form.addRow(
            self._t("2×2分割を始める画像サイズ", "Start 2×2 tiling at"),
            _with_help(self.tile_trigger, self._t(
                "画像の長辺がこの値以上で分割します。",
                "Tiling starts when the image's long side reaches this size.",
            )),
        )
        self.tile_3_trigger = QSpinBox()
        self.tile_3_trigger.setRange(512, 20000)
        self.tile_3_trigger.setSuffix(" px")
        form.addRow(
            self._t("3×3分割に切り替えるサイズ", "Switch to 3×3 tiling at"),
            _with_help(self.tile_3_trigger, self._t(
                "非常に大きな画像だけを9分割します。",
                "Only very large images are split into nine crops.",
            )),
        )
        self.tile_overlap = QSpinBox()
        self.tile_overlap.setRange(0, 40)
        self.tile_overlap.setSuffix(" %")
        form.addRow(
            self._t("分割領域の重なり", "Crop overlap"),
            _with_help(self.tile_overlap, self._t(
                "分割の境界で対象が切れるのを防ぎます。上げると安全ですが処理が増えます。",
                "Prevents targets from being cut at crop boundaries. Higher values are safer but increase work.",
            )),
        )
        return tab

    def _build_merge_tab(self) -> QWidget:
        tab = QWidget()
        form = _form(tab)
        warning = QLabel(self._t(
            "ここは誤検出が多い場合にだけ調整してください。通常は既定値が推奨です。",
            "Change these only when duplicate or merged detections are a real problem. Defaults are recommended.",
        ))
        warning.setWordWrap(True)
        warning.setObjectName("dialogNote")
        form.addRow(warning)
        self.model_iou = _float_spin(0.10, 0.95, 0.70, 0.05, 2)
        form.addRow(
            self._t("モデル内の重複除去 (NMS IoU)", "Model duplicate suppression (NMS IoU)"),
            _with_help(self.model_iou, self._t(
                "低くすると近い候補を消しやすく、高くすると候補を残しやすくなります。",
                "Lower values suppress nearby candidates more aggressively; higher values keep more candidates.",
            )),
        )
        self.merge_iou = _float_spin(0.10, 0.95, 0.45, 0.05, 2)
        form.addRow(
            self._t("別パスの検出を同一とみなす重なり (IoU)", "Cross-pass overlap treated as one target (IoU)"),
            _with_help(self.merge_iou, self._t(
                "低くするとまとめやすく、下げすぎると別の箇所まで1件になることがあります。",
                "Lower values merge more readily. Too low can combine separate targets.",
            )),
        )
        self.merge_ios = _float_spin(0.10, 1.00, 0.70, 0.05, 2)
        form.addRow(
            self._t("大小の箱が入れ子のときの統合 (IoS)", "Nested-box merge threshold (IoS)"),
            _with_help(self.merge_ios, self._t(
                "大きな箱の中に小さな箱がある場合に、同じ対象として広い方へまとめる基準です。",
                "Controls when a smaller box inside a larger box is treated as the same target and unioned to the wider area.",
            )),
        )
        return tab

    def _load(self, config: PipelineConfig) -> None:
        index = self.model_level.findData(config.model_level)
        self.model_level.setCurrentIndex(max(0, index))
        self.model_iou.setValue(config.model_iou_threshold)
        self.merge_iou.setValue(config.merge_iou_threshold)
        self.merge_ios.setValue(config.merge_ios_threshold)
        self.tile_trigger.setValue(config.tile_trigger_px)
        self.tile_3_trigger.setValue(config.tile_grid_3_trigger_px)
        self.tile_overlap.setValue(round(config.tile_overlap * 100))
        self.review_no_detection.setChecked(config.copy_no_detection_to_review)
        self.preview_max_side.setValue(config.preview_max_side)
        self.jpeg_quality.setValue(config.jpeg_quality)

    def _reset_defaults(self) -> None:
        self._load(PipelineConfig(language=self.language))

    def _accept_if_valid(self) -> None:
        if self.tile_3_trigger.value() < self.tile_trigger.value():
            self.tile_3_trigger.setValue(self.tile_trigger.value())
        self.accept()

    def apply_to(self, config: PipelineConfig) -> PipelineConfig:
        config.model_level = str(self.model_level.currentData())
        config.model_iou_threshold = self.model_iou.value()
        config.merge_iou_threshold = self.merge_iou.value()
        config.merge_ios_threshold = self.merge_ios.value()
        config.tile_trigger_px = self.tile_trigger.value()
        config.tile_grid_3_trigger_px = self.tile_3_trigger.value()
        config.tile_overlap = self.tile_overlap.value() / 100.0
        config.copy_no_detection_to_review = self.review_no_detection.isChecked()
        config.preview_max_side = self.preview_max_side.value()
        config.jpeg_quality = self.jpeg_quality.value()
        return config


def _form(parent: QWidget) -> QFormLayout:
    form = QFormLayout(parent)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setVerticalSpacing(14)
    return form


def _with_help(control: QWidget, description: str) -> QWidget:
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(control)
    help_label = QLabel(description)
    help_label.setWordWrap(True)
    help_label.setObjectName("dialogNote")
    layout.addWidget(help_label)
    control.setToolTip(description)
    return wrapper


def _float_spin(low: float, high: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin
