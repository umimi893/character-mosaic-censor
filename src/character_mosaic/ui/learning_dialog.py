from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QThread, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..corpus_miner import CorpusMinerConfig
from ..experience_store import ExperienceStore, MiningStats, default_learning_root
from ..i18n import normalize_language, t
from ..workers.corpus_miner_worker import CorpusMinerWorker


class LearningDialog(QDialog):
    """Manage resumable mining of noisy legacy image folders and ZIPs."""

    def __init__(self, language: str = "ja", parent=None):
        super().__init__(parent)
        self._language = normalize_language(language)
        self._settings = QSettings("CharacterMosaicCensor", "CharacterMosaicCensor")
        self._thread: QThread | None = None
        self._worker: CorpusMinerWorker | None = None
        self._close_when_done = False

        self.setWindowTitle(self._t("自動学習素材の採掘", "Automatic corpus mining"))
        self.resize(720, 620)
        self.setMinimumSize(620, 520)

        layout = QVBoxLayout(self)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)

        self.roots = QListWidget()
        self.roots.setAlternatingRowColors(True)
        layout.addWidget(self.roots, 1)

        root_buttons = QHBoxLayout()
        self.add_button = QPushButton()
        self.remove_button = QPushButton()
        self.open_store_button = QPushButton()
        self.add_button.clicked.connect(self._add_root)
        self.remove_button.clicked.connect(self._remove_root)
        self.open_store_button.clicked.connect(self._open_learning_store)
        root_buttons.addWidget(self.add_button)
        root_buttons.addWidget(self.remove_button)
        root_buttons.addStretch(1)
        root_buttons.addWidget(self.open_store_button)
        layout.addLayout(root_buttons)

        self.include_zip = QCheckBox()
        self.include_zip.setChecked(True)
        self.idle_only = QCheckBox()
        self.idle_only.setChecked(True)
        self.save_crops = QCheckBox()
        self.save_crops.setChecked(True)
        layout.addWidget(self.include_zip)
        layout.addWidget(self.idle_only)
        layout.addWidget(self.save_crops)

        util_row = QHBoxLayout()
        self.util_label = QLabel()
        self.max_gpu_util = QSpinBox()
        self.max_gpu_util.setRange(5, 95)
        self.max_gpu_util.setSuffix(" %")
        self.max_gpu_util.setValue(30)
        util_row.addWidget(self.util_label)
        util_row.addWidget(self.max_gpu_util)
        util_row.addStretch(1)
        layout.addLayout(util_row)

        limit_row = QHBoxLayout()
        self.limit_label = QLabel()
        self.max_images = QSpinBox()
        self.max_images.setRange(0, 10_000_000)
        self.max_images.setSpecialValueText(self._t("制限なし", "Unlimited"))
        self.max_images.setValue(0)
        limit_row.addWidget(self.limit_label)
        limit_row.addWidget(self.max_images)
        limit_row.addStretch(1)
        layout.addLayout(limit_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(self.stats_label.textInteractionFlags())
        layout.addWidget(self.stats_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton()
        self.stop_button = QPushButton()
        self.close_button = QPushButton()
        self.start_button.clicked.connect(self.start_mining)
        self.stop_button.clicked.connect(self.stop_mining)
        self.close_button.clicked.connect(self.close)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self._load_settings()
        self.retranslate()
        self._refresh_store_stats()

    def _t(self, ja: str, en: str) -> str:
        return t(self._language, ja, en)

    def retranslate(self) -> None:
        self.setWindowTitle(self._t("自動学習素材の採掘", "Automatic corpus mining"))
        self.description.setText(self._t(
            "古い生成画像を教師データとして信用するのではなく、現在の検出器が反応する『候補』を掘り出します。破損・重複・曖昧候補は自動でスキップ/隔離し、元画像やZIPは変更しません。途中停止しても次回は解析済み素材を飛ばして再開します。",
            "Legacy images are treated as a mine, not trusted ground truth. The current detector extracts candidate regions; corrupt, duplicate, and ambiguous items are skipped or quarantined automatically. Source images and ZIPs are never modified, and later runs resume past already-seen material.",
        ))
        self.add_button.setText(self._t("フォルダ追加", "Add folder"))
        self.remove_button.setText(self._t("選択を削除", "Remove selected"))
        self.open_store_button.setText(self._t("学習データを開く", "Open learning data"))
        self.include_zip.setText(self._t("ZIP内の画像も直接解析", "Read images inside ZIP archives"))
        self.idle_only.setText(self._t("GPUが空いている時だけ進める", "Run only while the GPU is idle"))
        self.save_crops.setText(self._t("高信頼候補の小さなcropを保存", "Store compact crops for high-confidence candidates"))
        self.util_label.setText(self._t("GPU使用率がこれ以下なら開始", "Start below GPU utilization"))
        self.limit_label.setText(self._t("今回の最大解析枚数", "Maximum images this run"))
        self.start_button.setText(self._t("▶ 採掘開始", "▶ Start mining"))
        self.stop_button.setText(self._t("■ 停止", "■ Stop"))
        self.close_button.setText(self._t("閉じる", "Close"))
        if self._thread is None:
            self.status_label.setText(self._t("待機中", "Ready"))

    def _load_settings(self) -> None:
        roots = self._settings.value("learning/corpus_roots", [])
        if isinstance(roots, str):
            roots = [roots]
        for value in roots or []:
            path = Path(str(value)).expanduser()
            if str(path) and not self._contains_root(path):
                self.roots.addItem(str(path))
        self.include_zip.setChecked(_as_bool(self._settings.value("learning/include_zip", True)))
        self.idle_only.setChecked(_as_bool(self._settings.value("learning/idle_only", True)))
        self.save_crops.setChecked(_as_bool(self._settings.value("learning/save_crops", True)))
        self.max_gpu_util.setValue(int(self._settings.value("learning/max_gpu_util", 30)))

    def _save_settings(self) -> None:
        roots = [self.roots.item(i).text() for i in range(self.roots.count())]
        self._settings.setValue("learning/corpus_roots", roots)
        self._settings.setValue("learning/include_zip", self.include_zip.isChecked())
        self._settings.setValue("learning/idle_only", self.idle_only.isChecked())
        self._settings.setValue("learning/save_crops", self.save_crops.isChecked())
        self._settings.setValue("learning/max_gpu_util", self.max_gpu_util.value())
        self._settings.sync()

    def _add_root(self) -> None:
        start = self.roots.item(0).text() if self.roots.count() else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, self._t("学習素材フォルダ", "Corpus folder"), start)
        if not path:
            return
        candidate = Path(path).resolve()
        if not self._contains_root(candidate):
            self.roots.addItem(str(candidate))
            self._save_settings()

    def _remove_root(self) -> None:
        for item in self.roots.selectedItems():
            self.roots.takeItem(self.roots.row(item))
        self._save_settings()

    def _contains_root(self, path: Path) -> bool:
        target = str(path)
        return any(self.roots.item(i).text() == target for i in range(self.roots.count()))

    def _open_learning_store(self) -> None:
        root = default_learning_root()
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root.resolve())))

    def start_mining(self) -> None:
        if self._thread is not None:
            return
        roots = [Path(self.roots.item(i).text()) for i in range(self.roots.count())]
        roots = [root for root in roots if root.is_dir()]
        if not roots:
            QMessageBox.information(
                self,
                self._t("学習素材", "Corpus mining"),
                self._t("解析するフォルダを1つ以上追加してください。", "Add at least one folder to mine."),
            )
            return

        self._save_settings()
        config = CorpusMinerConfig(
            include_zip=self.include_zip.isChecked(),
            recursive=True,
            save_crops=self.save_crops.isChecked(),
            idle_gpu_only=self.idle_only.isChecked(),
            max_gpu_utilization=self.max_gpu_util.value(),
            max_images=self.max_images.value() or None,
        )
        thread = QThread(self)
        worker = CorpusMinerWorker(roots, config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.status.connect(self.status_label.setText)
        worker.root_started.connect(lambda root: self.status_label.setText(self._t(f"採掘中: {root}", f"Mining: {root}")))
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        self.progress.setVisible(True)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_options_enabled(False)
        thread.start()

    def stop_mining(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText(self._t("停止要求中…", "Stopping…"))

    def _on_progress(self, stats: MiningStats, message: str) -> None:
        self.status_label.setText(message)
        self.stats_label.setText(self._format_stats(stats))

    def _on_finished(self, aggregate: dict) -> None:
        self.status_label.setText(self._t("採掘完了", "Mining complete"))
        self.stats_label.setText(self._t(
            f"今回: 解析 {aggregate.get('processed', 0):,} / 重複 {aggregate.get('duplicates', 0):,} / "
            f"候補 {aggregate.get('candidates', 0):,} / GOLD負例 {aggregate.get('gold_negative', 0):,} / "
            f"隔離 {aggregate.get('quarantine', 0):,}",
            f"This run: processed {aggregate.get('processed', 0):,} / duplicates {aggregate.get('duplicates', 0):,} / "
            f"candidates {aggregate.get('candidates', 0):,} / GOLD negatives {aggregate.get('gold_negative', 0):,} / "
            f"quarantine {aggregate.get('quarantine', 0):,}",
        ))
        self._refresh_store_stats(append=True)

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(self._t(f"採掘エラー: {message}", f"Mining error: {message}"))
        QMessageBox.warning(self, self._t("採掘エラー", "Mining error"), message)

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.progress.setVisible(False)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_options_enabled(True)
        self._refresh_store_stats(append=True)
        if self._close_when_done:
            self._close_when_done = False
            self.close()

    def _set_options_enabled(self, enabled: bool) -> None:
        for widget in (
            self.roots, self.add_button, self.remove_button, self.include_zip,
            self.idle_only, self.save_crops, self.max_gpu_util, self.max_images,
        ):
            widget.setEnabled(enabled)

    def _refresh_store_stats(self, append: bool = False) -> None:
        try:
            data = ExperienceStore().stats()
            text = self._t(
                f"累計DB: 素材 {data.get('sources', 0):,} / 候補 {data.get('candidates', 0):,} / "
                f"GOLD負例 {data.get('candidate_negative_gold', 0):,}",
                f"Database total: sources {data.get('sources', 0):,} / candidates {data.get('candidates', 0):,} / "
                f"GOLD negatives {data.get('candidate_negative_gold', 0):,}",
            )
        except Exception as exc:
            text = self._t(f"学習DBを読めません: {exc}", f"Cannot read learning database: {exc}")
        if append and self.stats_label.text():
            self.stats_label.setText(self.stats_label.text() + "\n" + text)
        else:
            self.stats_label.setText(text)

    def _format_stats(self, stats: MiningStats) -> str:
        return self._t(
            f"発見 {stats.discovered:,} / 解析 {stats.processed:,} / 重複 {stats.duplicates:,} / "
            f"Skip {stats.skipped:,} / Error {stats.errors:,}\n"
            f"候補 {stats.candidates:,} / GOLD負例 {stats.gold_negative:,} / SILVER {stats.silver:,} / 隔離 {stats.quarantine:,}",
            f"Found {stats.discovered:,} / processed {stats.processed:,} / duplicates {stats.duplicates:,} / "
            f"skipped {stats.skipped:,} / errors {stats.errors:,}\n"
            f"candidates {stats.candidates:,} / GOLD negatives {stats.gold_negative:,} / SILVER {stats.silver:,} / quarantine {stats.quarantine:,}",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._thread is not None:
            self._close_when_done = True
            self.stop_mining()
            event.ignore()
            return
        self._save_settings()
        event.accept()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
