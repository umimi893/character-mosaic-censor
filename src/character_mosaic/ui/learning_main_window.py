from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMessageBox

from .learning_dialog import LearningDialog
from .ux_enhancements import EnhancedMainWindow


class LearningMainWindow(EnhancedMainWindow):
    """Main window layer that exposes the persistent learning/mining workspace."""

    def __init__(self):
        super().__init__()
        self._learning_dialog: LearningDialog | None = None
        self._pending_close_after_mining = False
        requested = getattr(self.controls, "learning_requested", None)
        if requested is not None:
            requested.connect(self.open_learning_dialog)

    def open_learning_dialog(self) -> None:
        if self._learning_dialog is None:
            self._learning_dialog = LearningDialog(self._language, self)
            self._learning_dialog.mining_stopped.connect(self._on_mining_stopped)
        self._learning_dialog.setEnabled(self._thread is None)
        self._learning_dialog.show()
        self._learning_dialog.raise_()
        self._learning_dialog.activateWindow()

    def _miner_active(self) -> bool:
        return bool(
            self._learning_dialog is not None
            and getattr(self._learning_dialog, "_thread", None) is not None
        )

    def _warn_miner_active(self) -> None:
        QMessageBox.information(
            self,
            self._t("GPU使用中", "GPU in use"),
            self._t(
                "過去画像の採掘が動作中です。学習画面で停止してから通常処理を開始してください。",
                "Legacy-corpus mining is running. Stop it in the learning window before starting normal processing.",
            ),
        )

    def start_processing(self) -> None:
        if self._miner_active():
            self._warn_miner_active()
            return
        if self._learning_dialog is not None:
            self._learning_dialog.setEnabled(False)
        super().start_processing()
        # Validation errors can return before a worker thread is created.
        if self._thread is None and self._learning_dialog is not None:
            self._learning_dialog.setEnabled(True)

    def process_dropped_file(self, path_text: str) -> None:
        if self._miner_active():
            self._warn_miner_active()
            return
        if self._learning_dialog is not None:
            self._learning_dialog.setEnabled(False)
        super().process_dropped_file(path_text)
        if self._thread is None and self._learning_dialog is not None:
            self._learning_dialog.setEnabled(True)

    def _thread_finished(self) -> None:
        super()._thread_finished()
        if self._learning_dialog is not None:
            self._learning_dialog.setEnabled(True)

    def _on_mining_stopped(self) -> None:
        if not self._pending_close_after_mining:
            return
        self._pending_close_after_mining = False
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        # Closing a parent QObject while its child-owned QThread is still alive
        # can destroy the thread prematurely. Request cooperative mining stop and
        # retry the window close only after LearningDialog confirms QThread exit.
        if self._miner_active():
            self._pending_close_after_mining = True
            assert self._learning_dialog is not None
            self._learning_dialog.stop_mining()
            event.ignore()
            return
        super().closeEvent(event)
