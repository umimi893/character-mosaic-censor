from __future__ import annotations

from .learning_dialog import LearningDialog
from .ux_enhancements import EnhancedMainWindow


class LearningMainWindow(EnhancedMainWindow):
    """Main window layer that exposes the persistent learning/mining workspace."""

    def __init__(self):
        super().__init__()
        self._learning_dialog: LearningDialog | None = None
        requested = getattr(self.controls, "learning_requested", None)
        if requested is not None:
            requested.connect(self.open_learning_dialog)

    def open_learning_dialog(self) -> None:
        if self._learning_dialog is None:
            self._learning_dialog = LearningDialog(self._language, self)
        self._learning_dialog.show()
        self._learning_dialog.raise_()
        self._learning_dialog.activateWindow()
