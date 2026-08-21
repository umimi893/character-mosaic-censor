from __future__ import annotations

import os


# GUI tests run headlessly. Keep one QApplication alive for the entire pytest
# process instead of letting individual tests create short-lived application
# wrappers. On Windows/PySide6, destroying/recreating the application between
# tests can terminate the interpreter in Qt's native layer (0xC0000409).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - GUI tests already skip without PySide6
    _TEST_QT_APP = None
else:
    _TEST_QT_APP = QApplication.instance() or QApplication([])
