from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .ui.verifier_lab_dialog import VerifierLabDialog


def main(argv=None) -> int:
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)
    dialog = VerifierLabDialog(language="ja")
    dialog.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
