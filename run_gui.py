from __future__ import annotations

import ctypes
import os
import traceback
from pathlib import Path

from character_mosaic.gui import main


def _report_startup_error() -> None:
    details = traceback.format_exc()
    try:
        Path("startup_error.log").write_text(details, encoding="utf-8")
    except OSError:
        pass

    message = (
        "Character Mosaic Censor could not start.\n\n"
        "Details were written to startup_error.log.\n"
        "You can also run diagnose.bat for environment diagnostics."
    )
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, "Character Mosaic Censor", 0x10)
            return
        except Exception:
            pass
    print(message)
    print(details)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        _report_startup_error()
        raise SystemExit(1)
