from __future__ import annotations

import os
import sys
from typing import TextIO


# Keep replacement streams alive for the process lifetime. On Windows,
# pythonw.exe starts without a console and CPython may expose stdout/stderr as
# None. Some third-party ML/progress libraries assume a writable stream exists.
_replacement_streams: list[TextIO] = []


def ensure_standard_streams() -> None:
    """Provide writable stdout/stderr when running without a console.

    Normal console/CLI streams are preserved exactly as they are. Missing
    streams are redirected to the operating system null device so progress
    writers cannot crash the GUI process.
    """

    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8", buffering=1)
        setattr(sys, name, stream)
        _replacement_streams.append(stream)
