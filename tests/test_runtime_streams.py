from __future__ import annotations

import io
import sys

from character_mosaic.runtime_streams import ensure_standard_streams


def test_missing_standard_streams_are_replaced_with_writable_streams():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = None
        sys.stderr = None
        ensure_standard_streams()

        assert sys.stdout is not None
        assert sys.stderr is not None
        assert sys.stdout.write("test") == 4
        assert sys.stderr.write("test") == 4
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def test_existing_standard_streams_are_preserved():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        ensure_standard_streams()
        assert sys.stdout is stdout
        assert sys.stderr is stderr
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
