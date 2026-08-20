from __future__ import annotations

from pathlib import Path


def default_output_for_input(input_text: str) -> str:
    """Return the default output folder for an input path field value."""
    text = input_text.strip()
    if not text:
        return ""
    return str(Path(text) / "_censored")
