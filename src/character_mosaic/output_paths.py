from __future__ import annotations

from pathlib import Path


def default_output_for_input(input_text: str) -> str:
    """Return the default output folder for an input path field value."""
    text = input_text.strip()
    if not text:
        return ""
    return str(Path(text) / "_censored")


def output_differs_from_default(input_text: str, output_text: str) -> bool:
    """Return True when a saved output looks intentionally customized."""
    output = output_text.strip()
    if not output:
        return False
    return output != default_output_for_input(input_text)
