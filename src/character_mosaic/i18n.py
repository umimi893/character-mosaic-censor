from __future__ import annotations

SUPPORTED_LANGUAGES = ("ja", "en")


def normalize_language(language: str | None) -> str:
    value = str(language or "ja").strip().lower()
    return value if value in SUPPORTED_LANGUAGES else "ja"


def t(language: str, ja: str, en: str) -> str:
    return en if normalize_language(language) == "en" else ja
