from __future__ import annotations


def main() -> int:
    # Import lazily so CLI/core users do not need Qt modules merely to import
    # character_mosaic.gui during tooling or tests.
    try:
        from .ui.main_window import main as qt_main
    except ImportError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise RuntimeError("PySide6 がありません。install.bat を実行してください。") from exc
        raise
    return qt_main()


if __name__ == "__main__":
    raise SystemExit(main())
