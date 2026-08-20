from __future__ import annotations


def main() -> int:
    # Import lazily so CLI/core users do not need Qt modules merely to import
    # character_mosaic.gui during tooling or tests.
    try:
        from .ui import main_window
        from .ui.settings_safety import EnhancedControlPanel
        from .ui.ux_enhancements import EnhancedMainWindow
    except ImportError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise RuntimeError("PySide6 がありません。install.bat を実行してください。") from exc
        raise

    # Keep the existing worker implementation intact and layer UX behavior over
    # the stable window/panel classes.
    main_window.ControlPanel = EnhancedControlPanel
    main_window.MainWindow = EnhancedMainWindow
    return main_window.main()


if __name__ == "__main__":
    raise SystemExit(main())
