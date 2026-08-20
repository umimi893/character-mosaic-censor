from __future__ import annotations


def main() -> int:
    # pythonw.exe on Windows can expose stdout/stderr as None. Third-party
    # inference/progress libraries may write to those streams, so provide safe
    # null-device replacements before importing or running the GUI stack.
    from .runtime_streams import ensure_standard_streams

    ensure_standard_streams()

    # Import lazily so CLI/core users do not need Qt modules merely to import
    # character_mosaic.gui during tooling or tests.
    try:
        from .ui import main_window
        from .ui.output_follow import OutputFollowControlPanel
        from .ui.theme import DARK_STYLE
        from .ui.ux_enhancements import EnhancedMainWindow
    except ImportError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise RuntimeError("PySide6 がありません。install.bat を実行してください。") from exc
        raise

    # Keep the existing worker implementation intact and layer UX behavior over
    # the stable window/panel classes. Replace the legacy partial stylesheet with
    # a complete one so Windows light mode cannot leak white native backgrounds
    # into the dark application UI.
    main_window.ControlPanel = OutputFollowControlPanel
    main_window.MainWindow = EnhancedMainWindow
    main_window._STYLE = DARK_STYLE
    return main_window.main()


if __name__ == "__main__":
    raise SystemExit(main())
