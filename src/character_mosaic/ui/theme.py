from __future__ import annotations


# Keep the application readable regardless of the Windows light/dark setting.
# The previous stylesheet set foreground colors globally but left several
# container/viewport backgrounds to the native Windows palette. On a light
# Windows theme that produced white panels with near-white text.
DARK_STYLE = r"""
QWidget {
    background-color: #11161d;
    color: #dfe7ef;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #11161d;
    color: #dfe7ef;
}
QScrollArea, QAbstractScrollArea, QAbstractScrollArea::viewport {
    background-color: #151b23;
    color: #dfe7ef;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: #151b23;
}
QLabel, QCheckBox, QRadioButton {
    background-color: transparent;
    color: #dfe7ef;
}
QGroupBox {
    border: 1px solid #303a46;
    border-radius: 9px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
    background-color: #171e27;
    color: #dce6f0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    background-color: #171e27;
    color: #dce6f0;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0f141b;
    color: #e5edf5;
    border: 1px solid #34404e;
    border-radius: 6px;
    padding: 6px;
    min-height: 22px;
    selection-background-color: #3476a3;
    selection-color: #ffffff;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background-color: #171d24;
    color: #6f7b87;
    border-color: #29323c;
}
QComboBox QAbstractItemView {
    background-color: #111820;
    color: #e5edf5;
    border: 1px solid #34404e;
    selection-background-color: #24597d;
    selection-color: #ffffff;
    outline: 0;
}
QPushButton {
    background-color: #252f3b;
    color: #dfe7ef;
    border: 1px solid #3a4756;
    border-radius: 7px;
    padding: 7px 10px;
}
QPushButton:hover { background-color: #2d3947; }
QPushButton:pressed { background-color: #202a35; }
QPushButton:disabled {
    color: #687380;
    background-color: #1b222a;
    border-color: #29323c;
}
QPushButton#startButton {
    background-color: #176b4e;
    border-color: #228465;
    color: #f2fff9;
    font-weight: 700;
}
QPushButton#startButton:hover { background-color: #1c7a5a; }
QPushButton#stopButton {
    background-color: #69343b;
    border-color: #84454f;
    color: #fff3f4;
    font-weight: 700;
}
QPushButton[previewMode="true"]:checked {
    background-color: #24597d;
    border-color: #3476a3;
}
QLabel#panelTitle {
    background-color: transparent;
    color: #f0f5fa;
    font-size: 21px;
    font-weight: 750;
}
QLabel#panelSubtitle {
    background-color: transparent;
    color: #8d9aa8;
    margin-bottom: 3px;
}
QLabel#runtimeLabel, QLabel#summaryLabel, QLabel#previewStatus, QLabel#dialogNote {
    background-color: transparent;
    color: #9eabb8;
}
QLabel#previewFile { font-weight: 600; }
QProgressBar {
    color: #dfe7ef;
    border: 1px solid #34404e;
    border-radius: 5px;
    text-align: center;
    background-color: #0f141b;
}
QProgressBar::chunk {
    background-color: #2c8d68;
    border-radius: 4px;
}
QStatusBar {
    background-color: #0d1218;
    color: #dfe7ef;
    border-top: 1px solid #28313c;
}
QStatusBar QLabel { background-color: transparent; }
QSplitter::handle { background-color: #27303b; width: 1px; }
QTabWidget::pane {
    background-color: #151b23;
    border: 1px solid #303a46;
}
QTabBar::tab {
    background-color: #202a34;
    color: #cbd5df;
    border: 1px solid #303a46;
    padding: 7px 12px;
}
QTabBar::tab:selected {
    background-color: #2a3744;
    color: #ffffff;
}
QMenu {
    background-color: #151b23;
    color: #dfe7ef;
    border: 1px solid #34404e;
}
QMenu::item:selected { background-color: #24597d; }
QToolTip {
    background-color: #202a34;
    color: #f0f5fa;
    border: 1px solid #455363;
    padding: 4px;
}
QScrollBar:vertical {
    background-color: #111820;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #384653;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover { background-color: #465767; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #111820;
    height: 12px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: #384653;
    min-width: 30px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover { background-color: #465767; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""
