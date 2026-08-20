from character_mosaic.ui.theme import DARK_STYLE


def test_dark_theme_sets_widget_background_not_only_text_color():
    compact = " ".join(DARK_STYLE.split())
    assert "QWidget {" in DARK_STYLE
    assert "background-color: #11161d" in DARK_STYLE
    assert "QAbstractScrollArea::viewport" in DARK_STYLE
    assert "QScrollArea > QWidget > QWidget" in DARK_STYLE
    assert "QComboBox QAbstractItemView" in DARK_STYLE
    assert "color: #dfe7ef" in compact
