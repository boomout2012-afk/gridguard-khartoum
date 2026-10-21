"""Khartoum Atlas visual system shared by the native desktop UI."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic light-cartography colors with WCAG AA text contrast."""

    canvas: str = "#F2F6FA"
    surface: str = "#FFFFFF"
    ink: str = "#193348"
    muted: str = "#516579"
    nav: str = "#142C48"
    boundary: str = "#78909E"
    nile: str = "#007C91"
    evidence: str = "#007C91"
    full: str = "#15705C"
    critical: str = "#975F19"
    off: str = "#963A45"
    focus: str = "#005FCC"
    surface_raised: str = "#E6EDF3"
    accent_dark: str = "#005B70"
    rail_surface: str = "#284863"
    rail_text: str = "#FFFFFF"
    rail_muted: str = "#BFCFDD"
    loading_surface: str = "#FCF5E3"

    @property
    def text(self) -> str:
        return self.ink

    @property
    def text_muted(self) -> str:
        return self.muted

    @property
    def rail(self) -> str:
        return self.nav

    @property
    def border(self) -> str:
        return self.boundary

    @property
    def river(self) -> str:
        return self.nile

    @property
    def accent(self) -> str:
        return self.evidence

    @property
    def success(self) -> str:
        return self.full

    @property
    def warning(self) -> str:
        return self.critical

    @property
    def danger(self) -> str:
        return self.off


PALETTE = Palette()

DISPLAY_FONT_FAMILY = "Source Serif 4"
BODY_FONT_FAMILY = "Atkinson Hyperlegible"
MONO_FONT_FAMILY = "IBM Plex Mono"

_DISPLAY_FONT_FILES = (
    "SourceSerif4-Regular.ttf",
    "SourceSerif4-Semibold.ttf",
)
_BODY_FONT_FILES = (
    "AtkinsonHyperlegible-Regular.ttf",
    "AtkinsonHyperlegible-Bold.ttf",
)
_MONO_FONT_FILES = (
    "IBMPlexMono-Regular.ttf",
    "IBMPlexMono-SemiBold.ttf",
)
_EXPECTED_FONT_FAMILIES = (
    DISPLAY_FONT_FAMILY,
    BODY_FONT_FAMILY,
    MONO_FONT_FAMILY,
)
_FONT_FILES_BY_FAMILY = dict(
    zip(
        _EXPECTED_FONT_FAMILIES,
        (_DISPLAY_FONT_FILES, _BODY_FONT_FILES, _MONO_FONT_FILES),
        strict=True,
    )
)
_REGISTERED_FONT_PATHS: set[Path] = set()


def register_presentation_fonts(font_directory: Path | None = None) -> tuple[str, ...]:
    """Register the bundled, offline presentation fonts and return their families."""

    directory = font_directory or Path(__file__).resolve().parents[1] / "assets" / "fonts"
    for expected_family, filenames in _FONT_FILES_BY_FAMILY.items():
        discovered_families: set[str] = set()
        for filename in filenames:
            font_path = (directory / filename).resolve()
            if not font_path.is_file():
                raise RuntimeError(f"bundled font is missing: {font_path}")
            if font_path in _REGISTERED_FONT_PATHS:
                discovered_families.add(expected_family)
                continue
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            if font_id < 0:
                raise RuntimeError(f"Qt could not register bundled font: {font_path}")
            registered_families = set(QFontDatabase.applicationFontFamilies(font_id))
            if expected_family not in registered_families:
                names = ", ".join(sorted(registered_families)) or "none"
                raise RuntimeError(f"{font_path} registered unexpected font families: {names}")
            _REGISTERED_FONT_PATHS.add(font_path)
            discovered_families.update(registered_families)
        if expected_family not in discovered_families:
            raise RuntimeError(f"bundled font family did not register: {expected_family}")
    return _EXPECTED_FONT_FAMILIES


STATUS_COLORS = {
    0: QColor(PALETTE.off),
    1: QColor(PALETTE.critical),
    2: QColor(PALETTE.full),
}
STATUS_TEXT = {
    0: "× OFF",
    1: "! CRITICAL ONLY",
    2: "• FULL",
}


def apply_theme(application: QApplication) -> None:
    """Apply the single offline visual system to a Qt application."""

    try:
        register_presentation_fonts()
    except RuntimeError as error:
        warnings.warn(
            f"Khartoum Atlas bundled typography unavailable; using Windows fallbacks: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
    font = QFont()
    font.setFamilies([BODY_FONT_FAMILY, "Segoe UI Variable Text", "Segoe UI", "Arial"])
    font.setPointSizeF(10.5)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    application.setFont(font)
    application.setStyleSheet(_STYLE_SHEET)


_STYLE_SHEET = f"""
QWidget {{
    color: {PALETTE.text};
    background: {PALETTE.canvas};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI Variable Text", "Segoe UI", "Arial";
    font-size: 14px;
}}
QLabel {{
    background: transparent;
}}
QMainWindow, QWidget#applicationShell, QFrame#contentSurface {{
    background: {PALETTE.canvas};
}}
QFrame#navigationRail {{
    background: {PALETTE.rail};
    border: 0;
}}
QLabel#railKicker {{
    color: {PALETTE.rail_muted};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0px;
}}
QLabel#railTitle {{
    color: {PALETTE.rail_text};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI", "Arial";
    font-size: 23px;
    font-weight: 700;
}}
QLabel#railSubtitle, QLabel[railMuted="true"] {{
    color: {PALETTE.rail_muted};
}}
QFrame#railRule {{
    background: #3C5870;
    border: 0;
    max-height: 1px;
    min-height: 1px;
}}
QPushButton[nav="true"] {{
    color: {PALETTE.rail_muted};
    background: transparent;
    border: 0;
    border-left: 3px solid transparent;
    border-radius: 0;
    min-height: 44px;
    padding: 4px 10px;
    text-align: left;
    font-weight: 600;
}}
QPushButton[nav="true"]:hover {{
    color: {PALETTE.rail_text};
    background: {PALETTE.rail_surface};
}}
QPushButton[nav="true"]:checked {{
    color: {PALETTE.rail_text};
    background: {PALETTE.rail_surface};
    border-left: 3px solid {PALETTE.evidence};
    font-weight: 700;
}}
QPushButton[nav="true"]:focus {{
    color: {PALETTE.ink};
    background: {PALETTE.surface};
    border: 2px solid {PALETTE.focus};
    border-left: 4px solid {PALETTE.focus};
}}
QPushButton[nav="true"]:checked:focus {{
    color: {PALETTE.rail_text};
    background: {PALETTE.rail_surface};
    border: 2px solid {PALETTE.focus};
    border-left: 4px solid {PALETTE.focus};
}}
QFrame#trustStrip {{
    background: {PALETTE.surface};
    border: 0;
    border-top: 1px solid {PALETTE.border};
}}
QFrame#appHeader, QFrame[card="true"] {{
    background: {PALETTE.surface};
    border: 1px solid {PALETTE.border};
    border-radius: 8px;
}}
QFrame[cardVariant="accent"] {{
    background: #E9F4F6;
    border: 1px solid #A1BCC7;
    border-left: 4px solid {PALETTE.accent};
}}
QFrame[cardVariant="quiet"] {{
    background: #ECF2F7;
}}
QLabel#brandTitle {{
    color: {PALETTE.text};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI", "Arial";
    font-size: 25px;
    font-weight: 700;
}}
QLabel#brandSubtitle, QLabel[muted="true"] {{
    color: {PALETTE.text_muted};
}}
QLabel[kicker="true"] {{
    color: {PALETTE.accent_dark};
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0px;
}}
QLabel[compact="true"] {{
    font-size: 12px;
}}
QLabel[claim="boundary"] {{
    color: {PALETTE.text_muted};
    background: #EDF3F7;
    border: 1px solid #A1BCC7;
    border-radius: 3px;
    padding: 4px 7px;
    font-size: 11px;
}}
QLabel[heading="page"] {{
    color: {PALETTE.text};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI", "Arial";
    font-size: 30px;
    font-weight: 700;
}}
QLabel[heading="section"] {{
    color: {PALETTE.text};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI", "Arial";
    font-size: 18px;
    font-weight: 700;
}}
QLabel[metric="value"] {{
    color: {PALETTE.ink};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI", "Arial";
    font-size: 26px;
    font-weight: 700;
}}
QLabel[detail="value"] {{
    color: {PALETTE.accent_dark};
    font-family: "{MONO_FONT_FAMILY}", "Cascadia Mono", "Consolas";
    font-size: 18px;
    font-weight: 700;
}}
QLabel[badge="verified"] {{
    color: {PALETTE.success};
    background: #E3F1E8;
    border: 1px solid #7FAF96;
    border-radius: 3px;
    padding: 4px 8px;
    font-weight: 700;
}}
QLabel[badge="locked"] {{
    color: {PALETTE.accent_dark};
    background: #E1F0F3;
    border: 1px solid #A1BCC7;
    border-radius: 3px;
    padding: 4px 8px;
    font-weight: 700;
}}
QLabel[badge="neutral"] {{
    color: {PALETTE.text_muted};
    background: #ECF2F7;
    border: 1px solid {PALETTE.border};
    border-radius: 3px;
    padding: 4px 8px;
    font-weight: 650;
}}
QLabel[state="empty"] {{
    color: {PALETTE.text_muted};
    background: #ECF2F7;
    border: 1px dashed {PALETTE.border};
    padding: 5px 8px;
}}
QLabel[state="loading"] {{
    color: {PALETTE.warning};
    background: {PALETTE.loading_surface};
    border: 1px solid #C7A95F;
    padding: 5px 8px;
}}
QLabel[state="error"] {{
    color: {PALETTE.danger};
    background: #F5DFDA;
    border: 1px solid #C98D84;
    padding: 5px 8px;
}}
QLabel[state="ready"] {{
    color: {PALETTE.success};
    background: #E2F0EA;
    border: 1px solid #73A897;
    padding: 5px 8px;
}}
QLabel[eyebrow="true"] {{
    color: {PALETTE.text_muted};
    font-size: 12px;
    font-weight: 700;
}}
QLabel[mono="true"] {{
    font-family: "{MONO_FONT_FAMILY}", "Cascadia Mono", "Consolas";
}}
QPushButton {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    border: 1px solid #96ACBC;
    border-radius: 4px;
    min-height: 34px;
    padding: 4px 12px;
    font-weight: 650;
}}
QPushButton:hover {{
    background: #DDE9F0;
    border-color: {PALETTE.accent};
}}
QPushButton:pressed {{
    background: #C9DDE8;
}}
QPushButton:disabled {{
    color: #7B837E;
    background: #E9EDF1;
    border-color: #CBD5DD;
}}
QPushButton[primary="true"] {{
    color: {PALETTE.surface};
    background: {PALETTE.accent_dark};
    border-color: {PALETTE.evidence};
}}
QPushButton[primary="true"]:hover {{
    background: {PALETTE.accent_dark};
}}
QPushButton[quiet="true"] {{
    background: transparent;
}}
QPushButton:focus, QComboBox:focus, QSpinBox:focus,
QSlider:focus, QGraphicsView:focus, QTableWidget:focus {{
    border: 2px solid {PALETTE.focus};
}}
QComboBox, QSpinBox {{
    color: {PALETTE.text};
    background: {PALETTE.surface};
    border: 1px solid #96ACBC;
    border-radius: 4px;
    min-height: 34px;
    padding: 2px 9px;
}}
QComboBox:disabled, QSpinBox:disabled {{
    color: #747D77;
    background: #E9EDF1;
}}
QComboBox QAbstractItemView {{
    color: {PALETTE.text};
    background: {PALETTE.surface};
    selection-color: {PALETTE.surface};
    selection-background-color: {PALETTE.accent};
}}
QSlider::groove:horizontal {{
    height: 5px;
    background: #C2D2DE;
}}
QSlider::handle:horizontal {{
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {PALETTE.accent};
    border: 1px solid {PALETTE.accent_dark};
}}
QTableWidget {{
    color: {PALETTE.text};
    background: {PALETTE.surface};
    alternate-background-color: #F3F7FA;
    border: 1px solid {PALETTE.border};
    gridline-color: #D6E1E9;
    selection-color: {PALETTE.text};
    selection-background-color: #D9EDF2;
}}
QListWidget {{
    color: {PALETTE.text};
    background: {PALETTE.surface};
    border: 1px solid {PALETTE.border};
}}
QListWidget[localityList="true"] {{
    font-size: 13px;
}}
QListWidget::item {{
    padding: 1px 4px;
}}
QListWidget::item:selected {{
    color: {PALETTE.surface};
    background: {PALETTE.nile};
}}
QListWidget:focus {{
    border: 2px solid {PALETTE.focus};
}}
QHeaderView::section {{
    color: {PALETTE.rail_text};
    background: {PALETTE.rail};
    border: 0;
    border-right: 1px solid #3C5870;
    padding: 8px;
    font-weight: 700;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {PALETTE.canvas};
    border: 0;
}}
QToolTip {{
    color: {PALETTE.rail_text};
    background: {PALETTE.rail};
    border: 1px solid #708A81;
    padding: 6px;
}}
QScrollBar:vertical {{
    width: 12px;
    margin: 0;
    background: #E6EDF3;
}}
QScrollBar::handle:vertical {{
    background: #93A9BA;
    border-radius: 8px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""
