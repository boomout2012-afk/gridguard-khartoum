"""Approved A/D/F visual system shared by the native desktop UI."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic light-cartography colors with WCAG AA text contrast."""

    canvas: str = "#F4F7F5"
    surface: str = "#FFFFFF"
    ink: str = "#183D40"
    muted: str = "#536A67"
    nav: str = "#126F68"
    boundary: str = "#D9E4DF"
    nile: str = "#78AEB3"
    evidence: str = "#126F68"
    full: str = "#23634F"
    critical: str = "#785821"
    off: str = "#8B3A35"
    selection: str = "#E47A5E"
    focus: str = "#005FCC"
    surface_raised: str = "#E6EEEA"
    accent_dark: str = "#126F68"
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

BODY_FONT_FAMILY = "Atkinson Hyperlegible"
DISPLAY_FONT_FAMILY = BODY_FONT_FAMILY
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
    "Source Serif 4",
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
STATUS_FILL_COLORS = {
    0: QColor("#DCE4DF"),
    1: QColor("#C0DED2"),
    2: QColor("#5C9D8B"),
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
    color: {PALETTE.ink}; background: {PALETTE.canvas};
    font-family: "{BODY_FONT_FAMILY}", "Segoe UI", "Arial"; font-size: 14px;
}}
QLabel, QWidget#navigationIdentity {{ background: transparent; }}
QMainWindow, QWidget#applicationShell, QStackedWidget#pageStack {{
    background: {PALETTE.canvas}; border: 0;
}}
QFrame#navigationBar {{
    background: {PALETTE.surface}; border: 1px solid #E4ECE7; border-radius: 22px;
}}
QLabel#brandTitle {{ font-size: 23px; font-weight: 700; }}
QLabel#brandSubtitle, QLabel[muted="true"] {{ color: {PALETTE.muted}; }}
QPushButton[nav="true"] {{
    color: {PALETTE.muted}; background: transparent; border: 2px solid transparent;
    border-radius: 14px; padding: 0 18px; min-height: 40px; font-weight: 400;
}}
QPushButton[nav="true"]:hover {{ background: #EEF4F0; color: {PALETTE.ink}; }}
QPushButton[nav="true"]:checked {{
    background: {PALETTE.accent}; color: white; font-weight: 700;
}}
QPushButton[nav="true"]:focus {{ border: 2px solid {PALETTE.focus}; }}
QFrame[card="true"] {{ background: white; border: 0; border-radius: 22px; }}
QFrame[cardVariant="quiet"] {{ background: #EAF2ED; }}
QFrame[cardVariant="accent"] {{ background: white; }}
QLabel[heading="page"] {{ font-size: 38px; font-weight: 700; }}
QLabel[heading="section"] {{ font-size: 20px; font-weight: 700; }}
QLabel[metric="value"] {{ font-size: 32px; font-weight: 700; color: {PALETTE.accent}; }}
QLabel[detail="value"] {{ font-size: 26px; font-weight: 400; color: {PALETTE.ink}; }}
QLabel[compact="true"] {{ font-size: 12px; }}
QLabel[kicker="true"], QLabel[eyebrow="true"] {{
    color: {PALETTE.muted}; font-size: 12px; font-weight: 700;
}}
QLabel[claim="boundary"] {{ color: {PALETTE.muted}; font-size: 12px; padding: 0; }}
QLabel[badge="verified"], QLabel[badge="locked"], QLabel[badge="neutral"] {{
    color: {PALETTE.accent}; background: #EAF3EF; border: 0;
    border-radius: 8px; padding: 5px 9px; font-weight: 700;
}}
QLabel[badge="neutral"] {{ color: {PALETTE.muted}; }}
QLabel[state="empty"], QLabel[state="ready"] {{
    color: {PALETTE.muted}; background: transparent; border: 0; padding: 2px 0;
}}
QLabel[state="loading"] {{
    color: {PALETTE.warning}; background: {PALETTE.loading_surface};
    border-radius: 8px; padding: 6px 10px;
}}
QLabel[state="error"] {{
    color: {PALETTE.danger}; background: #F9E8E2; border-radius: 8px; padding: 6px 10px;
}}
QPushButton {{
    background: white; border: 1px solid {PALETTE.boundary}; border-radius: 12px;
    min-height: 30px; padding: 6px 16px; font-weight: 700;
}}
QPushButton:hover {{ background: #EAF3EF; border-color: {PALETTE.accent}; }}
QPushButton:pressed {{ background: #D9E9E1; }}
QPushButton[primary="true"] {{ background: {PALETTE.accent}; color: white; border: 0; }}
QPushButton[primary="true"]:hover {{ background: #0F5C56; }}
QPushButton[quiet="true"] {{ background: transparent; border: 0; }}
QPushButton:disabled {{ color: #64736C; background: #E9EEEB; border-color: #D8E2DC; }}
QPushButton:focus, QComboBox:focus, QSpinBox:focus, QSlider:focus,
QGraphicsView:focus, QTableWidget:focus, QListWidget:focus {{
    border: 2px solid {PALETTE.focus};
}}
QComboBox, QSpinBox {{
    color: {PALETTE.ink}; background: white; border: 1px solid {PALETTE.boundary};
    border-radius: 10px; min-height: 32px; padding: 5px 10px;
}}
QComboBox:disabled, QSpinBox:disabled {{ color: #64736C; background: #E9EEEB; }}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QComboBox QAbstractItemView {{
    color: {PALETTE.ink}; background: white; selection-color: white;
    selection-background-color: {PALETTE.accent}; padding: 4px;
}}
QSlider::groove:horizontal {{ height: 5px; border-radius: 2px; background: #D9E4DF; }}
QSlider::sub-page:horizontal {{ background: {PALETTE.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 16px; margin: -6px 0; border-radius: 8px;
    background: {PALETTE.accent}; border: 0;
}}
QTableWidget {{
    background: transparent; alternate-background-color: transparent;
    border: 0; gridline-color: {PALETTE.boundary};
    selection-color: {PALETTE.ink}; selection-background-color: #E4F0EA;
}}
QHeaderView, QHeaderView::section {{
    color: {PALETTE.muted}; background: {PALETTE.canvas};
    border: 0; border-bottom: 1px solid {PALETTE.boundary}; padding: 8px 6px; font-weight: 400;
}}
QListWidget {{ background: white; border: 0; border-radius: 10px; }}
QListWidget[localityList="true"] {{ font-size: 13px; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 6px; }}
QListWidget::item:selected {{ color: {PALETTE.ink}; background: #E4F0EA; }}
QGraphicsView {{ background: transparent; border: 0; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {PALETTE.canvas}; border: 0; }}
QDialog {{ background: {PALETTE.canvas}; }}
QToolTip {{ color: white; background: {PALETTE.ink}; border: 0; padding: 7px; }}
QScrollBar:vertical {{ width: 10px; margin: 0; background: #E6EEEA; }}
QScrollBar::handle:vertical {{ background: #99B5A9; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
