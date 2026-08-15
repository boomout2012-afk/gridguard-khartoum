"""Presentation-focused visual system shared by the native desktop UI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic colors with high-contrast presentation defaults."""

    canvas: str = "#081723"
    surface: str = "#102535"
    surface_raised: str = "#183246"
    border: str = "#365064"
    text: str = "#F4F7F8"
    text_muted: str = "#B9C7D0"
    accent: str = "#D8B55B"
    accent_dark: str = "#8A6B24"
    river: str = "#54A8C7"
    success: str = "#52B69A"
    danger: str = "#D66B72"
    warning: str = "#E2A84A"
    focus: str = "#F4D06F"


PALETTE = Palette()

STATUS_COLORS = {
    0: QColor("#873F48"),
    1: QColor("#B77A25"),
    2: QColor("#287D72"),
}
STATUS_TEXT = {
    0: "× OFF",
    1: "◆ CRITICAL ONLY",
    2: "● FULL",
}


def apply_theme(application: QApplication) -> None:
    """Apply the single offline visual system to a Qt application."""

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    application.setFont(font)
    application.setStyleSheet(_STYLE_SHEET)


_STYLE_SHEET = f"""
QWidget {{
    color: {PALETTE.text};
    background: {PALETTE.canvas};
    font-size: 14px;
}}
QLabel {{
    background: transparent;
}}
QMainWindow {{
    background: {PALETTE.canvas};
}}
QFrame#appHeader, QFrame[card="true"] {{
    background: {PALETTE.surface};
    border: 1px solid {PALETTE.border};
    border-radius: 8px;
}}
QLabel#brandTitle {{
    color: {PALETTE.text};
    font-size: 25px;
    font-weight: 700;
}}
QLabel#brandSubtitle, QLabel[muted="true"] {{
    color: {PALETTE.text_muted};
}}
QLabel[compact="true"] {{
    font-size: 12px;
}}
QLabel[heading="page"] {{
    color: {PALETTE.text};
    font-size: 23px;
    font-weight: 700;
}}
QLabel[heading="section"] {{
    color: {PALETTE.text};
    font-size: 17px;
    font-weight: 650;
}}
QLabel[metric="value"] {{
    color: {PALETTE.accent};
    font-size: 25px;
    font-weight: 750;
}}
QLabel[detail="value"] {{
    color: {PALETTE.accent};
    font-size: 17px;
    font-weight: 700;
}}
QLabel[badge="verified"] {{
    color: {PALETTE.success};
    background: #153B38;
    border: 1px solid #2C6B61;
    border-radius: 7px;
    padding: 5px 9px;
    font-weight: 650;
}}
QTabWidget::pane {{
    border: 1px solid {PALETTE.border};
    border-radius: 8px;
    top: -1px;
    background: {PALETTE.canvas};
}}
QTabBar::tab {{
    color: {PALETTE.text_muted};
    background: {PALETTE.surface};
    border: 1px solid {PALETTE.border};
    padding: 10px 20px;
    min-width: 120px;
}}
QTabBar::tab:selected {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    border-bottom: 3px solid {PALETTE.accent};
    font-weight: 650;
}}
QTabBar::tab:focus, QPushButton:focus, QComboBox:focus, QSpinBox:focus,
QSlider:focus, QGraphicsView:focus {{
    border: 2px solid {PALETTE.focus};
}}
QPushButton {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    border: 1px solid {PALETTE.border};
    border-radius: 6px;
    min-height: 34px;
    padding: 4px 12px;
    font-weight: 600;
}}
QPushButton:hover {{
    border-color: {PALETTE.accent};
}}
QPushButton:pressed {{
    background: {PALETTE.accent_dark};
}}
QPushButton[primary="true"] {{
    color: #101A20;
    background: {PALETTE.accent};
    border-color: {PALETTE.accent};
}}
QComboBox, QSpinBox {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    border: 1px solid {PALETTE.border};
    border-radius: 6px;
    min-height: 34px;
    padding: 2px 9px;
}}
QComboBox QAbstractItemView {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    selection-background-color: {PALETTE.accent_dark};
}}
QTableWidget {{
    color: {PALETTE.text};
    background: {PALETTE.surface};
    alternate-background-color: {PALETTE.surface_raised};
    border: 1px solid {PALETTE.border};
    gridline-color: {PALETTE.border};
    selection-background-color: {PALETTE.accent_dark};
}}
QHeaderView::section {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    border: 0;
    border-bottom: 1px solid {PALETTE.border};
    padding: 8px;
    font-weight: 650;
}}
QToolTip {{
    color: {PALETTE.text};
    background: {PALETTE.surface_raised};
    border: 1px solid {PALETTE.accent};
    padding: 6px;
}}
QScrollBar:vertical {{
    width: 12px;
    background: {PALETTE.surface};
}}
QScrollBar::handle:vertical {{
    background: {PALETTE.border};
    border-radius: 5px;
    min-height: 28px;
}}
"""
