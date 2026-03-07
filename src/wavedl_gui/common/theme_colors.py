"""
WaveDL GUI - Shared Theme Color Palette

Centralised, theme-aware colour helpers used across cards, widgets,
and wizard steps.  Every function returns the appropriate QColor for
the current light / dark theme.
"""

from PySide6.QtGui import QColor
from qfluentwidgets import isDarkTheme


def accent_color() -> QColor:
    """Primary accent (blue)."""
    return QColor("#3b82f6") if not isDarkTheme() else QColor("#60a5fa")


def success_color() -> QColor:
    return QColor("#16a34a") if not isDarkTheme() else QColor("#4ade80")


def error_color() -> QColor:
    return QColor("#dc2626") if not isDarkTheme() else QColor("#f87171")


def muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def subtle_border_color() -> QColor:
    return QColor(0, 0, 0, 18) if not isDarkTheme() else QColor(255, 255, 255, 18)


def section_bg_color() -> QColor:
    return QColor(0, 0, 0, 6) if not isDarkTheme() else QColor(255, 255, 255, 6)


def tag_bg_color() -> QColor:
    return QColor(0, 0, 0, 12) if not isDarkTheme() else QColor(255, 255, 255, 12)
