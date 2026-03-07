"""WaveDL GUI - Wizard step header helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QFont
from qfluentwidgets import CaptionLabel, TitleLabel, setFont


if TYPE_CHECKING:
    from PySide6.QtWidgets import QVBoxLayout, QWidget


def add_step_header(
    layout: QVBoxLayout,
    title_text: str,
    subtitle_text: str,
    parent: QWidget,
    *,
    bottom_spacing: int = 24,
) -> tuple[TitleLabel, CaptionLabel]:
    """Add the standard wizard step header (title + subtitle + spacing).

    Returns the created TitleLabel and CaptionLabel for further customisation.
    """
    title = TitleLabel(title_text, parent)
    setFont(title, 26, QFont.Weight.Bold)
    layout.addWidget(title)

    layout.addSpacing(4)

    sub = CaptionLabel(subtitle_text, parent)
    sub.setTextColor(QColor(140, 140, 140), QColor(160, 160, 160))
    layout.addWidget(sub)

    layout.addSpacing(bottom_spacing)

    return title, sub
