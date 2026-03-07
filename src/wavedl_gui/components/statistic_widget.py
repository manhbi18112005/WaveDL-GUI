"""
WaveDL GUI - Statistics Widget Component

A polished stat block widget with big value, caption, and optional subtitle.
Uses theme-aware colors consistent with the DataInfoCard design system.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, StrongBodyLabel, setFont

from ..common.theme_colors import (
    muted_text_color as _muted_text_color,
    section_bg_color as _section_bg_color,
)


# ─── Widget ───────────────────────────────────────────────────────────────────


class StatisticsWidget(QWidget):
    """A small vertical stat block: big number + caption beneath + optional subtitle."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(2)
        v.setAlignment(Qt.AlignCenter)

        self.valueLabel = StrongBodyLabel("—", self)
        setFont(self.valueLabel, 18, QFont.Weight.DemiBold)
        self.valueLabel.setAlignment(Qt.AlignCenter)

        self.captionLabel = CaptionLabel(caption, self)
        self.captionLabel.setTextColor(_muted_text_color(), _muted_text_color())
        self.captionLabel.setAlignment(Qt.AlignCenter)

        self.subtitleLabel = CaptionLabel("", self)
        self.subtitleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        self.subtitleLabel.setAlignment(Qt.AlignCenter)
        setFont(self.subtitleLabel, 10)
        self.subtitleLabel.hide()

        v.addWidget(self.valueLabel)
        v.addWidget(self.captionLabel)
        v.addWidget(self.subtitleLabel)

    def setValue(self, v: str, subtitle: str = ""):
        """Update the displayed value and optional subtitle."""
        self.valueLabel.setText(v)
        if subtitle:
            self.subtitleLabel.setText(subtitle)
            self.subtitleLabel.show()
        else:
            self.subtitleLabel.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(_section_bg_color())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 8, 8)
        p.end()
