"""WaveDL GUI - SectionHeader widget."""

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, FluentIcon as FIF, IconWidget, setFont

from ...common.theme_colors import muted_text_color


class SectionHeader(QWidget):
    """Section header with an icon and uppercase title."""

    def __init__(self, icon: FIF, title: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(8)

        ic = IconWidget(icon, self)
        ic.setFixedSize(16, 16)
        h.addWidget(ic)

        lbl = CaptionLabel(title.upper(), self)
        lbl.setTextColor(muted_text_color(), muted_text_color())
        setFont(lbl, 10, QFont.Weight.Bold)
        h.addWidget(lbl)
        h.addStretch()
