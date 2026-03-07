"""WaveDL GUI - Separator widget."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QFrame

from ...common.theme_colors import subtle_border_color


class Separator(QFrame):
    """A thin horizontal line separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFixedHeight(1)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        p.setBrush(subtle_border_color())
        p.drawRect(self.rect())
        p.end()
