"""WaveDL GUI - PropertyRow widget."""

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, setFont

from ...common.theme_colors import muted_text_color


class PropertyRow(QWidget):
    """Label → value row with optional monospaced value.

    Parameters
    ----------
    label : str
        Caption text on the left.
    mono : bool
        If ``True``, use a monospaced font for the value.
    label_width : int
        Fixed pixel width for the label column (default 100).
    """

    def __init__(
        self,
        label: str,
        mono: bool = False,
        label_width: int = 100,
        parent=None,
    ):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        self.label = CaptionLabel(label, self)
        self.label.setTextColor(muted_text_color(), muted_text_color())
        self.label.setFixedWidth(label_width)

        self.value = BodyLabel("", self)
        if mono:
            setFont(self.value, 12, QFont.Weight.Normal)
            self.value.setFont(QFont("Cascadia Code, Consolas, Monaco, monospace", 12))
        else:
            setFont(self.value, 12, QFont.Weight.Normal)

        h.addWidget(self.label)
        h.addWidget(self.value, 1)

    def setValue(self, v: str):
        self.value.setText(v)
