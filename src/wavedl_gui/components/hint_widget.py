from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, IconWidget


class HintWidget(QWidget):
    """Hint widget with icon + text.

    Parameters
    ----------
    icon
        A FluentIcon or any icon accepted by ``IconWidget``.
    text : str
        The hint text to display.
    word_wrap : bool
        If ``True``, the text label wraps long lines (uses ``CaptionLabel``
        instead of ``BodyLabel``).
    """

    def __init__(self, icon, text, parent=None, *, word_wrap: bool = False):
        super().__init__(parent=parent)
        self.hBoxLayout = QHBoxLayout(self)

        self.iconWidget = IconWidget(icon)
        self.iconWidget.setFixedSize(
            16 if not word_wrap else 20, 16 if not word_wrap else 20
        )

        if word_wrap:
            self.label = CaptionLabel(text)
            self.label.setWordWrap(True)
        else:
            self.label = BodyLabel(text)

        self.hBoxLayout.setContentsMargins(24, 24, 24, 24)
        self.hBoxLayout.addWidget(self.iconWidget)
        self.hBoxLayout.addWidget(self.label, 1 if word_wrap else 0)
        self.hBoxLayout.setSpacing(10)
