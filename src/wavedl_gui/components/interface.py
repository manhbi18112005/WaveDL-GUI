# coding:utf-8
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from qfluentwidgets import ScrollArea, FlowLayout, setFont, TitleLabel


class Interface(ScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.view = QWidget(self)
        self.titleLabel = TitleLabel(self.view)

        self.vBoxLayout = QVBoxLayout(self)
        self.viewLayout = QVBoxLayout(self.view)

        self.__initWidgets()

    def __initWidgets(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(30, 33, 30, 10)
        self.viewLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.setSpacing(20)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        setFont(self.titleLabel, 23, QFont.Weight.DemiBold)
        self.enableTransparentBackground()

    def setTitle(self, title: str):
        self.titleLabel.setText(title)
        self.setObjectName(title)
