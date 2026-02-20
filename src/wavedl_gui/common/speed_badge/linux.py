# coding: utf-8
from PySide6.QtCore import QObject


class LinuxSpeedBadge(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)

    def setSpeed(self, speed: str):
        pass

    def hide(self):
        pass
