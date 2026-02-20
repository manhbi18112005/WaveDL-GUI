# coding:utf-8
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QWidget, QSystemTrayIcon, QHBoxLayout, QLabel

from qfluentwidgets import Action, SystemTrayMenu, MessageBox, setTheme, Theme

from ..common.signal_bus import signalBus


class SystemTrayIcon(QSystemTrayIcon):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setIcon(parent.windowIcon())

        self.menu = SystemTrayMenu(parent=parent)
        self.menu.addActions([
            Action(
                self.tr('Show panel'),
                triggered=lambda: signalBus.appMessageSig.emit("show")
            ),
            Action(
                self.tr('Exit'),
                triggered=QApplication.instance().exit
            ),
        ])
        self.setContextMenu(self.menu)

