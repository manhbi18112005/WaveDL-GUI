from AppKit import NSApp
from PySide6.QtCore import QObject


class MacSpeedBadge(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def setSpeed(self, speed: str):
        tile = NSApp().dockTile()
        tile.setBadgeLabel_(speed)
        tile.display()

    def hide(self):
        NSApp().dockTile().setBadgeLabel_(None)
