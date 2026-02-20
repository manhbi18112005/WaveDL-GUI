# coding: utf-8
from PySide6.QtCore import QObject
from ..common.database.entity import Task


class SpeedService(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.speedMap = {}

    def update(self, task: Task, speed: str):
        if not speed:
            return

        self.speedMap[task.id] = speed

    def totalSpeed(self) -> str:
        total = 0

        for tid, speed in self.speedMap.items():
            speed = speed.strip().upper().replace(' ', '')
            if speed.endswith('KB/S'):
                total += float(speed[:-4]) / 1024
            elif speed.endswith('MB/S'):
                total += float(speed[:-4])
            elif speed.endswith('GB/S'):
                total += float(speed[:-4]) * 1024

        return f"{total:.2f} MB/s"


speedService = SpeedService()