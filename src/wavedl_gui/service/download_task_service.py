# coding:utf-8
from pathlib import Path
import shutil
from PySide6.QtCore import Qt, Signal, Property, QObject
from PySide6.QtGui import QPixmap, QPainter, QColor
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout

from ..common.database import sqlRequest
from ..common.database.entity import Task
from ..common.utils import removeFile, showInFolder


class DownloadTaskService(QObject):
    """ Download task service """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def showInFolder(self, task: Task):
        """ Show downloaded file in folder """
        return showInFolder(task.availableVideoPath())

    def removeDownloadingTask(self, task: Task, deleteFile=True):
        if deleteFile:
            self._removeTmpFolder(task)

    def removedSuccessTask(self, task: Task, deleteFile=True):
        """ remove success task """
        sqlRequest("taskService", "removeById", id=task.id)

        if deleteFile:
            removeFile(task.availableVideoPath())

    def removeFailedTask(self, task: Task, deleteFile=True):
        """ remove failed task """
        sqlRequest("taskService", "removeById", id=task.id)

        if deleteFile:
            self._removeTmpFolder(task)

    def _removeTmpFolder(self, task: Task):
        folder = task.videoPath.parent / task.fileName
        shutil.rmtree(folder, ignore_errors=True)


downloadTaskService = DownloadTaskService()