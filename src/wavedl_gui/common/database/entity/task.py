# coding:utf-8
from enum import Enum
from pathlib import Path
from typing import List
from .entity import Entity
from dataclasses import dataclass, field

from PySide6.QtCore import QDateTime, QFileInfo

from ..utils.uuid_utils import UUIDUtils


class TaskStatus:

    RUNNING = 0
    SUCCESS = 1
    FAILED = 2


@dataclass
class Task(Entity):

    id: str = field(default_factory=UUIDUtils.getUUID)
    url: str = None                     # 下载链接
    isLive: bool = False                # 是否为直播
    pid: int = None                     # 进程 id
    fileName: str = None                # 文件名
    saveFolder: str = None              # 保存文件夹
    size: str = '0MB'                   # 文件大小
    isBinaryMerge : bool = False        # 是否合并为 ts 文件
    isLiveRealTimeMerge: bool = False   # 录制直播时是否实时合并为 ts 文件
    command: str = None                 # 下载命令
    status: int = 0                     # 状态，0 为下载中，1 为下载完成，2 为下载失败
    logFile: str = None                 # 日志文件路径
    createTime: QDateTime = field(default_factory=QDateTime.currentDateTime)

    def isRunning(self):
        return self.status == TaskStatus.RUNNING

    def error(self):
        self.status = TaskStatus.FAILED

    def success(self):
        self.status = TaskStatus.SUCCESS