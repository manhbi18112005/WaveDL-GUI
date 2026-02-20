# coding:utf-8
from ..logger import Logger
from ..setting import DB_PATH
from PySide6.QtSql import QSqlDatabase
from PySide6.QtWidgets import QApplication

from .service import TaskService


class DBInitializer:
    """ Database initializer """

    logger = Logger("database")
    CONNECTION_NAME = "main"
    CACHE_FILE = str(DB_PATH)

    @classmethod
    def init(cls):
        """ Initialize database """
        db = QSqlDatabase.addDatabase('QSQLITE', cls.CONNECTION_NAME)
        db.setDatabaseName(cls.CACHE_FILE)
        if not db.open():
            cls.logger.error("Database connection failed")
            QApplication.instance().exit()

        TaskService(db).createTable()