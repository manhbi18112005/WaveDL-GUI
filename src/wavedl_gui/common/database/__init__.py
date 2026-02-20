from .db_initializer import DBInitializer
from .service import *
from collections import deque

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtSql import QSqlDatabase


class SqlRequest:
    """ Sql request """

    def __init__(self, service: str, method: str, slot=None, params: dict = None):
        self.service = service
        self.method = method
        self.slot = slot
        self.params = params or {}


class SqlResponse:
    """ Sql response """

    def __init__(self, data, slot):
        self.slot = slot
        self.data = data


class SqlSignalBus(QObject):
    """ Sql Signal bus """

    fetchDataSig = Signal(SqlRequest)
    dataFetched = Signal(SqlResponse)


sqlSignalBus = SqlSignalBus()


def sqlRequest(service: str, method: str, slot=None, **params):
    """ query sql from database """
    request = SqlRequest(service, method, slot, params)
    sqlSignalBus.fetchDataSig.emit(request)



class Database(QObject):
    """ Database """

    def __init__(self, db: QSqlDatabase = None, parent=None):
        """
        Parameters
        ----------
        directories: List[str]
            audio directories

        db: QDataBase
            database to be used

        watch: bool
            whether to monitor audio directories

        parent:
            parent instance
        """
        super().__init__(parent=parent)
        self.taskService = TaskService(db)

    def setDatabase(self, db: QSqlDatabase):
        """ set the database to be used """
        self.taskService.taskDao.setDatabase(db)



class DatabaseThread(QThread):
    """ Database thread """

    def __init__(self, db: QSqlDatabase = None, parent=None):
        """
        Parameters
        ----------
        directories: List[str]
            audio directories

        db: QDataBase
            database to be used

        watch: bool
            whether to monitor audio directories

        parent:
            parent instance
        """
        super().__init__(parent=parent)
        self.database = Database(db, self)
        self.tasks = deque()

        sqlSignalBus.fetchDataSig.connect(self.onFetchData)

    def run(self):
        while self.tasks:
            task, request = self.tasks.popleft()
            result = task(**request.params)
            sqlSignalBus.dataFetched.emit(SqlResponse(result, request.slot))

    def onFetchData(self, request: SqlRequest):
        service = getattr(self.database, request.service)
        task = getattr(service, request.method)
        self.tasks.append((task, request))

        if not self.isRunning():
            self.start()