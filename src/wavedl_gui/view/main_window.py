"""
WaveDL GUI - Main Window
Uses FluentWindow from qfluentwidgets for a modern fluent design with left sidebar navigation.
"""

import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon
from PySide6.QtSql import QSqlDatabase
from PySide6.QtWidgets import QApplication, QFileDialog
from qfluentwidgets import (
    FluentIcon as FIF,
    FluentWindow,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    NavigationItemPosition,
    SplashScreen,
)

from ..common.concurrent import TaskExecutor
from ..common.config import cfg
from ..common.database import DatabaseThread, DBInitializer, SqlResponse, sqlSignalBus
from ..common.setting import FEEDBACK_URL, KOFI_URL, RELEASE_URL
from ..common.signal_bus import signalBus
from ..common.utils import check_pytorch_installation, get_gpu_summary, openUrl
from ..components.menu_bar import MenuBar
from ..components.system_tray_icon import SystemTrayIcon
from ..service.training_service import trainingService
from ..service.version_service import VersionService
from .dashboard_interface import DashboardInterface
from .project_interface import ProjectInterface
from .setting_interface import SettingInterface
from .training_interface import TrainingInterface


class MainWindow(FluentWindow):
    """Main application window for WaveDL GUI.

    Uses FluentWindow for consistent qfluentwidgets styling with left sidebar navigation.
    Provides navigation between interfaces and orchestrates the training workflow.
    """

    def __init__(self):
        super().__init__()

        self.initDatabase()
        self.initWindow()

        self.versionManager = VersionService()

        # Initialize interfaces
        self.projectInterface = ProjectInterface(self)
        self.trainingInterface = TrainingInterface(self)
        self.dashboardInterface = DashboardInterface(self)
        self.settingInterface = SettingInterface(self)
        self.systemTrayIcon = SystemTrayIcon(self)

        self.connectSignalToSlot()

        # add items to navigation interface
        self.initMenuBar()
        self.initNavigation()

        # check for updates and environment
        self.onInitFinished()

    def connectSignalToSlot(self):
        """Connect signals to slots."""
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        signalBus.appErrorSig.connect(self.onAppError)
        signalBus.appMessageSig.connect(self.onAppMessage)
        signalBus.checkUpdateSig.connect(self.checkUpdate)

        # Navigation signals
        signalBus.switchToProjectSig.connect(
            lambda: self.switchTo(self.projectInterface)
        )
        signalBus.switchToTrainingSig.connect(
            lambda: self.switchTo(self.trainingInterface)
        )
        signalBus.switchToDashboardSig.connect(
            lambda: self.switchTo(self.dashboardInterface)
        )

        # Dashboard "Start Training" button triggers validation + start
        signalBus.requestStartTrainingSig.connect(self.onStartTraining)

        self.systemTrayIcon.messageClicked.connect(self.onSystemTrayMessageClicked)

    def initNavigation(self):
        """Initialize the navigation sidebar with training workflow interfaces."""
        # Main workflow interfaces
        self.addSubInterface(self.projectInterface, FIF.FOLDER, self.tr("Project"))
        self.addSubInterface(self.trainingInterface, FIF.SETTING, self.tr("Training"))
        self.addSubInterface(self.dashboardInterface, FIF.VIEW, self.tr("Dashboard"))

        # Action buttons
        self.navigationInterface.addItem(
            "start_training",
            FIF.PLAY,
            self.tr("Start Training"),
            self.onStartTraining,
            position=NavigationItemPosition.BOTTOM,
        )
        self.navigationInterface.addItem(
            "generate_cmd",
            FIF.COMMAND_PROMPT,
            self.tr("Generate Command"),
            self.onGenerateCommand,
            position=NavigationItemPosition.BOTTOM,
        )

        self.addSubInterface(
            self.settingInterface,
            FIF.SETTING,
            self.tr("Settings"),
            position=NavigationItemPosition.BOTTOM,
        )

    def initWindow(self):
        """Initialize window properties."""
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        self.setWindowIcon(QIcon(":/src/wavedl_gui/resource/images/logo.png"))
        self.setWindowTitle("WaveDL - Deep Learning Training GUI")
        QApplication.setQuitOnLastWindowClosed(False)

        self.setCustomBackgroundColor(QColor(240, 244, 249), QColor(32, 32, 32))
        self.setMicaEffectEnabled(False)

        # create splash screen
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.primaryScreen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()
        QApplication.processEvents()

    def initDatabase(self):
        """Initialize database."""
        DBInitializer.init()

        self.databaseThread = DatabaseThread(
            QSqlDatabase.database(DBInitializer.CONNECTION_NAME), self
        )

        sqlSignalBus.dataFetched.connect(self.onDataFetched)

    def initMenuBar(self):
        """Initialize macOS menu bar."""
        if sys.platform != "darwin":
            return

        self.menuBar = MenuBar(self)
        self.menuBar.openFileAct.triggered.connect(self.openFile)
        self.menuBar.closeWindowAct.triggered.connect(self.close)
        self.menuBar.donateAct.triggered.connect(self.support)
        self.menuBar.settingsAct.triggered.connect(
            lambda: self.switchTo(self.settingInterface)
        )
        self.menuBar.feedbackAct.triggered.connect(lambda: openUrl(FEEDBACK_URL))

    def onDataFetched(self, response: SqlResponse):
        """Handle database response."""
        if response.slot:
            response.slot(response.data)

    def onAppError(self, message: str):
        """Handle application error."""
        QApplication.clipboard().setText(message)
        self.showMessageBox(
            self.tr("Unhandled exception occurred"),
            self.tr(
                "The error message has been written to the paste board and log. Do you want to report?"
            ),
            True,
            lambda: openUrl(FEEDBACK_URL),
        )

    def onAppMessage(self, message: str):
        """Handle application message from another instance."""
        if message == "show":
            if self.windowState() & Qt.WindowMinimized:
                self.showNormal()
            else:
                self.show()
                self.raise_()
        else:
            self.show()

    def showMessageBox(
        self, title: str, content: str, showYesButton=False, yesSlot=None
    ):
        """Show a message box."""
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr("Close"))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def checkUpdate(self, ignore=False):
        """Check for software updates."""
        TaskExecutor.runTask(self.versionManager.hasNewVersion).then(
            lambda success: self.onVersionInfoFetched(success, ignore)
        )

    def onVersionInfoFetched(self, success, ignore=False):
        """Handle version check result."""
        if success:
            self.showMessageBox(
                self.tr("Updates available"),
                self.tr("A new version")
                + f" {self.versionManager.lastestVersion} "
                + self.tr("is available. Do you want to download this version?"),
                True,
                lambda: openUrl(RELEASE_URL),
            )
        elif not ignore:
            self.showMessageBox(
                self.tr("No updates available"),
                self.tr("WaveDL has been updated to the latest version."),
            )

    def onSystemTrayMessageClicked(self):
        """Handle system tray message clicked."""
        self.switchTo(self.dashboardInterface)
        self.show()
        self.raise_()

    def onStartTraining(self):
        """Start training with current configuration."""
        # Validate data path
        data_path = self.projectInterface.data_path
        output_dir = self.projectInterface.output_dir

        if not data_path:
            InfoBar.warning(
                title=self.tr("Missing Data"),
                content=self.tr("Please select a data file first"),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            self.switchTo(self.projectInterface)
            return

        # Get training config
        config = self.trainingInterface.get_training_config(data_path, output_dir)

        # Switch to dashboard and start training
        self.switchTo(self.dashboardInterface)
        trainingService.start_training(config)

    def onGenerateCommand(self):
        """Generate CLI command from current configuration."""
        data_path = self.projectInterface.data_path or "<DATA_PATH>"
        output_dir = self.projectInterface.output_dir or "<OUTPUT_DIR>"

        config = self.trainingInterface.get_training_config(data_path, output_dir)
        command = config.to_command()

        # Copy to clipboard
        QApplication.clipboard().setText(command)

        InfoBar.success(
            title=self.tr("Command Copied"),
            content=self.tr("CLI command copied to clipboard"),
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000,
        )

    def resizeEvent(self, e):
        """Handle resize event."""
        super().resizeEvent(e)
        if hasattr(self, "splashScreen"):
            self.splashScreen.resize(self.size())

    def closeEvent(self, event):
        """Handle close event - minimize to tray."""
        event.ignore()
        self.hide()

    def onInitFinished(self):
        """Called when initialization is complete."""
        self.splashScreen.finish()
        self.systemTrayIcon.show()

        # Check environment
        self._check_environment()

        if cfg.get(cfg.checkUpdateAtStartUp):
            self.checkUpdate(True)

    def _check_environment(self):
        """Check PyTorch and GPU availability."""
        installed, version, _cuda = check_pytorch_installation()

        if not installed:
            InfoBar.warning(
                title=self.tr("PyTorch Not Found"),
                content=self.tr("PyTorch is not installed. Training will not work."),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            gpu_summary = get_gpu_summary()
            InfoBar.info(
                title=f"PyTorch {version}",
                content=gpu_summary,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=4000,
            )

    def support(self):
        """Open donation page."""
        openUrl(KOFI_URL)

    def openFile(self):
        """Open a data file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open Data File"),
            cfg.get(cfg.outputFolder),
            "NPZ Files (*.npz);;MAT Files (*.mat);;HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if path:
            self.projectInterface.set_data_path(path)
            self.switchTo(self.projectInterface)

    def onExit(self):
        """Exit main window."""
        self.systemTrayIcon.hide()

        # Stop any running training
        trainingService.stop_training()
        trainingService.stop_test()

        # close database
        QSqlDatabase.database(DBInitializer.CONNECTION_NAME).close()
        QSqlDatabase.removeDatabase(DBInitializer.CONNECTION_NAME)
