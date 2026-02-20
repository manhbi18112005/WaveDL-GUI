# coding:utf-8
from PySide6.QtCore import Qt, Signal, Property
from PySide6.QtGui import QPixmap, QPainter, QColor, QAction, QKeySequence
from PySide6.QtWidgets import QWidget, QHBoxLayout, QMenuBar


class MenuBar(QMenuBar):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # file menu
        self.fileMenu = self.addMenu(self.tr("File")+"(&R)")
        self.openFileAct = QAction(self.tr("Open File"), shortcut="Ctrl+O", parent=self)
        self.settingsAct = QAction(self.tr("Preferences"), shortcuts=QKeySequence.StandardKey.Preferences, parent=self)
        self.closeWindowAct = QAction(self.tr("Close Window"), shortcut="Ctrl+W", parent=self)
        self.fileMenu.addActions([self.openFileAct, self.settingsAct])
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.closeWindowAct)

        # window menu
        self.helpMenu = self.addMenu(self.tr("Help")+"(&H)")
        self.docAct = QAction(self.tr("Documention"), shortcuts=QKeySequence.StandardKey.HelpContents, parent=self)
        self.videoTutorialAct = QAction(self.tr("Video Tutorials"), parent=self)
        self.feedbackAct = QAction(self.tr("Feedback"), parent=self)
        self.donateAct = QAction(self.tr("Support Us"), parent=self)
        self.helpMenu.addActions([self.docAct, self.videoTutorialAct, self.feedbackAct])
        self.helpMenu.addSeparator()
        self.helpMenu.addAction(self.donateAct)