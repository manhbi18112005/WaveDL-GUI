# coding:utf-8
import os
import sys
from inspect import getsourcefile
from pathlib import Path

os.chdir(Path(getsourcefile(lambda: 0)).resolve().parent)

from PySide6.QtCore import Qt, QTranslator
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentTranslator

from wavedl_gui.common.config import cfg
from wavedl_gui.common.application import SingletonApplication
from wavedl_gui.view.main_window import MainWindow


# enable dpi scale
if cfg.get(cfg.dpiScale) != "Auto":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))

# create application
app = SingletonApplication(sys.argv, "WaveDL-GUI")
app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

if sys.platform == "darwin":
    from AppKit import NSApplication
    NSApplication.sharedApplication()

# internationalization
locale = cfg.get(cfg.language).value
translator = FluentTranslator(locale)
galleryTranslator = QTranslator()
galleryTranslator.load(locale, "src/wavedl_gui", ".", ":/src/wavedl_gui/i18n")

app.installTranslator(translator)
app.installTranslator(galleryTranslator)

# create main window
w = MainWindow()
app.aboutToQuit.connect(w.onExit)
w.show()

app.exec()