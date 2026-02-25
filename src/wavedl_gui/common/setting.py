"""WaveDL GUI - Application Settings and Constants"""

import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths


# change DEBUG to False if you want to compile the code to exe
DEBUG = "__compiled__" not in globals()


YEAR = 2025
AUTHOR = "Ductho Le"
VERSION = "1.0.0"
APP_NAME = "WaveDL"
REPO_URL = "https://github.com/ductho-le/WaveDL"
FEEDBACK_URL = "https://github.com/ductho-le/WaveDL/issues"
RELEASE_URL = "https://github.com/ductho-le/WaveDL/releases"
KOFI_URL = "https://ko-fi.com/wavedl"

CONFIG_FOLDER = Path("AppData").absolute()

if sys.platform == "win32" and not DEBUG:
    CONFIG_FOLDER = (
        Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / APP_NAME
    )


CONFIG_FILE = CONFIG_FOLDER / "config.json"
DB_PATH = CONFIG_FOLDER / "database.db"
