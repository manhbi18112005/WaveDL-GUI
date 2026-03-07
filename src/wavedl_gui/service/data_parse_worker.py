"""
WaveDL GUI - Data Parse Worker

Background QThread that inspects a data file without blocking the GUI.
Emits structured DataInfo on success or an error string on failure.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..common.utils import DataInfo, inspect_data_file


class DataParseWorker(QThread):
    """Worker that calls inspect_data_file() on a background thread."""

    resultReady = Signal(object)  # DataInfo
    errorOccurred = Signal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self._file_path = file_path

    def run(self):
        try:
            info: DataInfo = inspect_data_file(self._file_path)
            if info.error:
                self.errorOccurred.emit(info.error)
            else:
                self.resultReady.emit(info)
        except Exception as e:
            self.errorOccurred.emit(str(e))
