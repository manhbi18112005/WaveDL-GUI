"""
WaveDL GUI - Wizard Step 2: Data Input & Output Configuration

File/folder pickers using the shared FilePickerCard and FolderPickerCard
components, ensuring visual consistency with the Project Interface.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    FluentIcon as FIF,
    IndeterminateProgressRing,
    InfoBar,
    InfoBarPosition,
    SmoothScrollArea,
)

from ...common.config import cfg
from ...components.data_info_card import DataInfoCard
from ...components.data_plots_card import DataPlotsCard
from ...components.hint_widget import HintWidget
from ...components.shared import (
    FilePickerCard,
    FolderPickerCard,
    SettingCardGroup,
    add_step_header,
)
from ...service.data_parse_worker import DataParseWorker


if TYPE_CHECKING:
    from ...common.utils import DataInfo

# Constant for the file-filter string (same as ProjectInterface)
_DATA_FILE_FILTER = (
    "NPZ Files (*.npz);;MATLAB Files (*.mat);;HDF5 Files (*.h5 *.hdf5);;All Files (*)"
)


class DataStep(QWidget):
    """Step 2: data input/output configuration.

    Uses the shared FilePickerCard and FolderPickerCard for a polished,
    consistent look matching the Project Interface.
    """

    dataValidated = Signal(object)  # DataInfo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_info: DataInfo | None = None
        self._worker: DataParseWorker | None = None
        self._init_ui()

    @property
    def data_info(self) -> DataInfo | None:
        return self._data_info

    def _init_ui(self):
        # Outer layout just holds the scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = SmoothScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )

        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        root = QVBoxLayout(scroll_widget)
        root.setContentsMargins(60, 36, 60, 36)
        root.setSpacing(0)

        # Title
        add_step_header(
            root,
            "Configure Your Data",
            "Select your training data file and choose where to save results.",
            scroll_widget,
        )

        # ── Info tip ──
        info_box = HintWidget(
            FIF.INFO,
            "Supported formats: NPZ, MAT, HDF5. The file must contain "
            "input (X) and output (Y) arrays. Samples are along the first axis.",
            word_wrap=True,
        )
        root.addWidget(info_box)

        root.addSpacing(20)

        # ── Training Data card group ──
        self._data_group = SettingCardGroup("Training Data", scroll_widget)
        self._file_card = FilePickerCard(
            title="Training Data File",
            description="Select data file (.npz, .mat, or .h5) containing input/output arrays",
            file_filter=_DATA_FILE_FILTER,
            button_text="Select File",
            dialog_title="Select Training Data",
            parent=self._data_group,
        )
        self._data_group.addSettingCard(self._file_card)
        root.addWidget(self._data_group)

        # Connect file selection → parse
        self._file_card.clicked.disconnect(self._file_card._on_clicked)
        self._file_card.clicked.connect(self._select_file)

        root.addSpacing(12)

        # ── Output Directory card group ──
        self._output_group = SettingCardGroup("Output Configuration", scroll_widget)
        self._folder_card = FolderPickerCard(
            title="Output Directory",
            description=cfg.get(cfg.outputFolder)
            or "Choose where to save training results",
            default_dir=cfg.get(cfg.outputFolder) or "",
            dialog_title="Select Output Directory",
            parent=self._output_group,
        )
        self._output_group.addSettingCard(self._folder_card)
        root.addWidget(self._output_group)

        root.addSpacing(20)

        # ── Spinner ──
        self._spinner = IndeterminateProgressRing(scroll_widget)
        self._spinner.setFixedSize(36, 36)
        self._spinner.hide()
        root.addWidget(self._spinner, 0, Qt.AlignCenter)

        # ── Reuse the production DataInfoCard ──
        self._data_card = DataInfoCard(scroll_widget)
        root.addWidget(self._data_card)

        root.addSpacing(12)

        # ── Data plots card ──
        self._plots_card = DataPlotsCard(scroll_widget)
        root.addWidget(self._plots_card)

        root.addStretch()

        self._scroll.setWidget(scroll_widget)
        outer.addWidget(self._scroll)

    # ── slots ─────────────────────────────────────────────────────────────

    def _select_file(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Data",
            "",
            _DATA_FILE_FILTER,
        )
        if not path:
            return
        self._file_card.set_path(path)
        self._parse_file(path)

    def _parse_file(self, path: str):
        self._data_card.hide()
        self._spinner.show()
        self._data_info = None

        self._worker = DataParseWorker(path, self)
        self._worker.resultReady.connect(self._on_parse_success)
        self._worker.errorOccurred.connect(self._on_parse_error)
        self._worker.finished.connect(lambda: self._spinner.hide())
        self._worker.start()

    def _on_parse_success(self, info: DataInfo):
        self._data_info = info
        self._data_card.set_data_info(info)
        self._plots_card.set_data_path(info.path)
        self.dataValidated.emit(info)

    def _on_parse_error(self, error: str):
        self._data_info = None
        w = self.window()
        InfoBar.error(
            title="Data Error",
            content=error,
            parent=w if w else self,
            position=InfoBarPosition.TOP,
            duration=5000,
        )

    # ── public API ────────────────────────────────────────────────────────

    def get_data_path(self) -> str:
        return self._file_card.path

    def get_output_dir(self) -> str:
        return self._folder_card.folder

    def is_valid(self) -> bool:
        return (
            self._data_info is not None
            and not self._data_info.error
            and os.path.isfile(self.get_data_path())
        )
