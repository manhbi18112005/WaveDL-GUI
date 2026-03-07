"""WaveDL GUI - FilePickerCard & FolderPickerCard."""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import FluentIcon as FIF, PushSettingCard


class FilePickerCard(PushSettingCard):
    """A PushSettingCard that opens a file dialog on click.

    Parameters
    ----------
    title : str
        Card title (e.g. "Training Data File").
    description : str
        Card subtitle / description.
    file_filter : str
        Qt file filter string, e.g.
        ``"NPZ Files (*.npz);;MATLAB Files (*.mat);;All Files (*)"``
    button_text : str
        Text on the push button (default ``"Select File"``).
    icon : FIF
        Fluent icon for the card (default ``FIF.DOCUMENT``).
    dialog_title : str
        Title of the file dialog (default ``"Select File"``).
    """

    def __init__(
        self,
        title: str,
        description: str,
        file_filter: str = "All Files (*)",
        *,
        button_text: str = "Select File",
        icon: FIF = FIF.DOCUMENT,
        dialog_title: str = "Select File",
        parent=None,
    ):
        super().__init__(button_text, icon, title, description, parent)
        self._file_filter = file_filter
        self._dialog_title = dialog_title
        self._path = ""
        self.clicked.connect(self._on_clicked)

    @property
    def path(self) -> str:
        return self._path

    def set_path(self, path: str):
        """Programmatically set the path and update the card content."""
        self._path = path
        self.setContent(path)

    def _on_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window(),
            self._dialog_title,
            "",
            self._file_filter,
        )
        if path:
            self.set_path(path)


class FolderPickerCard(PushSettingCard):
    """A PushSettingCard that opens a folder dialog on click.

    Parameters
    ----------
    title : str
        Card title (e.g. "Output Directory").
    description : str
        Card subtitle / description.
    default_dir : str
        Initial directory shown in the dialog.
    button_text : str
        Text on the push button (default ``"Choose"``).
    icon : FIF
        Fluent icon for the card (default ``FIF.FOLDER``).
    dialog_title : str
        Title of the folder dialog (default ``"Select Directory"``).
    """

    def __init__(
        self,
        title: str,
        description: str,
        default_dir: str = "",
        *,
        button_text: str = "Choose",
        icon: FIF = FIF.FOLDER,
        dialog_title: str = "Select Directory",
        parent=None,
    ):
        super().__init__(button_text, icon, title, description, parent)
        self._dialog_title = dialog_title
        self._folder = default_dir
        if default_dir:
            self.setContent(default_dir)
        self.clicked.connect(self._on_clicked)

    @property
    def folder(self) -> str:
        return self._folder

    def set_folder(self, folder: str):
        """Programmatically set the folder and update the card content."""
        self._folder = folder
        self.setContent(folder)

    def _on_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self.window(),
            self._dialog_title,
            self._folder,
        )
        if folder:
            self.set_folder(folder)
