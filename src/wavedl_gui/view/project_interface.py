"""
WaveDL GUI - Project Setup Interface

Interface for selecting data files, configuring output directories,
viewing system information, and launching training.
"""

from __future__ import annotations

import os
import subprocess
import sys

import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFileDialog, QWidget
from qfluentwidgets import (
    ExpandLayout,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    PushSettingCard,
    ScrollArea,
    SettingCardGroup as CardGroup,
    TitleLabel,
    setFont,
)

from ..common.config import cfg
from ..common.constants.index import TrainingConfig
from ..common.signal_bus import signalBus
from ..common.utils import DataInfo, inspect_data_file
from ..components.controls_card import ControlsCard
from ..components.data_info_card import DataInfoCard
from ..components.system_info_card import SystemInfoCard
from ..service.training_service import ProcessState


class SettingCardGroup(CardGroup):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        setFont(self.titleLabel, 14, QFont.Weight.DemiBold)


class ProjectInterface(ScrollArea):
    """Project setup interface for data and output configuration."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        self._data_path = ""
        self._output_dir = cfg.get(cfg.outputFolder)
        self.settingLabel = TitleLabel(self.tr("Project Setup"), self)

        self.dataGroup = SettingCardGroup(self.tr("Training Data"), self.scrollWidget)
        self.dataCard = PushSettingCard(
            self.tr("Select File"),
            FIF.DOCUMENT,
            self.tr("Training Data File"),
            self.tr(
                "Select data file (.npz, .mat, or .h5) containing input/output arrays"
            ),
            self.dataGroup,
        )

        self.dataInfoCard = DataInfoCard(self.scrollWidget)

        self.outputGroup = SettingCardGroup(
            self.tr("Output Configuration"), self.scrollWidget
        )
        self.outputCard = PushSettingCard(
            self.tr("Choose"),
            FIF.FOLDER,
            self.tr("Output Directory"),
            cfg.get(cfg.outputFolder),
            self.outputGroup,
        )

        self.systemInfoCard = SystemInfoCard(self.scrollWidget)
        self.controlsCard = ControlsCard(self.scrollWidget)

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 90, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName("projectInterface")

        setFont(self.settingLabel, 23, QFont.Weight.DemiBold)
        self.enableTransparentBackground()

        self.__initLayout()
        self._connect_signals()

    def __initLayout(self):
        self.settingLabel.move(36, 40)
        self.dataGroup.addSettingCard(self.dataCard)
        self.outputGroup.addSettingCard(self.outputCard)

        self.expandLayout.setSpacing(26)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.systemInfoCard)
        self.expandLayout.addWidget(self.dataGroup)
        self.expandLayout.addWidget(self.dataInfoCard)
        self.expandLayout.addWidget(self.outputGroup)
        self.expandLayout.addWidget(self.controlsCard)

    def _connect_signals(self):
        self.dataCard.clicked.connect(self._select_data_file)
        self.outputCard.clicked.connect(self._select_output_dir)

        self.controlsCard.openOutputClicked.connect(self._open_output_folder)
        self.controlsCard.openConfigClicked.connect(self._open_config_file)
        self.controlsCard.resetSettingsClicked.connect(self._reset_settings)
        self.controlsCard.exportConfigClicked.connect(self._export_config)

        signalBus.trainingStateChangedSig.connect(self._on_state_changed)

    def _select_data_file(self):
        """Open file dialog to select data file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Training Data"),
            "",
            "NPZ Files (*.npz);;MATLAB Files (*.mat);;HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if path:
            self.set_data_path(path)

    def _select_output_dir(self):
        """Open folder dialog to select output directory."""
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Select Output Directory"), self._output_dir
        )
        if folder:
            self.set_output_dir(folder)

    def set_data_path(self, path: str):
        self._data_path = path
        self.dataCard.setContent(path)

        info = inspect_data_file(path)
        self.dataInfoCard.set_data_info(info)

        if info.error:
            InfoBar.error(
                title=self.tr("Error"),
                content=info.error,
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            InfoBar.success(
                title=self.tr("Data loaded"),
                content=f"{info.num_samples:,} samples, {info.dimensionality} data",
                parent=self.window(),
                position=InfoBarPosition.TOP,
                duration=3000,
            )

    def set_output_dir(self, folder: str):
        self._output_dir = folder
        self.outputCard.setContent(folder)
        cfg.set(cfg.outputFolder, folder)

    def _open_output_folder(self):
        folder = self._output_dir
        if not folder or not os.path.isdir(folder):
            InfoBar.warning(
                title=self.tr("No Output Folder"),
                content=self.tr("Please select an output directory first."),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        elif sys.platform == "win32":
            os.startfile(folder)
        else:
            subprocess.Popen(["xdg-open", folder])

    def _open_config_file(self):
        from ..common.setting import CONFIG_FILE

        path = str(CONFIG_FILE.absolute())
        if not os.path.isfile(path):
            InfoBar.warning(
                title=self.tr("Config Not Found"),
                content=self.tr("No config file exists yet."),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000,
            )
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])

    def _reset_settings(self):
        dialog = MessageBox(
            self.tr("Reset Settings"),
            self.tr(
                "This will reset all training parameters to their default values. "
                "Data file and output directory will not be changed.\n\n"
                "Continue?"
            ),
            self.window(),
        )
        if not dialog.exec():
            return

        cfg.set(cfg.batchSize, cfg.batchSize.defaultValue)
        cfg.set(cfg.learningRate, cfg.learningRate.defaultValue)
        cfg.set(cfg.epochs, cfg.epochs.defaultValue)
        cfg.set(cfg.patience, cfg.patience.defaultValue)
        cfg.set(cfg.weightDecay, cfg.weightDecay.defaultValue)
        cfg.set(cfg.gradClip, cfg.gradClip.defaultValue)
        cfg.set(cfg.seed, cfg.seed.defaultValue)
        cfg.set(cfg.model, cfg.model.defaultValue)
        cfg.set(cfg.pretrained, cfg.pretrained.defaultValue)
        cfg.set(cfg.loss, cfg.loss.defaultValue)
        cfg.set(cfg.optimizer, cfg.optimizer.defaultValue)
        cfg.set(cfg.scheduler, cfg.scheduler.defaultValue)
        cfg.set(cfg.precision, cfg.precision.defaultValue)
        cfg.set(cfg.compile, cfg.compile.defaultValue)
        cfg.set(cfg.deterministic, cfg.deterministic.defaultValue)
        cfg.set(cfg.noCache, cfg.noCache.defaultValue)
        cfg.set(cfg.cv, cfg.cv.defaultValue)
        cfg.set(cfg.cvStratify, cfg.cvStratify.defaultValue)

        InfoBar.success(
            title=self.tr("Settings Reset"),
            content=self.tr("All training parameters have been reset to defaults."),
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000,
        )

    def _export_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export Config"),
            os.path.join(self._output_dir, "config.yaml"),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return

        config = TrainingConfig(
            data_path=self._data_path,
            output_dir=self._output_dir,
            model=cfg.get(cfg.model),
            pretrained=cfg.get(cfg.pretrained),
            batch_size=cfg.get(cfg.batchSize),
            lr=cfg.get(cfg.learningRate),
            epochs=cfg.get(cfg.epochs),
            patience=cfg.get(cfg.patience),
            loss=cfg.get(cfg.loss),
            optimizer=cfg.get(cfg.optimizer),
            scheduler=cfg.get(cfg.scheduler),
            precision=cfg.get(cfg.precision),
            compile=cfg.get(cfg.compile),
            deterministic=cfg.get(cfg.deterministic),
            seed=cfg.get(cfg.seed),
            no_cache=cfg.get(cfg.noCache),
            cv=cfg.get(cfg.cv),
            cv_stratify=cfg.get(cfg.cvStratify),
        )

        with open(path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

        InfoBar.success(
            title=self.tr("Config Exported"),
            content=path,
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000,
        )

    def _on_state_changed(self, state: ProcessState):
        is_running = state in (ProcessState.STARTING, ProcessState.RUNNING)
        if is_running:
            self.systemInfoCard.start_refresh()
        else:
            self.systemInfoCard.stop_refresh()

    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def get_data_info(self) -> DataInfo | None:
        return self.dataInfoCard.data_info
