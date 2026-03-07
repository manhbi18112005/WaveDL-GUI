"""WaveDL GUI - Settings Interface"""

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import QFileDialog, QWidget
from qfluentwidgets import (
    ComboBoxSettingCard,
    ExpandLayout,
    FluentIcon as FIF,
    InfoBar,
    PrimaryPushSettingCard,
    PushSettingCard,
    ScrollArea,
    SettingCard,
    SwitchSettingCard,
    TitleLabel,
    setFont,
    setTheme,
    setThemeColor,
)
from qframelesswindow.utils import getSystemAccentColor

from ..common.config import cfg
from ..common.setting import AUTHOR, FEEDBACK_URL, VERSION, YEAR
from ..common.signal_bus import signalBus
from ..components.shared import SettingCardGroup


class SettingInterface(ScrollArea):
    """Setting interface"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        # setting label
        self.settingLabel = TitleLabel(self.tr("Settings"), self)

        # personalization
        self.personalGroup = SettingCardGroup(
            self.tr("Personalization"), self.scrollWidget
        )
        self.themeCard = ComboBoxSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            self.tr("Application theme"),
            self.tr("Change the appearance of your application"),
            texts=[self.tr("Light"), self.tr("Dark"), self.tr("Use system setting")],
            parent=self.personalGroup,
        )
        self.accentColorCard = ComboBoxSettingCard(
            cfg.accentColor,
            FIF.PALETTE,
            self.tr("Accent color"),
            self.tr("Change the accent color of your application"),
            texts=[self.tr("Sea foam green"), self.tr("Use system setting")],
            parent=self.personalGroup,
        )
        self.zoomCard = ComboBoxSettingCard(
            cfg.dpiScale,
            FIF.ZOOM,
            self.tr("Interface zoom"),
            self.tr("Change the size of widgets and fonts"),
            texts=[
                "100%",
                "125%",
                "150%",
                "175%",
                "200%",
                self.tr("Use system setting"),
            ],
            parent=self.personalGroup,
        )
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            self.tr("Language"),
            self.tr("Set your preferred language for UI"),
            texts=["Vietnamese", "English", self.tr("Use system setting")],
            parent=self.personalGroup,
        )

        # training defaults
        self.trainingGroup = SettingCardGroup(
            self.tr("Training Defaults"), self.scrollWidget
        )
        self.outputFolderCard = PushSettingCard(
            self.tr("Choose"),
            FIF.FOLDER,
            self.tr("Default output folder"),
            cfg.get(cfg.outputFolder),
            self.trainingGroup,
        )
        self.precisionCard = ComboBoxSettingCard(
            cfg.precision,
            FIF.SPEED_HIGH,
            self.tr("Mixed precision"),
            self.tr("Training precision mode (BF16 recommended for modern GPUs)"),
            texts=["BFloat16", "Float16", "Full Precision (FP32)"],
            parent=self.trainingGroup,
        )
        self.wandbCard = SwitchSettingCard(
            FIF.CLOUD,
            self.tr("Enable Weights & Biases"),
            self.tr("Log training metrics to W&B for visualization"),
            configItem=cfg.wandb,
            parent=self.trainingGroup,
        )
        self.deterministicCard = SwitchSettingCard(
            FIF.SYNC,
            self.tr("Deterministic mode"),
            self.tr("Enable for reproducible results (may reduce performance)"),
            configItem=cfg.deterministic,
            parent=self.trainingGroup,
        )

        # update software
        self.updateSoftwareGroup = SettingCardGroup(
            self.tr("Software update"), self.scrollWidget
        )
        self.updateOnStartUpCard = SwitchSettingCard(
            FIF.UPDATE,
            self.tr("Check for updates when the application starts"),
            self.tr("The new version will be more stable and have more features"),
            configItem=cfg.checkUpdateAtStartUp,
            parent=self.updateSoftwareGroup,
        )
        self.wizardOnStartupCard = SwitchSettingCard(
            FIF.EDUCATION,
            self.tr("Show setup wizard on startup"),
            self.tr("Display the guided onboarding wizard when the application starts"),
            configItem=cfg.showWizardOnStartup,
            parent=self.updateSoftwareGroup,
        )

        # application
        self.aboutGroup = SettingCardGroup(self.tr("About"), self.scrollWidget)
        self.feedbackCard = PrimaryPushSettingCard(
            self.tr("Provide feedback"),
            FIF.FEEDBACK,
            self.tr("Provide feedback"),
            self.tr("Help us improve WaveDL by providing feedback"),
            self.aboutGroup,
        )
        self.aboutCard = PrimaryPushSettingCard(
            self.tr("Check update"),
            FIF.PEOPLE,
            self.tr("About"),
            "© "
            + self.tr("Copyright")
            + f" {YEAR}, {AUTHOR}. "
            + self.tr("Version")
            + " v"
            + VERSION,
            self.aboutGroup,
        )

        self.__initWidget()

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 90, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName("settingInterface")

        # initialize style sheet
        setFont(self.settingLabel, 23, QFont.Weight.DemiBold)
        self.enableTransparentBackground()

        # initialize layout
        self.__initLayout()
        self._connect_signals()

    def __initLayout(self):
        self.settingLabel.move(36, 40)

        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.zoomCard)
        self.personalGroup.addSettingCard(self.languageCard)
        self.personalGroup.addSettingCard(self.accentColorCard)

        self.trainingGroup.addSettingCard(self.outputFolderCard)
        self.trainingGroup.addSettingCard(self.precisionCard)
        self.trainingGroup.addSettingCard(self.wandbCard)
        self.trainingGroup.addSettingCard(self.deterministicCard)

        self.updateSoftwareGroup.addSettingCard(self.updateOnStartUpCard)
        self.updateSoftwareGroup.addSettingCard(self.wizardOnStartupCard)

        self.aboutGroup.addSettingCard(self.feedbackCard)
        self.aboutGroup.addSettingCard(self.aboutCard)

        # add setting card group to layout
        self.expandLayout.setSpacing(26)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.trainingGroup)
        self.expandLayout.addWidget(self.updateSoftwareGroup)
        self.expandLayout.addWidget(self.aboutGroup)

        # adjust icon size
        for card in self.findChildren(SettingCard):
            card.setIconSize(18, 18)

    def _showRestartTooltip(self):
        """show restart tooltip"""
        InfoBar.success(
            self.tr("Updated successfully"),
            self.tr("Configuration takes effect after restart"),
            duration=1500,
            parent=self,
        )

    def _onOutputFolderCardClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Choose Output Folder"), cfg.get(cfg.outputFolder)
        )

        if not folder or cfg.get(cfg.outputFolder) == folder:
            return

        cfg.set(cfg.outputFolder, folder)
        self.outputFolderCard.setContent(folder)

    def _onAccentColorChanged(self):
        color = cfg.get(cfg.accentColor)
        if color != "Auto":
            setThemeColor(color, save=False)
        else:
            sysColor = getSystemAccentColor()
            if sysColor.isValid():
                setThemeColor(sysColor, save=False)
            else:
                setThemeColor(color, save=False)

    def _connect_signals(self):
        """connect signal to slot"""
        cfg.appRestartSig.connect(self._showRestartTooltip)

        # training
        self.outputFolderCard.clicked.connect(self._onOutputFolderCardClicked)

        # personalization
        cfg.themeChanged.connect(setTheme)
        cfg.accentColor.valueChanged.connect(self._onAccentColorChanged)

        # check update
        self.aboutCard.clicked.connect(signalBus.checkUpdateSig)

        # about
        self.feedbackCard.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL))
        )
