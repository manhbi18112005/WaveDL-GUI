"""
WaveDL GUI - Training Configuration Interface

Interface for configuring model, hyperparameters, and training options.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    ComboBoxSettingCard,
    DoubleSpinBox,
    ExpandLayout,
    FluentIcon as FIF,
    RangeSettingCard,
    ScrollArea,
    SettingCard,
    SettingCardGroup as CardGroup,
    SpinBox,
    SwitchSettingCard,
    TitleLabel,
    ToolTipFilter,
    ToolTipPosition,
    setFont,
)

from ..common.config import cfg
from ..common.constants import (
    ALL_MODELS,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    PRECISION_OPTIONS,
    SCHEDULERS,
    TOOLTIPS,
    TrainingConfig,
)


class SettingCardGroup(CardGroup):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        setFont(self.titleLabel, 14, QFont.Weight.DemiBold)


class SpinBoxSettingCard(SettingCard):
    """Setting card with a SpinBox for integer values."""

    def __init__(self, configItem, icon, title, content=None, parent=None):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem

        self.spinBox = SpinBox(self)
        self.spinBox.setFixedWidth(160)

        validator = getattr(configItem, "validator", None)
        if validator and hasattr(validator, "min") and hasattr(validator, "max"):
            self.spinBox.setRange(validator.min, validator.max)

        self.spinBox.setValue(cfg.get(configItem))
        self.spinBox.valueChanged.connect(self._onValueChanged)

        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

    def _onValueChanged(self, value):
        cfg.set(self.configItem, value)


class DoubleSpinBoxSettingCard(SettingCard):
    """Setting card with a DoubleSpinBox for float values."""

    def __init__(
        self,
        configItem,
        icon,
        title,
        content=None,
        decimals=6,
        single_step=0.0001,
        minimum=0.0,
        maximum=1.0,
        parent=None,
    ):
        super().__init__(icon, title, content, parent)
        self.configItem = configItem

        self.spinBox = DoubleSpinBox(self)
        self.spinBox.setFixedWidth(160)
        self.spinBox.setDecimals(decimals)
        self.spinBox.setSingleStep(single_step)
        self.spinBox.setRange(minimum, maximum)
        self.spinBox.setValue(cfg.get(configItem))
        self.spinBox.valueChanged.connect(self._onValueChanged)

        self.hBoxLayout.addWidget(self.spinBox, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addSpacing(16)

    def _onValueChanged(self, value):
        cfg.set(self.configItem, value)


class TrainingInterface(ScrollArea):
    """Training configuration interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)

        self._init_ui()
        self._connect_signals()
        self._applyTooltips()

    def _init_ui(self):
        self.setObjectName("trainingInterface")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 90, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        # Title
        self.titleLabel = TitleLabel(self.tr("Training Configuration"), self)
        setFont(self.titleLabel, 23, QFont.Weight.DemiBold)
        self.titleLabel.move(36, 40)

        # Model section
        self.modelGroup = SettingCardGroup(self.tr("Model"), self.scrollWidget)

        self.modelCard = ComboBoxSettingCard(
            cfg.model,
            FIF.ROBOT,
            self.tr("Neural Network Model"),
            self.tr("Select the architecture to train"),
            texts=ALL_MODELS,
            parent=self.modelGroup,
        )

        self.pretrainedCard = SwitchSettingCard(
            FIF.DOWNLOAD,
            self.tr("Use Pretrained Weights"),
            self.tr("Initialize with ImageNet weights (recommended for 2D image data)"),
            configItem=cfg.pretrained,
            parent=self.modelGroup,
        )

        # Hyperparameters section
        self.hyperGroup = SettingCardGroup(
            self.tr("Hyperparameters"), self.scrollWidget
        )

        self.batchSizeCard = SpinBoxSettingCard(
            cfg.batchSize,
            FIF.SPEED_HIGH,
            self.tr("Batch Size"),
            self.tr("Number of samples per training batch"),
            parent=self.hyperGroup,
        )

        self.lrCard = DoubleSpinBoxSettingCard(
            cfg.learningRate,
            FIF.EDIT,
            self.tr("Learning Rate"),
            self.tr("Initial learning rate for optimization"),
            decimals=6,
            single_step=0.0001,
            minimum=0.0,
            maximum=1.0,
            parent=self.hyperGroup,
        )

        self.epochsCard = SpinBoxSettingCard(
            cfg.epochs,
            FIF.HISTORY,
            self.tr("Max Epochs"),
            self.tr("Maximum number of training epochs"),
            parent=self.hyperGroup,
        )

        self.patienceCard = SpinBoxSettingCard(
            cfg.patience,
            FIF.STOP_WATCH,
            self.tr("Early Stopping Patience"),
            self.tr("Stop if no improvement for this many epochs"),
            parent=self.hyperGroup,
        )

        # Optimization section
        self.optimGroup = SettingCardGroup(self.tr("Optimization"), self.scrollWidget)

        self.lossCard = ComboBoxSettingCard(
            cfg.loss,
            FIF.ALIGNMENT,
            self.tr("Loss Function"),
            self.tr("Loss function for training"),
            texts=[loss[0] for loss in LOSS_FUNCTIONS],
            parent=self.optimGroup,
        )

        self.optimizerCard = ComboBoxSettingCard(
            cfg.optimizer,
            FIF.SPEED_OFF,
            self.tr("Optimizer"),
            self.tr("Optimization algorithm"),
            texts=[o[0] for o in OPTIMIZERS],
            parent=self.optimGroup,
        )

        self.schedulerCard = ComboBoxSettingCard(
            cfg.scheduler,
            FIF.DATE_TIME,
            self.tr("LR Scheduler"),
            self.tr("Learning rate scheduling strategy"),
            texts=[s[0] for s in SCHEDULERS],
            parent=self.optimGroup,
        )

        self.precisionCard = ComboBoxSettingCard(
            cfg.precision,
            FIF.SPEED_HIGH,
            self.tr("Mixed Precision"),
            self.tr("Training precision (BF16 recommended for modern GPUs)"),
            texts=[p[0] for p in PRECISION_OPTIONS],
            parent=self.optimGroup,
        )

        # Advanced section
        self.advancedGroup = SettingCardGroup(self.tr("Advanced"), self.scrollWidget)

        self.compileCard = SwitchSettingCard(
            FIF.DEVELOPER_TOOLS,
            self.tr("Compile Model"),
            self.tr("Use torch.compile() for faster training (PyTorch 2.0+)"),
            configItem=cfg.compile,
            parent=self.advancedGroup,
        )

        self.deterministicCard = SwitchSettingCard(
            FIF.SYNC,
            self.tr("Deterministic Mode"),
            self.tr("Enable for reproducible results (may reduce performance)"),
            configItem=cfg.deterministic,
            parent=self.advancedGroup,
        )

        self.seedCard = SpinBoxSettingCard(
            cfg.seed,
            FIF.LABEL,
            self.tr("Random Seed"),
            self.tr("Seed for reproducibility"),
            parent=self.advancedGroup,
        )

        self.noCacheCard = SwitchSettingCard(
            FIF.DELETE,
            self.tr("Disable Data Cache"),
            self.tr(
                "Delete cached data before training so it is reloaded from scratch"
            ),
            configItem=cfg.noCache,
            parent=self.advancedGroup,
        )

        # Cross-validation section
        self.cvGroup = SettingCardGroup(self.tr("Cross-Validation"), self.scrollWidget)

        self.cvCard = RangeSettingCard(
            cfg.cv,
            FIF.LIBRARY,
            self.tr("K-Fold CV"),
            self.tr("Number of folds (0 = disabled)"),
            parent=self.cvGroup,
        )

        self.cvStratifyCard = SwitchSettingCard(
            FIF.FILTER,
            self.tr("Stratified Splitting"),
            self.tr("Ensure similar target distributions in each fold"),
            configItem=cfg.cvStratify,
            parent=self.cvGroup,
        )

        # Add cards to groups
        self.modelGroup.addSettingCard(self.modelCard)
        self.modelGroup.addSettingCard(self.pretrainedCard)

        self.hyperGroup.addSettingCard(self.batchSizeCard)
        self.hyperGroup.addSettingCard(self.lrCard)
        self.hyperGroup.addSettingCard(self.epochsCard)
        self.hyperGroup.addSettingCard(self.patienceCard)

        self.optimGroup.addSettingCard(self.lossCard)
        self.optimGroup.addSettingCard(self.optimizerCard)
        self.optimGroup.addSettingCard(self.schedulerCard)
        self.optimGroup.addSettingCard(self.precisionCard)

        self.advancedGroup.addSettingCard(self.compileCard)
        self.advancedGroup.addSettingCard(self.deterministicCard)
        self.advancedGroup.addSettingCard(self.seedCard)
        self.advancedGroup.addSettingCard(self.noCacheCard)

        self.cvGroup.addSettingCard(self.cvCard)
        self.cvGroup.addSettingCard(self.cvStratifyCard)

        # Layout
        self.expandLayout.setSpacing(20)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.modelGroup)
        self.expandLayout.addWidget(self.hyperGroup)
        self.expandLayout.addWidget(self.optimGroup)
        self.expandLayout.addWidget(self.advancedGroup)
        self.expandLayout.addWidget(self.cvGroup)

    def _connect_signals(self):
        """Connect signals for config updates."""
        pass  # Config items auto-save

    def _applyTooltips(self):
        mapping = {
            "model": self.modelCard,
            "pretrained": self.pretrainedCard,
            "batch_size": self.batchSizeCard,
            "learning_rate": self.lrCard,
            "epochs": self.epochsCard,
            "patience": self.patienceCard,
            "loss": self.lossCard,
            "optimizer": self.optimizerCard,
            "scheduler": self.schedulerCard,
            "precision": self.precisionCard,
            "compile": self.compileCard,
            "deterministic": self.deterministicCard,
            "seed": self.seedCard,
            "no_cache": self.noCacheCard,
            "cv": self.cvCard,
            "cv_stratify": self.cvStratifyCard,
        }
        for key, card in mapping.items():
            text = TOOLTIPS.get(key)
            if text:
                card.setToolTip(text)
                card.installEventFilter(
                    ToolTipFilter(card, 300, ToolTipPosition.BOTTOM)
                )

    def get_training_config(
        self, data_path: str = "", output_dir: str = ""
    ) -> TrainingConfig:
        """Build a TrainingConfig from current UI state."""
        return TrainingConfig(
            data_path=data_path,
            output_dir=output_dir,
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
