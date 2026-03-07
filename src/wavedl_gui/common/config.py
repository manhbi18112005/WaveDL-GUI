import sys
from enum import Enum

from PySide6.QtCore import QLocale, QStandardPaths
from qfluentwidgets import (
    BoolValidator,
    ConfigItem,
    ConfigSerializer,
    FolderValidator,
    OptionsConfigItem,
    OptionsValidator,
    QConfig,
    RangeConfigItem,
    RangeValidator,
    Theme,
    qconfig,
)

from .constants.models import ALL_MODELS
from .setting import CONFIG_FILE


class Language(Enum):
    """Language enumeration"""

    CHINESE_SIMPLIFIED = QLocale(QLocale.Chinese, QLocale.China)
    ENGLISH = QLocale(QLocale.English)
    AUTO = QLocale()


class LanguageSerializer(ConfigSerializer):
    """Language serializer"""

    def serialize(self, language):
        return language.value.name() if language != Language.AUTO else "Auto"

    def deserialize(self, value: str):
        return Language(QLocale(value)) if value != "Auto" else Language.AUTO


def isWin11():
    return sys.platform == "win32" and sys.getwindowsversion().build >= 22000


class Config(QConfig):
    """Config of application"""

    # main window
    micaEnabled = ConfigItem("MainWindow", "MicaEnabled", isWin11(), BoolValidator())
    dpiScale = OptionsConfigItem(
        "MainWindow",
        "DpiScale",
        "Auto",
        OptionsValidator([1, 1.25, 1.5, 1.75, 2, "Auto"]),
        restart=True,
    )
    language = OptionsConfigItem(
        "MainWindow",
        "Language",
        Language.AUTO,
        OptionsValidator(Language),
        LanguageSerializer(),
        restart=True,
    )
    accentColor = OptionsConfigItem(
        "MainWindow", "AccentColor", "#009faa", OptionsValidator(["#009faa", "Auto"])
    )

    # software update
    checkUpdateAtStartUp = ConfigItem(
        "Update", "CheckUpdateAtStartUp", True, BoolValidator()
    )

    # onboarding wizard
    showWizardOnStartup = ConfigItem(
        "MainWindow", "ShowWizardOnStartup", True, BoolValidator()
    )

    # training - output
    outputFolder = ConfigItem(
        "Training",
        "OutputFolder",
        QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
        FolderValidator(),
    )

    # training - hyperparameters
    batchSize = RangeConfigItem("Training", "BatchSize", 128, RangeValidator(1, 4096))
    learningRate = ConfigItem("Training", "LearningRate", 0.001)
    epochs = RangeConfigItem("Training", "Epochs", 1000, RangeValidator(1, 100000))
    patience = RangeConfigItem("Training", "Patience", 20, RangeValidator(1, 1000))
    weightDecay = ConfigItem("Training", "WeightDecay", 0.0001)
    gradClip = ConfigItem("Training", "GradClip", 1.0)
    gradAccumSteps = RangeConfigItem(
        "Training", "GradAccumSteps", 1, RangeValidator(1, 128)
    )
    seed = RangeConfigItem("Training", "Seed", 2025, RangeValidator(0, 999999))
    workers = RangeConfigItem("Training", "Workers", -1, RangeValidator(-1, 64))

    # training - model
    model = OptionsConfigItem("Training", "Model", "cnn", OptionsValidator(ALL_MODELS))
    pretrained = ConfigItem("Training", "Pretrained", True, BoolValidator())

    # training - optimization
    loss = OptionsConfigItem(
        "Training",
        "Loss",
        "mse",
        OptionsValidator(["mse", "mae", "huber", "smooth_l1", "log_cosh"]),
    )
    optimizer = OptionsConfigItem(
        "Training",
        "Optimizer",
        "adamw",
        OptionsValidator(["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]),
    )
    scheduler = OptionsConfigItem(
        "Training",
        "Scheduler",
        "plateau",
        OptionsValidator(
            [
                "plateau",
                "cosine",
                "cosine_restarts",
                "onecycle",
                "step",
                "multistep",
                "exponential",
                "linear_warmup",
            ]
        ),
    )
    precision = OptionsConfigItem(
        "Training", "Precision", "bf16", OptionsValidator(["bf16", "fp16", "no"])
    )

    # training - advanced
    compile = ConfigItem("Training", "Compile", False, BoolValidator())
    deterministic = ConfigItem("Training", "Deterministic", False, BoolValidator())
    wandb = ConfigItem("Training", "Wandb", False, BoolValidator())
    noCache = ConfigItem("Training", "NoCache", False, BoolValidator())
    cacheValidate = OptionsConfigItem(
        "Training",
        "CacheValidate",
        "sha256",
        OptionsValidator(["sha256", "fast", "size"]),
    )

    # cross-validation
    cv = RangeConfigItem("Training", "CV", 0, RangeValidator(0, 20))
    cvStratify = ConfigItem("Training", "CVStratify", False, BoolValidator())


cfg = Config()
cfg.themeMode.value = Theme.AUTO
qconfig.load(str(CONFIG_FILE.absolute()), cfg)
