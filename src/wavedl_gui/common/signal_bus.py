from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    """Signal bus for application-wide event communication"""

    # Application signals
    appMessageSig = Signal(str)
    appErrorSig = Signal(str)

    # Window signals
    checkUpdateSig = Signal()
    micaEnableChanged = Signal(bool)

    # Training signals
    trainingProgressSig = Signal(object)  # TrainingProgress dataclass
    trainingOutputSig = Signal(str)  # Log output line
    trainingStateChangedSig = Signal(object)  # ProcessState enum
    trainingCompletedSig = Signal(bool, str)  # (success, message)
    historyUpdatedSig = Signal(list)  # List of TrainingMetrics

    # Testing signals
    testCompletedSig = Signal(bool, str, dict)  # (success, message, results)

    # Navigation signals
    switchToProjectSig = Signal()
    switchToTrainingSig = Signal()
    switchToDashboardSig = Signal()
    switchToResultsSig = Signal()

    # Action signals
    requestStartTrainingSig = Signal()  # Request MainWindow to start training


signalBus = SignalBus()
