"""
WaveDL GUI - Controls Card Component

A premium card with Start/Stop training buttons and quick action shortcuts.
Self-contained: listens to the signal bus for state changes and manages
its own button enable/disable state.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon as FIF,
    IconWidget,
    PrimaryPushButton,
    PushButton,
    SimpleCardWidget,
    TransparentPushButton,
    isDarkTheme,
    setFont,
)

from ..common.signal_bus import signalBus
from ..service.training_service import ProcessState, trainingService


# ─── Color palette (shared with dashboard) ─────────────────────────────────────


def _muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def _subtle_border_color() -> QColor:
    return QColor(0, 0, 0, 18) if not isDarkTheme() else QColor(255, 255, 255, 18)


# ─── Reusable sub-widgets ──────────────────────────────────────────────────────


class _Separator(QFrame):
    """Thin horizontal line separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFixedHeight(1)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(Qt.NoPen)
        p.setBrush(_subtle_border_color())
        p.drawRect(self.rect())
        p.end()


class _SectionHeader(QWidget):
    """Section header with an icon and title label."""

    def __init__(self, icon: FIF, title: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(8)

        ic = IconWidget(icon, self)
        ic.setFixedSize(16, 16)
        h.addWidget(ic)

        lbl = CaptionLabel(title.upper(), self)
        lbl.setTextColor(_muted_text_color(), _muted_text_color())
        setFont(lbl, 10, QFont.Weight.Bold)
        h.addWidget(lbl)
        h.addStretch()


# ─── Controls Card ─────────────────────────────────────────────────────────────


class ControlsCard(SimpleCardWidget):
    """Card with Start/Stop training buttons and quick-action shortcuts.

    Listens to ``signalBus.trainingStateChangedSig`` to enable/disable
    buttons automatically.  Emits signals for quick actions so the parent
    view can handle them with the right context.
    """

    # Quick-action signals (parent view handles the logic)
    openOutputClicked = Signal()
    openConfigClicked = Signal()
    resetSettingsClicked = Signal()
    exportConfigClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        QTimer.singleShot(0, self._adjustHeight)

    # ── ExpandLayout fix ──────────────────────────────────────────────

    def _adjustHeight(self):
        self.setFixedHeight(self.layout().sizeHint().height())
        QTimer.singleShot(0, self._updateParentLayout)

    def _updateParentLayout(self):
        parent = self.parentWidget()
        if parent is not None:
            parent.adjustSize()
            parent.updateGeometry()

    # ── UI construction ──────────────────────────────────────────────

    def _init_ui(self):
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)

        # ── Training controls section ────────────────────────────────
        root.addWidget(_SectionHeader(FIF.PLAY, "Controls", self))
        root.addSpacing(4)

        # Hint text
        self.hintLabel = CaptionLabel(
            self.tr(
                "Make sure you have selected a data file and configured "
                "training parameters before starting."
            ),
            self,
        )
        self.hintLabel.setWordWrap(True)
        self.hintLabel.setTextColor(_muted_text_color(), _muted_text_color())
        root.addWidget(self.hintLabel)
        root.addSpacing(4)

        # Start / Stop buttons
        btnRow = QHBoxLayout()
        btnRow.setSpacing(10)

        self.startBtn = PrimaryPushButton(FIF.PLAY, self.tr("Start Training"), self)
        self.stopBtn = PushButton(FIF.PAUSE, self.tr("Stop Training"), self)
        self.stopBtn.setEnabled(False)

        btnRow.addWidget(self.startBtn)
        btnRow.addWidget(self.stopBtn)
        btnRow.addStretch()
        root.addLayout(btnRow)

        # ── Separator ────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_Separator(self))
        root.addSpacing(4)

        # ── Quick actions section ────────────────────────────────────
        root.addWidget(_SectionHeader(FIF.SPEED_HIGH, "Quick Actions", self))
        root.addSpacing(4)

        grid = QGridLayout()
        grid.setSpacing(8)

        self.openOutputBtn = TransparentPushButton(
            FIF.FOLDER, self.tr("Open Output Folder"), self
        )
        self.openConfigBtn = TransparentPushButton(
            FIF.DOCUMENT, self.tr("Open Config File"), self
        )
        self.resetBtn = TransparentPushButton(FIF.SYNC, self.tr("Reset Settings"), self)
        self.exportBtn = TransparentPushButton(FIF.SAVE, self.tr("Export Config"), self)

        grid.addWidget(self.openOutputBtn, 0, 0)
        grid.addWidget(self.openConfigBtn, 0, 1)
        grid.addWidget(self.resetBtn, 1, 0)
        grid.addWidget(self.exportBtn, 1, 1)

        root.addLayout(grid)

    # ── Signal wiring ────────────────────────────────────────────────

    def _connect_signals(self):
        # Training controls → signal bus / service
        self.startBtn.clicked.connect(self._on_start_clicked)
        self.stopBtn.clicked.connect(self._on_stop_clicked)

        # State changes → button enable/disable
        signalBus.trainingStateChangedSig.connect(self._on_state_changed)

        # Quick actions → parent-handled signals
        self.openOutputBtn.clicked.connect(self.openOutputClicked)
        self.openConfigBtn.clicked.connect(self.openConfigClicked)
        self.resetBtn.clicked.connect(self.resetSettingsClicked)
        self.exportBtn.clicked.connect(self.exportConfigClicked)

    # ── Handlers ─────────────────────────────────────────────────────

    def _on_start_clicked(self):
        """Request MainWindow to validate and start training."""
        signalBus.requestStartTrainingSig.emit()

    def _on_stop_clicked(self):
        """Stop the running training process."""
        trainingService.stop_training()

    @Slot(object)
    def _on_state_changed(self, state: ProcessState):
        """Enable/disable buttons based on training state."""
        is_running = state in (ProcessState.STARTING, ProcessState.RUNNING)
        self.startBtn.setEnabled(not is_running)
        self.stopBtn.setEnabled(is_running)
