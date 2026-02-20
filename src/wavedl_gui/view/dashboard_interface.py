"""
WaveDL GUI - Training Dashboard Interface

A premium monitoring dashboard for real-time training progress.
Uses the same visual design system as DataInfoCard for consistency.
"""

from __future__ import annotations

from typing import ClassVar

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon as FIF,
    IconWidget,
    InfoBar,
    InfoBarPosition,
    PlainTextEdit,
    ProgressBar,
    ScrollArea,
    SimpleCardWidget,
    StrongBodyLabel,
    TitleLabel,
    TransparentPushButton,
    isDarkTheme,
    setFont,
)

from ..common.signal_bus import signalBus
from ..components.statistic_widget import StatisticsWidget
from ..service.training_service import ProcessState, TrainingProgress


# ─── Color palette (shared with DataInfoCard) ─────────────────────────────────


def _accent_color() -> QColor:
    return QColor("#3b82f6") if not isDarkTheme() else QColor("#60a5fa")


def _success_color() -> QColor:
    return QColor("#16a34a") if not isDarkTheme() else QColor("#4ade80")


def _warning_color() -> QColor:
    return QColor("#d97706") if not isDarkTheme() else QColor("#fbbf24")


def _error_color() -> QColor:
    return QColor("#dc2626") if not isDarkTheme() else QColor("#f87171")


def _muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def _subtle_border_color() -> QColor:
    return QColor(0, 0, 0, 18) if not isDarkTheme() else QColor(255, 255, 255, 18)


def _section_bg_color() -> QColor:
    return QColor(0, 0, 0, 6) if not isDarkTheme() else QColor(255, 255, 255, 6)


# ─── Separator ─────────────────────────────────────────────────────────────────


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


# ─── Section header ───────────────────────────────────────────────────────────


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


# ─── Status badge ─────────────────────────────────────────────────────────────


class _StatusBadge(QWidget):
    """Small rounded badge showing the current process state."""

    _STATE_MAP: ClassVar[dict[ProcessState, tuple[str, str | None]]] = {
        ProcessState.IDLE: ("Idle", None),
        ProcessState.STARTING: ("Starting", "warning"),
        ProcessState.RUNNING: ("Running", "accent"),
        ProcessState.STOPPING: ("Stopping", "warning"),
        ProcessState.COMPLETED: ("Completed", "success"),
        ProcessState.FAILED: ("Failed", "error"),
        ProcessState.CANCELLED: ("Cancelled", "error"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = "Idle"
        self._color_key = None
        self.setFixedHeight(24)
        self._update_width()

    def setState(self, state: ProcessState):
        text, color_key = self._STATE_MAP.get(state, ("Unknown", None))
        self._text = text
        self._color_key = color_key
        self._update_width()
        self.update()

    def _resolve_color(self) -> QColor:
        if self._color_key == "accent":
            return _accent_color()
        if self._color_key == "success":
            return _success_color()
        if self._color_key == "warning":
            return _warning_color()
        if self._color_key == "error":
            return _error_color()
        return _muted_text_color()

    def _update_width(self):
        from PySide6.QtGui import QFontMetrics

        fm = QFontMetrics(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.setFixedWidth(fm.horizontalAdvance(self._text) + 24)

    def paintEvent(self, _):
        if not self._text:
            return
        color = self._resolve_color()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        bg = QColor(color)
        bg.setAlpha(30 if not isDarkTheme() else 45)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)

        p.setPen(color)
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        p.drawText(self.rect(), Qt.AlignCenter, self._text)
        p.end()


# ─── Duration formatting ───────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


# ─── Progress card ─────────────────────────────────────────────────────────────


class ProgressCard(SimpleCardWidget):
    """Premium card showing epoch progress, status badge, progress bar, and ETA."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status_color = _accent_color()
        self._init_ui()

    def _init_ui(self):
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        root.addLayout(content)

        # ── Header row: icon + title + status badge ──────────────
        header = QHBoxLayout()
        header.setSpacing(10)

        ic = IconWidget(FIF.SPEED_HIGH, self)
        ic.setFixedSize(20, 20)
        header.addWidget(ic)

        titleCol = QVBoxLayout()
        titleCol.setSpacing(2)
        titleLabel = StrongBodyLabel(self.tr("Training Progress"), self)
        setFont(titleLabel, 15, QFont.Weight.DemiBold)
        self.subtitleLabel = CaptionLabel(
            self.tr("Configure your project and press Start to begin"), self
        )
        self.subtitleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        titleCol.addWidget(titleLabel)
        titleCol.addWidget(self.subtitleLabel)
        header.addLayout(titleCol, 1)

        self.statusBadge = _StatusBadge(self)
        header.addWidget(self.statusBadge)
        content.addLayout(header)

        content.addSpacing(16)

        # ── Epoch + patience + progress bar ──────────────────────────
        infoRow = QHBoxLayout()
        self.epochLabel = CaptionLabel("Epoch: —/—", self)
        self.epochLabel.setTextColor(_muted_text_color(), _muted_text_color())
        self.patienceLabel = CaptionLabel("", self)
        self.patienceLabel.setTextColor(_muted_text_color(), _muted_text_color())
        infoRow.addWidget(self.epochLabel)
        infoRow.addStretch()
        infoRow.addWidget(self.patienceLabel)
        content.addLayout(infoRow)
        content.addSpacing(6)

        self.progressBar = ProgressBar(self)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setFixedHeight(6)
        content.addWidget(self.progressBar)

        content.addSpacing(8)

        # ── Footer row: percentage + ETA ──────────────────────────
        footerRow = QHBoxLayout()
        self.percentLabel = CaptionLabel("0%", self)
        self.percentLabel.setTextColor(_muted_text_color(), _muted_text_color())
        self.etaLabel = CaptionLabel("", self)
        self.etaLabel.setTextColor(_muted_text_color(), _muted_text_color())
        footerRow.addWidget(self.percentLabel)
        footerRow.addStretch()
        footerRow.addWidget(self.etaLabel)
        content.addLayout(footerRow)

    def update_progress(self, progress: TrainingProgress):
        """Update with live training progress."""
        self.epochLabel.setText(f"Epoch: {progress.epoch}/{progress.total_epochs}")
        pct = int(progress.progress_percent)
        self.progressBar.setValue(pct)
        self.percentLabel.setText(f"{pct}%")
        self.subtitleLabel.setText(self.tr("Training in progress"))

        if progress.max_patience > 0:
            self.patienceLabel.setText(
                f"Patience: {progress.patience_counter}/{progress.max_patience}"
            )

        eta_parts = []
        if progress.eta_seconds > 0:
            eta_parts.append(f"ETA: {_format_duration(progress.eta_seconds)}")
        if progress.total_time > 0:
            eta_parts.append(f"Elapsed: {_format_duration(progress.total_time)}")
        self.etaLabel.setText("  ·  ".join(eta_parts))

    def set_state(self, state: ProcessState):
        """Update the status badge."""
        self.statusBadge.setState(state)
        self._status_color = {
            ProcessState.IDLE: _accent_color,
            ProcessState.STARTING: _warning_color,
            ProcessState.RUNNING: _accent_color,
            ProcessState.STOPPING: _warning_color,
            ProcessState.COMPLETED: _success_color,
            ProcessState.FAILED: _error_color,
            ProcessState.CANCELLED: _error_color,
        }.get(state, _accent_color)()

        if state == ProcessState.COMPLETED:
            self.subtitleLabel.setText(self.tr("Training finished successfully"))
        elif state == ProcessState.FAILED:
            self.subtitleLabel.setText(self.tr("Training encountered an error"))
        elif state == ProcessState.CANCELLED:
            self.subtitleLabel.setText(self.tr("Training was cancelled"))
        elif state == ProcessState.IDLE:
            self.subtitleLabel.setText(
                self.tr("Configure your project and press Start to begin")
            )
        self.update()

    def reset(self):
        """Reset to initial state."""
        self.epochLabel.setText("Epoch: —/—")
        self.patienceLabel.setText("")
        self.progressBar.setValue(0)
        self.percentLabel.setText("0%")
        self.etaLabel.setText("")
        self.statusBadge.setState(ProcessState.IDLE)
        self.subtitleLabel.setText(
            self.tr("Configure your project and press Start to begin")
        )
        self._status_color = _accent_color()
        self.update()

    # ── Custom painting (accent bar at top, matching DataInfoCard) ──

    def _normalBackgroundColor(self):
        return QColor(255, 255, 255, 13 if isDarkTheme() else 170)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.borderRadius

        # Card background
        p.setBrush(self._normalBackgroundColor())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

        # Top accent bar (4 px)
        accent_rect = self.rect().adjusted(1, 1, -1, 0)
        accent_rect.setHeight(4)
        path = QPainterPath()
        path.addRoundedRect(
            accent_rect.x(),
            accent_rect.y(),
            accent_rect.width(),
            r * 2,
            r,
            r,
        )
        clip_rect = QPainterPath()
        clip_rect.addRect(
            accent_rect.x(),
            accent_rect.y(),
            accent_rect.width(),
            accent_rect.height(),
        )
        path = path.intersected(clip_rect)
        p.setBrush(self._status_color)
        p.drawPath(path)

        p.end()


# ─── Log card ──────────────────────────────────────────────────────────────────


class _LogCard(SimpleCardWidget):
    """Card with monospaced log output and clear button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(8)

        # Header row: section title + clear button
        headerRow = QHBoxLayout()
        headerRow.addWidget(_SectionHeader(FIF.COMMAND_PROMPT, "Training Log", self))
        headerRow.addStretch()

        self.clearBtn = TransparentPushButton(FIF.DELETE, self.tr("Clear"), self)
        headerRow.addWidget(self.clearBtn)
        root.addLayout(headerRow)

        # Log output
        self.logOutput = PlainTextEdit(self)
        self.logOutput.setReadOnly(True)
        self.logOutput.setMinimumHeight(220)
        self.logOutput.setPlaceholderText(
            self.tr("Training output will appear here...")
        )
        self.logOutput.setStyleSheet("""
            PlainTextEdit {
                font-family: 'Cascadia Code', 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                border: none;
                background: transparent;
            }
        """)
        root.addWidget(self.logOutput)

        self.clearBtn.clicked.connect(self.logOutput.clear)


# ─── Dashboard interface ──────────────────────────────────────────────────────


class DashboardInterface(ScrollArea):
    """Training dashboard interface for monitoring progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollWidget = QWidget()
        self.expandLayout = QVBoxLayout(self.scrollWidget)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        self.setObjectName("dashboardInterface")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 90, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        # ── Page title ─────────────────────────────────────────────
        self.titleLabel = TitleLabel(self.tr("Training Dashboard"), self)
        setFont(self.titleLabel, 23, QFont.Weight.DemiBold)
        self.titleLabel.move(36, 40)

        # ── Progress card ──────────────────────────────────────────
        self.progressCard = ProgressCard(self.scrollWidget)

        # ── Metrics row ────────────────────────────────────────────
        metricsWidget = QWidget(self.scrollWidget)
        metricsLayout = QHBoxLayout(metricsWidget)
        metricsLayout.setContentsMargins(0, 0, 0, 0)
        metricsLayout.setSpacing(10)

        self.trainLossCard = StatisticsWidget(self.tr("Train Loss"), metricsWidget)
        self.valLossCard = StatisticsWidget(self.tr("Val Loss"), metricsWidget)
        self.r2Card = StatisticsWidget(self.tr("R\u00b2 Score"), metricsWidget)
        self.lrCard = StatisticsWidget(self.tr("Learning Rate"), metricsWidget)

        for card in (self.trainLossCard, self.valLossCard, self.r2Card, self.lrCard):
            metricsLayout.addWidget(card)

        # ── Metrics row 2 ─────────────────────────────────────────────
        metricsWidget2 = QWidget(self.scrollWidget)
        metricsLayout2 = QHBoxLayout(metricsWidget2)
        metricsLayout2.setContentsMargins(0, 0, 0, 0)
        metricsLayout2.setSpacing(10)

        self.pearsonCard = StatisticsWidget(self.tr("Pearson"), metricsWidget2)
        self.gradNormCard = StatisticsWidget(self.tr("Grad Norm"), metricsWidget2)
        self.maeCard = StatisticsWidget(self.tr("MAE Avg"), metricsWidget2)
        self.epochTimeCard = StatisticsWidget(self.tr("Epoch Time"), metricsWidget2)

        for card in (
            self.pearsonCard,
            self.gradNormCard,
            self.maeCard,
            self.epochTimeCard,
        ):
            metricsLayout2.addWidget(card)

        # ── Log card ───────────────────────────────────────────────
        self.logCard = _LogCard(self.scrollWidget)

        # ── Assemble layout ────────────────────────────────────────
        self.expandLayout.setSpacing(15)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.progressCard)
        self.expandLayout.addWidget(metricsWidget)
        self.expandLayout.addWidget(metricsWidget2)
        self.expandLayout.addWidget(self.logCard)

    def _connect_signals(self):
        signalBus.trainingProgressSig.connect(self._on_progress)
        signalBus.trainingOutputSig.connect(self._on_output)
        signalBus.trainingStateChangedSig.connect(self._on_state_changed)
        signalBus.trainingCompletedSig.connect(self._on_completed)

    # ── Signal handlers ────────────────────────────────────────────

    @Slot(object)
    def _on_progress(self, progress: TrainingProgress):
        """Handle real-time progress update."""
        self.progressCard.update_progress(progress)

        self.trainLossCard.setValue(f"{progress.train_loss:.6f}")
        self.valLossCard.setValue(
            f"{progress.val_loss:.6f}",
            f"Best: {progress.best_val_loss:.6f}",
        )
        self.r2Card.setValue(f"{progress.r2_score:.4f}")
        self.lrCard.setValue(f"{progress.learning_rate:.2e}")

        self.pearsonCard.setValue(f"{progress.pearson:.4f}")
        self.gradNormCard.setValue(f"{progress.grad_norm:.4f}")
        self.maeCard.setValue(f"{progress.mae_avg:.4f}")
        self.epochTimeCard.setValue(
            f"{progress.time_per_epoch:.1f}s",
            f"Total: {_format_duration(progress.total_time)}"
            if progress.total_time > 0
            else "",
        )

    @Slot(str)
    def _on_output(self, line: str):
        """Append a log line and auto-scroll."""
        self.logCard.logOutput.appendPlainText(line)
        sb = self.logCard.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(object)
    def _on_state_changed(self, state: ProcessState):
        """Update progress card when process state changes."""
        self.progressCard.set_state(state)

    @Slot(bool, str)
    def _on_completed(self, success: bool, message: str):
        """Show an info bar on training completion."""
        if success:
            InfoBar.success(
                title=self.tr("Training Complete"),
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )
        else:
            InfoBar.error(
                title=self.tr("Training Failed"),
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=5000,
            )

    # ── Public API ─────────────────────────────────────────────────

    def reset(self):
        """Reset dashboard to initial placeholder state."""
        self.progressCard.reset()
        self.trainLossCard.setValue("—")
        self.valLossCard.setValue("—")
        self.r2Card.setValue("—")
        self.lrCard.setValue("—")
        self.pearsonCard.setValue("—")
        self.gradNormCard.setValue("—")
        self.maeCard.setValue("—")
        self.epochTimeCard.setValue("—")
        self.logCard.logOutput.clear()
