"""
WaveDL GUI - Training Dashboard Interface
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
from ..components.convergence_card import ConvergenceCard, PerParamCard
from ..components.loss_chart_card import LossChartCard
from ..components.metric_card import MetricCard
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


# ─── Epoch summary strip ─────────────────────────────────────────────────────


class _EpochSummaryStrip(QWidget):
    """Compact horizontal strip showing key epoch-over-epoch summaries.

    Displays: best val loss, best R², total parameters updated,
    improvement rate — all in a single tinted row.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(16, 6, 16, 6)
        h.setSpacing(24)

        self._items: dict[str, CaptionLabel] = {}
        for key, label in [
            ("best_val", "Best Val Loss"),
            ("best_r2", "Best R\u00b2"),
            ("improvement", "Improvement"),
            ("throughput", "Throughput"),
        ]:
            lbl = CaptionLabel(f"{label}: —", self)
            lbl.setTextColor(_muted_text_color(), _muted_text_color())
            setFont(lbl, 10, QFont.Weight.DemiBold)
            h.addWidget(lbl)
            self._items[key] = lbl

        h.addStretch()

    def update_summary(
        self,
        best_val: float,
        best_r2: float,
        improvement_pct: float,
        samples_per_sec: float,
    ):
        """Update all summary values."""
        if best_val < float("inf"):
            self._items["best_val"].setText(f"Best Val Loss: {best_val:.6f}")
        if best_r2 > -float("inf"):
            self._items["best_r2"].setText(f"Best R\u00b2: {best_r2:.4f}")
        if improvement_pct != 0:
            arrow = "▲" if improvement_pct > 0 else "▼"
            self._items["improvement"].setText(
                f"Improvement: {arrow} {abs(improvement_pct):.2f}%"
            )
        if samples_per_sec > 0:
            self._items["throughput"].setText(
                f"Throughput: {samples_per_sec:.0f} samples/s"
            )

    def reset(self):
        """Reset to placeholder state."""
        labels = {
            "best_val": "Best Val Loss",
            "best_r2": "Best R\u00b2",
            "improvement": "Improvement",
            "throughput": "Throughput",
        }
        for key, lbl_text in labels.items():
            self._items[key].setText(f"{lbl_text}: —")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(_section_bg_color())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)
        p.end()


# ─── Dashboard interface ──────────────────────────────────────────────────────


class DashboardInterface(ScrollArea):
    """Comprehensive training dashboard with real-time monitoring metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollWidget = QWidget()
        self.expandLayout = QVBoxLayout(self.scrollWidget)

        # Track best-ever values for summary strip
        self._best_r2: float = -float("inf")
        self._prev_val_loss: float = float("inf")

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

        # ── Epoch summary strip ────────────────────────────────────
        self.summaryStrip = _EpochSummaryStrip(self.scrollWidget)

        # ── Metrics row 1 (with sparklines) ────────────────────────
        metricsWidget = QWidget(self.scrollWidget)
        metricsLayout = QHBoxLayout(metricsWidget)
        metricsLayout.setContentsMargins(0, 0, 0, 0)
        metricsLayout.setSpacing(10)

        self.trainLossCard = MetricCard(
            self.tr("Train Loss"),
            line_color=_accent_color(),
            parent=metricsWidget,
        )
        self.valLossCard = MetricCard(
            self.tr("Val Loss"),
            line_color=_warning_color(),
            parent=metricsWidget,
        )
        self.r2Card = MetricCard(
            self.tr("R\u00b2 Score"),
            line_color=_success_color(),
            higher_is_better=True,
            parent=metricsWidget,
        )
        self.lrCard = MetricCard(
            self.tr("Learning Rate"),
            line_color=QColor("#8b5cf6") if not isDarkTheme() else QColor("#a78bfa"),
            parent=metricsWidget,
        )

        for card in (self.trainLossCard, self.valLossCard, self.r2Card, self.lrCard):
            metricsLayout.addWidget(card)

        # ── Metrics row 2 (with sparklines) ────────────────────────
        metricsWidget2 = QWidget(self.scrollWidget)
        metricsLayout2 = QHBoxLayout(metricsWidget2)
        metricsLayout2.setContentsMargins(0, 0, 0, 0)
        metricsLayout2.setSpacing(10)

        self.pearsonCard = MetricCard(
            self.tr("Pearson"),
            line_color=_accent_color(),
            higher_is_better=True,
            parent=metricsWidget2,
        )
        self.gradNormCard = MetricCard(
            self.tr("Grad Norm"),
            line_color=_error_color(),
            parent=metricsWidget2,
        )
        self.maeCard = MetricCard(
            self.tr("MAE Avg"), line_color=_warning_color(), parent=metricsWidget2
        )
        self.epochTimeCard = MetricCard(
            self.tr("Epoch Time"),
            line_color=_muted_text_color(),
            parent=metricsWidget2,
        )

        for card in (
            self.pearsonCard,
            self.gradNormCard,
            self.maeCard,
            self.epochTimeCard,
        ):
            metricsLayout2.addWidget(card)

        # ── Loss chart card ────────────────────────────────────────
        self.lossChartCard = LossChartCard(self.scrollWidget)

        # ── Analysis row: Convergence + Per-Param side by side ─────
        analysisWidget = QWidget(self.scrollWidget)
        analysisLayout = QHBoxLayout(analysisWidget)
        analysisLayout.setContentsMargins(0, 0, 0, 0)
        analysisLayout.setSpacing(15)

        self.convergenceCard = ConvergenceCard(analysisWidget)
        self.perParamCard = PerParamCard(analysisWidget)

        analysisLayout.addWidget(self.convergenceCard, 1)
        analysisLayout.addWidget(self.perParamCard, 1)

        # ── Log card ───────────────────────────────────────────────
        self.logCard = _LogCard(self.scrollWidget)

        # ── Assemble layout ────────────────────────────────────────
        self.expandLayout.setSpacing(12)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.progressCard)
        self.expandLayout.addWidget(self.summaryStrip)
        self.expandLayout.addWidget(metricsWidget)
        self.expandLayout.addWidget(metricsWidget2)
        self.expandLayout.addWidget(self.lossChartCard)
        self.expandLayout.addWidget(analysisWidget)
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

        # ── MetricCards (with sparklines + trend deltas) ──
        self.trainLossCard.setValue(
            f"{progress.train_loss:.6f}",
            raw=progress.train_loss,
        )
        self.valLossCard.setValue(
            f"{progress.val_loss:.6f}",
            raw=progress.val_loss,
            subtitle=f"Best: {progress.best_val_loss:.6f}",
        )
        self.r2Card.setValue(
            f"{progress.r2_score:.4f}",
            raw=progress.r2_score,
        )
        self.lrCard.setValue(
            f"{progress.learning_rate:.2e}",
            raw=progress.learning_rate,
        )
        self.pearsonCard.setValue(
            f"{progress.pearson:.4f}",
            raw=progress.pearson,
        )
        self.gradNormCard.setValue(
            f"{progress.grad_norm:.4f}",
            raw=progress.grad_norm,
        )
        self.maeCard.setValue(
            f"{progress.mae_avg:.4f}",
            raw=progress.mae_avg,
        )
        self.epochTimeCard.setValue(
            f"{progress.time_per_epoch:.1f}s",
            raw=progress.time_per_epoch,
            subtitle=f"Total: {_format_duration(progress.total_time)}"
            if progress.total_time > 0
            else "",
        )

        # ── Loss chart ──
        self.lossChartCard.addPoint(
            progress.train_loss, progress.val_loss, progress.epoch
        )

        # ── Convergence health ──
        self.convergenceCard.update_health(
            train_loss=progress.train_loss,
            val_loss=progress.val_loss,
            best_val_loss=progress.best_val_loss,
            learning_rate=progress.learning_rate,
            grad_norm=progress.grad_norm,
            patience_counter=progress.patience_counter,
            max_patience=progress.max_patience,
            epoch=progress.epoch,
            total_epochs=progress.total_epochs,
        )

        # ── Per-parameter MAE ──
        if progress.mae_per_param:
            self.perParamCard.setValues(progress.mae_per_param)

        # ── Summary strip ──
        if progress.r2_score > self._best_r2:
            self._best_r2 = progress.r2_score

        improvement_pct = 0.0
        if self._prev_val_loss < float("inf") and self._prev_val_loss > 0:
            improvement_pct = (
                (self._prev_val_loss - progress.val_loss) / self._prev_val_loss
            ) * 100
        self._prev_val_loss = progress.val_loss

        # Estimate throughput (rough: batch_size isn't available, use epoch time)
        throughput = 0.0
        if progress.time_per_epoch > 0:
            throughput = 1.0 / progress.time_per_epoch  # epochs/sec as proxy

        self.summaryStrip.update_summary(
            best_val=progress.best_val_loss,
            best_r2=self._best_r2,
            improvement_pct=improvement_pct,
            samples_per_sec=throughput,
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
        self.summaryStrip.reset()

        self.trainLossCard.reset()
        self.valLossCard.reset()
        self.r2Card.reset()
        self.lrCard.reset()
        self.pearsonCard.reset()
        self.gradNormCard.reset()
        self.maeCard.reset()
        self.epochTimeCard.reset()

        self.lossChartCard.clear()
        self.convergenceCard.reset()
        self.perParamCard.reset()

        self.logCard.logOutput.clear()

        self._best_r2 = -float("inf")
        self._prev_val_loss = float("inf")
