"""
WaveDL GUI - Convergence Card Component

A premium Apple-like card that displays training convergence health indicators.
Shows overfitting detection, learning rate status, gradient health,
patience utilization, and convergence trajectory assessment.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon as FIF,
    IconWidget,
    SimpleCardWidget,
    StrongBodyLabel,
    isDarkTheme,
    setFont,
)


# ─── Color palette ────────────────────────────────────────────────────────────


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


def _bg_color() -> QColor:
    return QColor(255, 255, 255, 13 if isDarkTheme() else 170)


# ─── Separator ────────────────────────────────────────────────────────────────


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


# ─── Section header ──────────────────────────────────────────────────────────


class _SectionHeader(QWidget):
    """Section header with icon + uppercased title."""

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


# ─── Health indicator row ─────────────────────────────────────────────────────


class _HealthRow(QWidget):
    """A single health indicator: colored dot + label + value + status badge."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(8)

        self._dot_color = _muted_text_color()

        self.label = CaptionLabel(label, self)
        self.label.setTextColor(_muted_text_color(), _muted_text_color())
        self.label.setFixedWidth(120)
        h.addWidget(self.label)

        self.valueLabel = CaptionLabel("—", self)
        setFont(self.valueLabel, 12, QFont.Weight.DemiBold)
        h.addWidget(self.valueLabel, 1)

        self.statusLabel = CaptionLabel("", self)
        setFont(self.statusLabel, 10, QFont.Weight.DemiBold)
        h.addWidget(self.statusLabel)

    def setValue(self, value: str, status: str = "", level: str = "neutral"):
        """Update value and status indicator.

        Args:
            value: Display value string.
            status: Short status text (e.g. "Healthy", "Warning").
            level: One of "good", "warning", "error", "neutral".
        """
        self.valueLabel.setText(value)
        self.statusLabel.setText(status)

        color_map = {
            "good": _success_color(),
            "warning": _warning_color(),
            "error": _error_color(),
            "neutral": _muted_text_color(),
        }
        color = color_map.get(level, _muted_text_color())
        self._dot_color = color
        self.statusLabel.setTextColor(color, color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Colored dot
        p.setPen(Qt.NoPen)
        p.setBrush(self._dot_color)
        p.drawEllipse(QRectF(0, self.height() / 2 - 3, 6, 6))
        p.end()


# ─── Convergence Card ────────────────────────────────────────────────────────


class ConvergenceCard(SimpleCardWidget):
    """Premium card analyzing training convergence health with multiple indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBorderRadius(10)
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        root.addLayout(content)

        # Header
        header = QHBoxLayout()
        header.setSpacing(10)

        ic = IconWidget(FIF.HEART, self)
        ic.setFixedSize(20, 20)
        header.addWidget(ic)

        titleCol = QVBoxLayout()
        titleCol.setSpacing(2)
        titleLabel = StrongBodyLabel(self.tr("Convergence Health"), self)
        setFont(titleLabel, 15, QFont.Weight.DemiBold)
        self.subtitleLabel = CaptionLabel(
            self.tr("Real-time training diagnostics"), self
        )
        self.subtitleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        titleCol.addWidget(titleLabel)
        titleCol.addWidget(self.subtitleLabel)
        header.addLayout(titleCol, 1)

        content.addLayout(header)
        content.addSpacing(14)
        content.addWidget(_Separator(self))
        content.addSpacing(10)

        # ── Health indicators ──
        content.addWidget(_SectionHeader(FIF.CARE_RIGHT_SOLID, "Training Health", self))
        content.addSpacing(4)

        self.overfitRow = _HealthRow(self.tr("Overfitting"), self)
        self.lrRow = _HealthRow(self.tr("Learning Rate"), self)
        self.gradientRow = _HealthRow(self.tr("Gradient Health"), self)
        self.patienceRow = _HealthRow(self.tr("Patience Used"), self)
        self.convergenceRow = _HealthRow(self.tr("Convergence"), self)

        for row in (
            self.overfitRow,
            self.lrRow,
            self.gradientRow,
            self.patienceRow,
            self.convergenceRow,
        ):
            content.addWidget(row)

    # ── Public API ────────────────────────────────────────────────────

    def update_health(
        self,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        learning_rate: float,
        grad_norm: float,
        patience_counter: int,
        max_patience: int,
        epoch: int,
        total_epochs: int,
    ):
        """Analyze and display convergence health indicators."""
        # ── Overfitting detection ──
        if train_loss > 0:
            gap = (val_loss - train_loss) / train_loss
            gap_pct = gap * 100
            if gap_pct > 50:
                self.overfitRow.setValue(f"Gap: {gap_pct:.1f}%", "Severe", "error")
            elif gap_pct > 20:
                self.overfitRow.setValue(f"Gap: {gap_pct:.1f}%", "Moderate", "warning")
            elif gap_pct > 0:
                self.overfitRow.setValue(f"Gap: {gap_pct:.1f}%", "Healthy", "good")
            else:
                self.overfitRow.setValue(
                    f"Gap: {gap_pct:.1f}%", "Underfitting", "warning"
                )
        else:
            self.overfitRow.setValue("—", "", "neutral")

        # ── Learning rate status ──
        if learning_rate > 0:
            lr_str = f"{learning_rate:.2e}"
            if learning_rate < 1e-7:
                self.lrRow.setValue(lr_str, "Very Low", "warning")
            elif learning_rate > 1e-2:
                self.lrRow.setValue(lr_str, "High", "warning")
            else:
                self.lrRow.setValue(lr_str, "Normal", "good")
        else:
            self.lrRow.setValue("—", "", "neutral")

        # ── Gradient health ──
        if grad_norm > 0:
            gn_str = f"{grad_norm:.4f}"
            if grad_norm > 100:
                self.gradientRow.setValue(gn_str, "Exploding", "error")
            elif grad_norm > 10:
                self.gradientRow.setValue(gn_str, "High", "warning")
            elif grad_norm < 1e-6:
                self.gradientRow.setValue(gn_str, "Vanishing", "error")
            else:
                self.gradientRow.setValue(gn_str, "Stable", "good")
        else:
            self.gradientRow.setValue("—", "", "neutral")

        # ── Patience utilization ──
        if max_patience > 0:
            pct_used = (patience_counter / max_patience) * 100
            pct_str = f"{patience_counter}/{max_patience} ({pct_used:.0f}%)"
            if pct_used > 80:
                self.patienceRow.setValue(pct_str, "Critical", "error")
            elif pct_used > 50:
                self.patienceRow.setValue(pct_str, "Elevated", "warning")
            else:
                self.patienceRow.setValue(pct_str, "OK", "good")
        else:
            self.patienceRow.setValue("—", "", "neutral")

        # ── Convergence trajectory ──
        if total_epochs > 0 and epoch > 0:
            improvement = 0.0
            improvement = 0.0
            if best_val_loss < float("inf") and val_loss > 0:
                improvement = ((val_loss - best_val_loss) / val_loss) * 100

            if improvement < 1:
                self.convergenceRow.setValue(
                    f"Epoch {epoch}/{total_epochs}", "Converging", "good"
                )
            elif improvement < 10:
                self.convergenceRow.setValue(
                    f"Epoch {epoch}/{total_epochs}", "Plateau", "warning"
                )
            else:
                self.convergenceRow.setValue(
                    f"Epoch {epoch}/{total_epochs}", "Diverging", "error"
                )
        else:
            self.convergenceRow.setValue("—", "", "neutral")

    def reset(self):
        """Reset all indicators to placeholder state."""
        for row in (
            self.overfitRow,
            self.lrRow,
            self.gradientRow,
            self.patienceRow,
            self.convergenceRow,
        ):
            row.setValue("—", "", "neutral")

    # ── Custom painting ───────────────────────────────────────────────

    def _normalBackgroundColor(self):
        return _bg_color()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.borderRadius

        p.setBrush(self._normalBackgroundColor())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

        # Top accent bar — green for health theme
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
        clip = QPainterPath()
        clip.addRect(
            accent_rect.x(),
            accent_rect.y(),
            accent_rect.width(),
            accent_rect.height(),
        )
        path = path.intersected(clip)
        p.setBrush(_success_color())
        p.drawPath(path)

        p.end()


# ─── Per-Parameter MAE Card ──────────────────────────────────────────────────


class PerParamCard(SimpleCardWidget):
    """Card showing per-parameter MAE breakdown with horizontal bar indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBorderRadius(10)
        self._bars: list[tuple[str, float]] = []
        self._max_val: float = 1.0
        self._init_ui()

    def _init_ui(self):
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        self._root.addLayout(content)
        self._content = content

        # Header
        header = QHBoxLayout()
        header.setSpacing(10)

        ic = IconWidget(FIF.MARKET, self)
        ic.setFixedSize(20, 20)
        header.addWidget(ic)

        titleCol = QVBoxLayout()
        titleCol.setSpacing(2)
        titleLabel = StrongBodyLabel(self.tr("Per-Parameter MAE"), self)
        setFont(titleLabel, 15, QFont.Weight.DemiBold)
        self.subtitleLabel = CaptionLabel(
            self.tr("Error breakdown by output parameter"), self
        )
        self.subtitleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        titleCol.addWidget(titleLabel)
        titleCol.addWidget(self.subtitleLabel)
        header.addLayout(titleCol, 1)

        content.addLayout(header)
        content.addSpacing(14)

        # Bar area (will be painted)
        self._bar_area = _BarChartArea(self)
        self._bar_area.setMinimumHeight(40)
        content.addWidget(self._bar_area)

    def setValues(self, mae_per_param: list[float]):
        """Update the bar chart with per-parameter MAE values."""
        if not mae_per_param:
            self._bar_area.setData([])
            return

        bars = []
        for i, val in enumerate(mae_per_param):
            bars.append((f"Param {i + 1}", val))

        self._bar_area.setData(bars)

        # Adjust height
        bar_height = max(120, 28 * len(bars) + 20)
        self._bar_area.setMinimumHeight(bar_height)

    def reset(self):
        """Clear the bar chart."""
        self._bar_area.setData([])

    # ── Custom painting ───────────────────────────────────────────────

    def _normalBackgroundColor(self):
        return _bg_color()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.borderRadius

        p.setBrush(self._normalBackgroundColor())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

        # Top accent bar — orange/amber theme
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
        clip = QPainterPath()
        clip.addRect(
            accent_rect.x(),
            accent_rect.y(),
            accent_rect.width(),
            accent_rect.height(),
        )
        path = path.intersected(clip)
        p.setBrush(_warning_color())
        p.drawPath(path)

        p.end()


# ─── Bar chart area widget ───────────────────────────────────────────────────


class _BarChartArea(QWidget):
    """Custom-painted horizontal bar chart for per-parameter MAE."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list[tuple[str, float]] = []

    def setData(self, data: list[tuple[str, float]]):
        self._data = data
        self.update()

    def paintEvent(self, _):
        if not self._data:
            p = QPainter(self)
            p.setPen(_muted_text_color())
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "No per-parameter data yet")
            p.end()
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        max_val = max(v for _, v in self._data) if self._data else 1.0
        if max_val == 0:
            max_val = 1.0

        label_width = 70
        bar_height = 18
        row_height = 26
        bar_x = label_width + 8
        bar_max_w = self.width() - bar_x - 70  # leave room for value label

        # Gradient colors for bars (blue to orange based on relative magnitude)
        for i, (name, val) in enumerate(self._data):
            y = i * row_height + 4

            # Label
            p.setPen(_muted_text_color())
            p.setFont(QFont("Segoe UI", 10))
            label_rect = QRectF(0, y, label_width, bar_height)
            p.drawText(label_rect, Qt.AlignRight | Qt.AlignVCenter, name)

            # Bar background
            bg_rect = QRectF(bar_x, y + 2, bar_max_w, bar_height - 4)
            bg_c = _section_bg_color()
            p.setPen(Qt.NoPen)
            p.setBrush(bg_c)
            p.drawRoundedRect(bg_rect, 4, 4)

            # Bar fill
            fill_frac = min(val / max_val, 1.0) if max_val > 0 else 0
            fill_w = max(fill_frac * bar_max_w, 2)
            fill_rect = QRectF(bar_x, y + 2, fill_w, bar_height - 4)

            # Color: interpolate blue → orange based on relative error
            if fill_frac < 0.5:
                bar_color = _accent_color()
            elif fill_frac < 0.8:
                bar_color = _warning_color()
            else:
                bar_color = _error_color()

            bar_c = QColor(bar_color)
            bar_c.setAlpha(180)
            p.setBrush(bar_c)
            p.drawRoundedRect(fill_rect, 4, 4)

            # Value label
            p.setPen(bar_color)
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            val_rect = QRectF(bar_x + bar_max_w + 6, y, 60, bar_height)
            p.drawText(val_rect, Qt.AlignLeft | Qt.AlignVCenter, f"{val:.4f}")

        p.end()
