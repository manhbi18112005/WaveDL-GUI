"""
WaveDL GUI - Metric Card Component

An enhanced statistics card with large value display, trend indicator
(delta arrow + percentage), and embedded sparkline. Designed for
the training dashboard's real-time monitoring grid.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, StrongBodyLabel, isDarkTheme, setFont

from .sparkline_widget import SparklineWidget


# ─── Theme-aware colors (shared palette) ──────────────────────────────────────


def _muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def _section_bg_color() -> QColor:
    return QColor(0, 0, 0, 6) if not isDarkTheme() else QColor(255, 255, 255, 6)


def _success_color() -> QColor:
    return QColor("#16a34a") if not isDarkTheme() else QColor("#4ade80")


def _error_color() -> QColor:
    return QColor("#dc2626") if not isDarkTheme() else QColor("#f87171")


# ─── Metric Card ──────────────────────────────────────────────────────────────


class MetricCard(QWidget):
    """Stat card with value, trend indicator, sparkline, and subtitle.

    Layout::

        ┌─────────────────────────────┐
        │  Caption Label              │
        │  VALUE   ▲ +2.3%           │
        │  ──── sparkline ────────    │
        │  subtitle (optional)        │
        └─────────────────────────────┘
    """

    def __init__(
        self,
        caption: str,
        line_color: QColor | None = None,
        higher_is_better: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._prev_value: float | None = None
        self._line_color = line_color
        self._higher_is_better = higher_is_better

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(2)

        # Caption row
        self.captionLabel = CaptionLabel(caption, self)
        self.captionLabel.setTextColor(_muted_text_color(), _muted_text_color())
        setFont(self.captionLabel, 10, QFont.Weight.DemiBold)
        v.addWidget(self.captionLabel)

        v.addSpacing(2)

        # Value + delta row
        valueRow = QHBoxLayout()
        valueRow.setSpacing(6)
        valueRow.setContentsMargins(0, 0, 0, 0)

        self.valueLabel = StrongBodyLabel("—", self)
        setFont(self.valueLabel, 20, QFont.Weight.DemiBold)
        valueRow.addWidget(self.valueLabel)

        self.deltaLabel = CaptionLabel("", self)
        setFont(self.deltaLabel, 10, QFont.Weight.DemiBold)
        valueRow.addWidget(self.deltaLabel)
        valueRow.addStretch()
        v.addLayout(valueRow)

        v.addSpacing(4)

        # Sparkline
        self.sparkline = SparklineWidget(
            max_points=60, line_color=line_color, parent=self
        )
        self.sparkline.setFixedHeight(32)
        v.addWidget(self.sparkline)

        # Subtitle
        self.subtitleLabel = CaptionLabel("", self)
        self.subtitleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        setFont(self.subtitleLabel, 10)
        self.subtitleLabel.hide()
        v.addWidget(self.subtitleLabel)

    # ── Public API ────────────────────────────────────────────────────

    def setValue(self, display: str, raw: float | None = None, subtitle: str = ""):
        """Update the displayed value, optionally track numeric trend.

        Args:
            display: Human-readable string shown as the main value.
            raw: Numeric value for sparkline and delta calculation.
            subtitle: Optional secondary text shown below the sparkline.
        """
        self.valueLabel.setText(display)

        if raw is not None:
            self.sparkline.addPoint(raw)
            self._update_delta(raw)
            self._prev_value = raw

        if subtitle:
            self.subtitleLabel.setText(subtitle)
            self.subtitleLabel.show()
        else:
            self.subtitleLabel.hide()

    def reset(self):
        """Reset to placeholder state."""
        self.valueLabel.setText("—")
        self.deltaLabel.setText("")
        self.subtitleLabel.hide()
        self.sparkline.clear()
        self._prev_value = None

    # ── Internals ─────────────────────────────────────────────────────

    def _update_delta(self, current: float):
        """Compute and display the delta from the previous value."""
        if self._prev_value is None or self._prev_value == 0.0:
            self.deltaLabel.setText("")
            return

        delta = current - self._prev_value
        pct = (delta / abs(self._prev_value)) * 100

        if abs(pct) < 0.01:
            self.deltaLabel.setText("")
            return

        arrow = "▲" if delta > 0 else "▼"
        self.deltaLabel.setText(f"{arrow} {abs(pct):.1f}%")

        # Color logic: for loss metrics (higher_is_better=False), down is green.
        # For accuracy metrics (higher_is_better=True), up is green.
        improving = (delta > 0) if self._higher_is_better else (delta < 0)
        if improving:
            self.deltaLabel.setTextColor(_success_color(), _success_color())
        else:
            self.deltaLabel.setTextColor(_error_color(), _error_color())

    # ── Custom painting ───────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(_section_bg_color())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 8, 8)
        p.end()
