"""
WaveDL GUI - Loss Chart Card Component

A premium dual-axis loss curve card drawn entirely with QPainter.
Shows train/val loss history with smooth Catmull-Rom splines,
gradient fills, axis labels, a legend, and a best-loss marker.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QVBoxLayout
from qfluentwidgets import CaptionLabel, SimpleCardWidget, isDarkTheme, setFont


# ─── Color palette ────────────────────────────────────────────────────────────


def _accent_color() -> QColor:
    return QColor("#3b82f6") if not isDarkTheme() else QColor("#60a5fa")


def _success_color() -> QColor:
    return QColor("#16a34a") if not isDarkTheme() else QColor("#4ade80")


def _warning_color() -> QColor:
    return QColor("#f59e0b") if not isDarkTheme() else QColor("#fbbf24")


def _muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def _grid_color() -> QColor:
    return QColor(0, 0, 0, 15) if not isDarkTheme() else QColor(255, 255, 255, 12)


def _bg_color() -> QColor:
    return QColor(255, 255, 255, 13 if isDarkTheme() else 170)


# ─── Loss Chart Card ─────────────────────────────────────────────────────────


class LossChartCard(SimpleCardWidget):
    """Premium dual-line loss chart with gradient fills and best-loss marker."""

    MAX_POINTS = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self._train_data: deque[float] = deque(maxlen=self.MAX_POINTS)
        self._val_data: deque[float] = deque(maxlen=self.MAX_POINTS)
        self._best_val: float = float("inf")
        self._best_val_epoch: int = 0
        self._current_epoch: int = 0
        self.setBorderRadius(10)
        self.setMinimumHeight(280)
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 8)
        root.setSpacing(0)

        self.titleLabel = CaptionLabel("LOSS CURVE", self)
        self.titleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        setFont(self.titleLabel, 10, QFont.Weight.Bold)
        root.addWidget(self.titleLabel)

        root.addStretch()

    # ── Public API ────────────────────────────────────────────────────

    def addPoint(self, train_loss: float, val_loss: float, epoch: int = 0):
        """Append a train/val loss pair and trigger repaint."""
        self._train_data.append(train_loss)
        self._val_data.append(val_loss)
        self._current_epoch = epoch
        if val_loss < self._best_val:
            self._best_val = val_loss
            self._best_val_epoch = epoch
        self.update()

    def clear(self):
        """Remove all data and reset."""
        self._train_data.clear()
        self._val_data.clear()
        self._best_val = float("inf")
        self._best_val_epoch = 0
        self._current_epoch = 0
        self.update()

    # ── Painting ──────────────────────────────────────────────────────

    def _normalBackgroundColor(self):
        return _bg_color()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.borderRadius

        # Card background
        p.setBrush(self._normalBackgroundColor())
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

        # Top accent bar
        accent_rect = self.rect().adjusted(1, 1, -1, 0)
        accent_rect.setHeight(4)
        bar_path = QPainterPath()
        bar_path.addRoundedRect(
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
        bar_path = bar_path.intersected(clip)
        p.setBrush(QColor("#8b5cf6") if not isDarkTheme() else QColor("#a78bfa"))
        p.drawPath(bar_path)

        # Chart area
        chart_rect = QRectF(
            self.rect().x() + 60,
            self.rect().y() + 40,
            self.rect().width() - 80,
            self.rect().height() - 88,
        )

        if len(self._train_data) >= 2:
            self._draw_chart(p, chart_rect)
        else:
            self._draw_placeholder(p, chart_rect)

        p.end()

    def _draw_placeholder(self, p: QPainter, rect: QRectF):
        """Show placeholder text when there's no data."""
        p.setPen(_muted_text_color())
        p.setFont(QFont("Segoe UI", 12))
        p.drawText(rect, Qt.AlignCenter, "Loss curve will appear here")

    def _draw_chart(self, p: QPainter, rect: QRectF):
        """Draw the full chart with axes, gridlines, curves, legend."""
        train = list(self._train_data)
        val = list(self._val_data)
        n = len(train)

        all_vals = train + val
        y_min = min(all_vals)
        y_max = max(all_vals)
        if y_max == y_min:
            y_max = y_min + 1.0

        # Add 5% padding
        padding = (y_max - y_min) * 0.05
        y_min -= padding
        y_max += padding

        # ── Grid lines & Y-axis labels ──
        grid_pen = QPen(_grid_color(), 1, Qt.SolidLine)
        label_font = QFont("Segoe UI", 9)
        num_grid = 5

        for i in range(num_grid + 1):
            frac = i / num_grid
            y = rect.bottom() - frac * rect.height()
            val_at = y_min + frac * (y_max - y_min)

            p.setPen(grid_pen)
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

            p.setPen(_muted_text_color())
            p.setFont(label_font)
            label = f"{val_at:.4f}" if val_at < 1 else f"{val_at:.2f}"
            label_rect = QRectF(rect.left() - 56, y - 8, 50, 16)
            p.drawText(label_rect, Qt.AlignRight | Qt.AlignVCenter, label)

        # ── X-axis labels ──
        x_ticks = min(6, n)
        if x_ticks > 1:
            for i in range(x_ticks):
                idx = int(i * (n - 1) / (x_ticks - 1))
                x = rect.left() + (idx / (n - 1)) * rect.width()
                epoch_num = max(1, idx + 1)
                if self._current_epoch > 0:
                    epoch_num = max(1, self._current_epoch - n + 1 + idx)

                p.setPen(_muted_text_color())
                p.setFont(label_font)
                label_rect = QRectF(x - 20, rect.bottom() + 4, 40, 16)
                p.drawText(label_rect, Qt.AlignCenter, str(epoch_num))

        # ── Helper to map data → points ──
        def to_points(data: list[float]) -> list[QPointF]:
            pts = []
            for i, v in enumerate(data):
                x = rect.left() + (i / (len(data) - 1)) * rect.width()
                y = rect.bottom() - ((v - y_min) / (y_max - y_min)) * rect.height()
                pts.append(QPointF(x, y))
            return pts

        def smooth_path(pts: list[QPointF]) -> QPainterPath:
            path = QPainterPath()
            path.moveTo(pts[0])
            tension = 0.3
            for i in range(len(pts) - 1):
                p0 = pts[max(i - 1, 0)]
                p1 = pts[i]
                p2 = pts[min(i + 1, len(pts) - 1)]
                p3 = pts[min(i + 2, len(pts) - 1)]
                cp1 = QPointF(
                    p1.x() + (p2.x() - p0.x()) * tension,
                    p1.y() + (p2.y() - p0.y()) * tension,
                )
                cp2 = QPointF(
                    p2.x() - (p3.x() - p1.x()) * tension,
                    p2.y() - (p3.y() - p1.y()) * tension,
                )
                path.cubicTo(cp1, cp2, p2)
            return path

        def draw_gradient_fill(
            painter: QPainter,
            path: QPainterPath,
            pts: list[QPointF],
            color: QColor,
        ):
            fill = QPainterPath(path)
            fill.lineTo(pts[-1].x(), rect.bottom())
            fill.lineTo(pts[0].x(), rect.bottom())
            fill.closeSubpath()

            grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
            top_c = QColor(color)
            top_c.setAlpha(35 if not isDarkTheme() else 25)
            grad.setColorAt(0, top_c)
            bot_c = QColor(color)
            bot_c.setAlpha(0)
            grad.setColorAt(1, bot_c)
            painter.setBrush(grad)
            painter.setPen(Qt.NoPen)
            painter.drawPath(fill)

        # ── Draw train loss ──
        train_color = _accent_color()
        train_pts = to_points(train)
        train_path = smooth_path(train_pts)
        draw_gradient_fill(p, train_path, train_pts, train_color)
        p.setPen(QPen(train_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(train_path)

        # ── Draw val loss ──
        val_color = _warning_color()
        val_pts = to_points(val)
        val_path = smooth_path(val_pts)
        draw_gradient_fill(p, val_path, val_pts, val_color)
        p.setPen(QPen(val_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(val_path)

        # ── Best val marker ──
        if self._best_val < float("inf") and len(val) > 0:
            best_idx = None
            for i, v in enumerate(val):
                if abs(v - self._best_val) < 1e-10:
                    best_idx = i
            if best_idx is not None:
                bx = rect.left() + (best_idx / (len(val) - 1)) * rect.width()
                by = (
                    rect.bottom()
                    - ((self._best_val - y_min) / (y_max - y_min)) * rect.height()
                )

                # Dashed horizontal line
                dash_pen = QPen(_success_color(), 1, Qt.DashLine)
                p.setPen(dash_pen)
                p.drawLine(QPointF(rect.left(), by), QPointF(rect.right(), by))

                # Diamond marker
                p.setPen(Qt.NoPen)
                p.setBrush(_success_color())
                diamond = QPainterPath()
                s = 5.0
                diamond.moveTo(bx, by - s)
                diamond.lineTo(bx + s, by)
                diamond.lineTo(bx, by + s)
                diamond.lineTo(bx - s, by)
                diamond.closeSubpath()
                p.drawPath(diamond)

        # ── Current-value dots ──
        p.setPen(Qt.NoPen)
        if train_pts:
            p.setBrush(train_color)
            p.drawEllipse(train_pts[-1], 3.5, 3.5)
        if val_pts:
            p.setBrush(val_color)
            p.drawEllipse(val_pts[-1], 3.5, 3.5)

        # ── Legend ──
        self._draw_legend(p, rect)

    def _draw_legend(self, p: QPainter, chart_rect: QRectF):
        """Draw the legend at the bottom-right."""
        legend_font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        p.setFont(legend_font)

        items = [
            ("Train Loss", _accent_color()),
            ("Val Loss", _warning_color()),
            ("Best Val", _success_color()),
        ]

        x = chart_rect.right() - 200
        y = chart_rect.bottom() + 22

        for label, color in items:
            # Color swatch
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, y, 10, 10), 2, 2)

            # Label
            p.setPen(color)
            p.drawText(
                QRectF(x + 14, y - 2, 60, 14), Qt.AlignLeft | Qt.AlignVCenter, label
            )
            x += 72
