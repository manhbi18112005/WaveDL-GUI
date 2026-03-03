"""
WaveDL GUI - Sparkline Widget

A compact inline sparkline chart drawn with QPainter using smooth
Catmull-Rom splines and gradient fill. Designed for embedding inside
stat cards and metric displays.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QWidget
from qfluentwidgets import isDarkTheme


class SparklineWidget(QWidget):
    """Mini inline chart widget with smooth Catmull-Rom curve and gradient fill."""

    def __init__(
        self,
        max_points: int = 60,
        line_color: QColor | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._data: deque[float] = deque(maxlen=max_points)
        self._line_color = line_color
        self._show_dot = True
        self.setFixedHeight(36)
        self.setMinimumWidth(60)

    def setLineColor(self, color: QColor):
        """Update the sparkline color."""
        self._line_color = color
        self.update()

    def addPoint(self, value: float):
        """Append a data point and trigger repaint."""
        self._data.append(value)
        self.update()

    def clear(self):
        """Remove all data points."""
        self._data.clear()
        self.update()

    def _default_color(self) -> QColor:
        return QColor("#3b82f6") if not isDarkTheme() else QColor("#60a5fa")

    def paintEvent(self, _):
        if len(self._data) < 2:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        color = self._line_color or self._default_color()
        rect = self.rect().adjusted(4, 4, -4, -4)
        w, h = float(rect.width()), float(rect.height())

        data = list(self._data)
        mn, mx = min(data), max(data)
        rng = mx - mn if mx != mn else 1.0

        # Map data → pixel coordinates
        points: list[QPointF] = []
        for i, v in enumerate(data):
            x = rect.x() + (i / (len(data) - 1)) * w
            y = rect.y() + h - ((v - mn) / rng) * h
            points.append(QPointF(x, y))

        # Build smooth curve via Catmull-Rom → cubic Bézier conversion
        path = QPainterPath()
        path.moveTo(points[0])

        for i in range(len(points) - 1):
            pt0 = points[max(i - 1, 0)]
            pt1 = points[i]
            pt2 = points[min(i + 1, len(points) - 1)]
            pt3 = points[min(i + 2, len(points) - 1)]

            tension = 0.35
            cp1 = QPointF(
                pt1.x() + (pt2.x() - pt0.x()) * tension,
                pt1.y() + (pt2.y() - pt0.y()) * tension,
            )
            cp2 = QPointF(
                pt2.x() - (pt3.x() - pt1.x()) * tension,
                pt2.y() - (pt3.y() - pt1.y()) * tension,
            )
            path.cubicTo(cp1, cp2, pt2)

        # Gradient fill beneath the curve
        fill_path = QPainterPath(path)
        fill_path.lineTo(points[-1].x(), rect.bottom())
        fill_path.lineTo(points[0].x(), rect.bottom())
        fill_path.closeSubpath()

        grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
        fill_top = QColor(color)
        fill_top.setAlpha(40 if not isDarkTheme() else 30)
        grad.setColorAt(0, fill_top)
        fill_bottom = QColor(color)
        fill_bottom.setAlpha(0)
        grad.setColorAt(1, fill_bottom)
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawPath(fill_path)

        # Draw the curve line
        pen = QPen(color, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        # Current-value dot at the trailing edge
        if self._show_dot and points:
            last = points[-1]
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(last, 2.5, 2.5)

        p.end()
