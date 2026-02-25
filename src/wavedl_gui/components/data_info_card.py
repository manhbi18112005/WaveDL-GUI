"""
WaveDL GUI - Data Info Card Component

A premium card widget that displays training data file information with
clear visual hierarchy, grouped sections, and at-a-glance statistics.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon as FIF,
    IconWidget,
    SimpleCardWidget,
    StrongBodyLabel,
    isDarkTheme,
    setFont,
)

from ..common.utils import DataInfo
from ..components.statistic_widget import StatisticsWidget


# ─── Color palette ────────────────────────────────────────────────────────────


def _accent_color() -> QColor:
    """Primary accent (blue)."""
    return QColor("#3b82f6") if not isDarkTheme() else QColor("#60a5fa")


def _success_color() -> QColor:
    return QColor("#16a34a") if not isDarkTheme() else QColor("#4ade80")


def _error_color() -> QColor:
    return QColor("#dc2626") if not isDarkTheme() else QColor("#f87171")


def _muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def _subtle_border_color() -> QColor:
    return QColor(0, 0, 0, 18) if not isDarkTheme() else QColor(255, 255, 255, 18)


def _tag_bg_color() -> QColor:
    return QColor(0, 0, 0, 12) if not isDarkTheme() else QColor(255, 255, 255, 12)


def _section_bg_color() -> QColor:
    return QColor(0, 0, 0, 6) if not isDarkTheme() else QColor(255, 255, 255, 6)


# ─── Tiny reusable sub-widgets ────────────────────────────────────────────────


class _Tag(QWidget):
    """A small rounded tag/chip for quick-glance metadata (e.g. 'NPZ', '1D')."""

    def __init__(self, text: str = "", color: QColor | None = None, parent=None):
        super().__init__(parent)
        self._text = text
        self._color = color or _accent_color()
        self.setFixedHeight(22)
        self._update_width()

    def setText(self, text: str):
        self._text = text
        self._update_width()
        self.update()

    def setColor(self, color: QColor):
        self._color = color
        self.update()

    def _update_width(self):
        from PySide6.QtGui import QFontMetrics

        fm = QFontMetrics(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self.setFixedWidth(fm.horizontalAdvance(self._text) + 20)

    def paintEvent(self, _):
        if not self._text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QColor(self._color)
        bg.setAlpha(30 if not isDarkTheme() else 45)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)
        p.setPen(self._color)
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        p.drawText(self.rect(), Qt.AlignCenter, self._text)
        p.end()


class _Separator(QFrame):
    """A thin horizontal line separator."""

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


class _PropertyRow(QWidget):
    """Label → value row with optional monospaced value."""

    def __init__(self, label: str, mono: bool = False, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        self.label = CaptionLabel(label, self)
        self.label.setTextColor(_muted_text_color(), _muted_text_color())
        self.label.setFixedWidth(100)

        self.value = BodyLabel("", self)
        if mono:
            setFont(self.value, 12, QFont.Weight.Normal)
            self.value.setFont(QFont("Cascadia Code, Consolas, Monaco, monospace", 12))
        else:
            setFont(self.value, 12, QFont.Weight.Normal)

        h.addWidget(self.label)
        h.addWidget(self.value, 1)

    def setValue(self, v: str):
        self.value.setText(v)


class _SectionHeader(QWidget):
    """Section header with an icon and title."""

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


# ─── Main card ────────────────────────────────────────────────────────────────


class DataInfoCard(SimpleCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_info: DataInfo | None = None
        self._status_color = _success_color()
        self._init_ui()
        self.hide()

    # ── build UI ──────────────────────────────────────────────────────

    def _init_ui(self):
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # thin status accent bar at the top (painted in paintEvent)
        # -- we just leave 4 px for it in padding
        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        root.addLayout(content)

        # ── Row 1: filename + tags ────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(10)

        self.iconWidget = IconWidget(FIF.DOCUMENT, self)
        self.iconWidget.setFixedSize(20, 20)
        header.addWidget(self.iconWidget)

        titleCol = QVBoxLayout()
        titleCol.setSpacing(2)
        self.fileLabel = StrongBodyLabel("", self)
        setFont(self.fileLabel, 15, QFont.Weight.DemiBold)
        self.pathLabel = CaptionLabel("", self)
        self.pathLabel.setTextColor(_muted_text_color(), _muted_text_color())
        titleCol.addWidget(self.fileLabel)
        titleCol.addWidget(self.pathLabel)
        header.addLayout(titleCol, 1)

        # Tags
        self.formatTag = _Tag("", parent=self)
        self.dimTag = _Tag("", parent=self)
        header.addWidget(self.formatTag)
        header.addWidget(self.dimTag)
        content.addLayout(header)

        content.addSpacing(14)

        # ── Row 2: stat boxes ──────────────────────────────────────
        statsRow = QHBoxLayout()
        statsRow.setSpacing(10)
        self.samplesStat = StatisticsWidget("Samples", self)
        self.dimStat = StatisticsWidget("Data Type", self)
        self.outputsStat = StatisticsWidget("Outputs", self)
        self.sizeStat = StatisticsWidget("File Size", self)
        for s in (self.samplesStat, self.dimStat, self.outputsStat, self.sizeStat):
            statsRow.addWidget(s)
        content.addLayout(statsRow)

        content.addSpacing(14)
        content.addWidget(_Separator(self))
        content.addSpacing(10)

        # ── Row 3: Input tensor section ────────────────────────────
        content.addWidget(_SectionHeader(FIF.DOWNLOAD, "Input Tensor", self))
        content.addSpacing(4)
        self.inputShapeRow = _PropertyRow("Shape", mono=True, parent=self)
        self.inputDtypeRow = _PropertyRow("Dtype", mono=True, parent=self)
        self.inputKeyRow = _PropertyRow("Key", mono=True, parent=self)
        content.addWidget(self.inputShapeRow)
        content.addWidget(self.inputDtypeRow)
        content.addWidget(self.inputKeyRow)

        content.addSpacing(10)
        content.addWidget(_Separator(self))
        content.addSpacing(10)

        # ── Row 4: Output tensor section ───────────────────────────
        content.addWidget(_SectionHeader(FIF.UP, "Output Tensor", self))
        content.addSpacing(4)
        self.outputShapeRow = _PropertyRow("Shape", mono=True, parent=self)
        self.outputDtypeRow = _PropertyRow("Dtype", mono=True, parent=self)
        self.outputKeyRow = _PropertyRow("Key", mono=True, parent=self)
        content.addWidget(self.outputShapeRow)
        content.addWidget(self.outputDtypeRow)
        content.addWidget(self.outputKeyRow)

        # ── Error state (hidden by default) ────────────────────────
        content.addSpacing(8)
        self.errorLabel = CaptionLabel("", self)
        self.errorLabel.setWordWrap(True)
        self.errorLabel.setStyleSheet(
            "color: #dc2626;" if not isDarkTheme() else "color: #f87171;"
        )
        self.errorLabel.hide()
        content.addWidget(self.errorLabel)

        self._content_layout = content

    # ── ExpandLayout fix (same logic from previous fix) ───────────

    def _showCard(self):
        """Show the card and force the parent ExpandLayout to recalculate."""
        ideal_height = self.layout().sizeHint().height()
        self.setFixedHeight(ideal_height)
        self.show()
        QTimer.singleShot(0, self._updateParentLayout)

    def _updateParentLayout(self):
        parent = self.parentWidget()
        if parent is not None:
            parent.adjustSize()
            parent.updateGeometry()

    # ── Public API ────────────────────────────────────────────────

    def set_data_info(self, info: DataInfo):
        """Update the card with new data info."""
        self.data_info = info
        self.errorLabel.hide()

        if info.error:
            self._set_error_state(info)
            return

        self._set_success_state(info)

    def _set_error_state(self, info: DataInfo):
        self._status_color = _error_color()

        self.fileLabel.setText(self.tr("Error loading file"))
        self.pathLabel.setText(info.path)
        self.errorLabel.setText(info.error)
        self.errorLabel.show()

        self.formatTag.setText("")
        self.dimTag.setText("")
        for stat in (self.samplesStat, self.dimStat, self.outputsStat, self.sizeStat):
            stat.setValue("—")
        for row in (
            self.inputShapeRow,
            self.inputDtypeRow,
            self.inputKeyRow,
            self.outputShapeRow,
            self.outputDtypeRow,
            self.outputKeyRow,
        ):
            row.setValue("—")

        self._showCard()

    def _set_success_state(self, info: DataInfo):
        self._status_color = _success_color()

        filename = Path(info.path).name
        self.fileLabel.setText(filename)
        self.pathLabel.setText(info.path)

        # Tags
        self.formatTag.setText(info.format.upper())
        self.formatTag.setColor(_accent_color())
        self.dimTag.setText(info.dimensionality or "—")
        self.dimTag.setColor(
            QColor("#8b5cf6") if not isDarkTheme() else QColor("#a78bfa")
        )

        # Stat boxes
        self.samplesStat.setValue(f"{info.num_samples:,}")
        self.dimStat.setValue(info.dimensionality or "—")
        self.outputsStat.setValue(str(info.num_outputs))
        self.sizeStat.setValue(info.file_size_str)

        # Input tensor
        shape_str = (
            "(" + ", ".join(str(d) for d in info.input_shape) + ")"
            if info.input_shape
            else "—"
        )
        self.inputShapeRow.setValue(shape_str)
        self.inputDtypeRow.setValue(info.input_dtype or "—")
        self.inputKeyRow.setValue(info.input_key or "X")

        # Output tensor
        shape_str = (
            "(" + ", ".join(str(d) for d in info.output_shape) + ")"
            if info.output_shape
            else "—"
        )
        self.outputShapeRow.setValue(shape_str)
        self.outputDtypeRow.setValue(info.output_dtype or "—")
        self.outputKeyRow.setValue(info.output_key or "Y")

        self._showCard()

    def clear(self):
        """Clear the data info and hide the card."""
        self.data_info = None
        self.setFixedHeight(0)
        self.hide()
        QTimer.singleShot(0, self._updateParentLayout)

    # ── Custom painting ───────────────────────────────────────────

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

        # Top accent bar (3 px)
        accent_rect = self.rect().adjusted(1, 1, -1, 0)
        accent_rect.setHeight(4)
        path = QPainterPath()
        path.addRoundedRect(
            accent_rect.x(), accent_rect.y(), accent_rect.width(), r * 2, r, r
        )
        clip_rect = QPainterPath()
        clip_rect.addRect(
            accent_rect.x(), accent_rect.y(), accent_rect.width(), accent_rect.height()
        )
        path = path.intersected(clip_rect)
        p.setBrush(self._status_color)
        p.drawPath(path)

        p.end()
