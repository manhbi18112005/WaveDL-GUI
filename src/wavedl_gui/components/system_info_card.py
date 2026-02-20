"""
WaveDL GUI - System Information Card

A premium Apple-like card displaying hardware & software environment info.
Matches the visual design system of DataInfoCard and ProgressCard.
"""

from PySide6.QtCore import Qt, QTimer
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

from ..common.utils import (
    GPUInfo,
    check_pytorch_installation,
    detect_gpus,
    get_cpu_name,
    get_os_display_name,
    get_system_memory_mb,
)
from ..components.statistic_widget import StatisticsWidget


# ─── Color palette (shared with dashboard) ─────────────────────────────────────


def _accent_color() -> QColor:
    return QColor("#0891b2") if not isDarkTheme() else QColor("#22d3ee")  # Teal/Cyan


def _muted_text_color() -> QColor:
    return QColor(110, 110, 110) if not isDarkTheme() else QColor(160, 160, 160)


def _subtle_border_color() -> QColor:
    return QColor(0, 0, 0, 18) if not isDarkTheme() else QColor(255, 255, 255, 18)


def _section_bg_color() -> QColor:
    return QColor(0, 0, 0, 6) if not isDarkTheme() else QColor(255, 255, 255, 6)


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


class _Tag(QWidget):
    """Small rounded chip/badge for quick-glance metadata."""

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


class _PropertyRow(QWidget):
    """Label → value row for environment details."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 3, 0, 3)
        h.setSpacing(12)

        self.label = CaptionLabel(label, self)
        self.label.setTextColor(_muted_text_color(), _muted_text_color())
        self.label.setFixedWidth(110)

        self.value = CaptionLabel("—", self)
        setFont(self.value, 12, QFont.Weight.Normal)

        self._tag = None

        h.addWidget(self.label)
        h.addWidget(self.value, 1)

    def setValue(self, v: str):
        self.value.setText(v)

    def addTag(self, text: str, color: QColor):
        """Add a tag badge to the right side."""
        if self._tag is None:
            self._tag = _Tag(text, color, self)
            self.layout().addWidget(self._tag)
        else:
            self._tag.setText(text)
            self._tag.setColor(color)


# ─── System Info Card ──────────────────────────────────────────────────────────


class SystemInfoCard(SimpleCardWidget):
    """Premium card showing system hardware & software environment.

    Displays GPU, CPU, RAM, Python version as stat blocks,
    and PyTorch, GPU memory, OS as detail rows.
    Supports periodic GPU stats refresh during training.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._accent = _accent_color()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(5000)
        self._refresh_timer.timeout.connect(self._refresh_gpu_stats)
        self._init_ui()
        self.populate()
        QTimer.singleShot(0, self._adjustHeight)

    def _init_ui(self):
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        root.addLayout(content)

        # ── Header ──────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(10)

        ic = IconWidget(FIF.IOT, self)
        ic.setFixedSize(20, 20)
        header.addWidget(ic)

        titleCol = QVBoxLayout()
        titleCol.setSpacing(2)
        titleLabel = StrongBodyLabel(self.tr("System Environment"), self)
        setFont(titleLabel, 15, QFont.Weight.DemiBold)
        self.subtitleLabel = CaptionLabel(self.tr("Hardware & software overview"), self)
        self.subtitleLabel.setTextColor(_muted_text_color(), _muted_text_color())
        titleCol.addWidget(titleLabel)
        titleCol.addWidget(self.subtitleLabel)
        header.addLayout(titleCol, 1)

        content.addLayout(header)
        content.addSpacing(16)

        # ── Stat blocks row ─────────────────────────────────────────
        statsRow = QHBoxLayout()
        statsRow.setSpacing(10)

        self.gpuStat = StatisticsWidget(self.tr("GPU"), self)
        self.cpuStat = StatisticsWidget(self.tr("CPU"), self)
        self.ramStat = StatisticsWidget(self.tr("RAM"), self)
        self.pythonStat = StatisticsWidget(self.tr("Python"), self)

        for s in (self.gpuStat, self.cpuStat, self.ramStat, self.pythonStat):
            statsRow.addWidget(s)

        content.addLayout(statsRow)
        content.addSpacing(14)
        content.addWidget(_Separator(self))
        content.addSpacing(10)

        # ── Environment details section ─────────────────────────────
        content.addWidget(
            _SectionHeader(FIF.DEVELOPER_TOOLS, "Environment Details", self)
        )
        content.addSpacing(4)

        self.pytorchRow = _PropertyRow(self.tr("PyTorch"), self)
        self.gpuMemoryRow = _PropertyRow(self.tr("GPU Memory"), self)
        self.osRow = _PropertyRow(self.tr("Operating System"), self)

        content.addWidget(self.pytorchRow)
        content.addWidget(self.gpuMemoryRow)
        content.addWidget(self.osRow)

    # ── Data population ─────────────────────────────────────────────

    def populate(self):
        """Gather and display all system information."""
        # GPU
        gpus = detect_gpus()
        if gpus:
            gpu = gpus[0]
            # Shorten long GPU names for the stat block
            short_name = self._shorten_gpu_name(gpu.name)
            self.gpuStat.setValue(short_name)
            if len(gpus) > 1:
                self.gpuStat.setValue(short_name, f"+{len(gpus) - 1} more")
            self._update_gpu_memory(gpus)
        else:
            self.gpuStat.setValue("CPU Only")
            self.gpuMemoryRow.setValue("N/A")

        # CPU
        cpu_name = get_cpu_name()
        self.cpuStat.setValue(self._shorten_cpu_name(cpu_name))

        # RAM
        ram_mb = get_system_memory_mb()
        if ram_mb > 0:
            ram_gb = ram_mb / 1024
            self.ramStat.setValue(f"{ram_gb:.0f} GB")
        else:
            self.ramStat.setValue("—")

        # Python
        import sys

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.pythonStat.setValue(py_ver)

        # PyTorch
        installed, version, cuda = check_pytorch_installation()
        if installed:
            self.pytorchRow.setValue(str(version))
            # Add CUDA/MPS badge
            if cuda:
                self.pytorchRow.addTag(
                    "CUDA",
                    QColor("#16a34a") if not isDarkTheme() else QColor("#4ade80"),
                )
            else:
                # Check MPS
                try:
                    import torch

                    if (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        self.pytorchRow.addTag(
                            "MPS",
                            QColor("#8b5cf6")
                            if not isDarkTheme()
                            else QColor("#a78bfa"),
                        )
                    else:
                        self.pytorchRow.addTag(
                            "CPU",
                            QColor("#d97706")
                            if not isDarkTheme()
                            else QColor("#fbbf24"),
                        )
                except ImportError:
                    self.pytorchRow.addTag(
                        "CPU",
                        QColor("#d97706") if not isDarkTheme() else QColor("#fbbf24"),
                    )
        else:
            self.pytorchRow.setValue("Not installed")
            self.pytorchRow.addTag(
                "Missing",
                QColor("#dc2626") if not isDarkTheme() else QColor("#f87171"),
            )

        # OS
        self.osRow.setValue(get_os_display_name())

    def _update_gpu_memory(self, gpus: list[GPUInfo] | None = None):
        """Update the GPU memory row with current stats."""
        if gpus is None:
            gpus = detect_gpus()
        if gpus:
            gpu = gpus[0]
            used = gpu.memory_total - gpu.memory_free
            self.gpuMemoryRow.setValue(
                f"{used:,} / {gpu.memory_total:,} MB ({gpu.memory_free:,} MB free)"
            )
        else:
            self.gpuMemoryRow.setValue("N/A")

    @staticmethod
    def _shorten_gpu_name(name: str) -> str:
        """Abbreviate verbose GPU names to fit stat blocks."""
        # Remove common prefixes
        for prefix in ("NVIDIA ", "AMD ", "Intel "):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        # Truncate if still too long
        if len(name) > 22:
            name = name[:20] + "…"
        return name

    @staticmethod
    def _shorten_cpu_name(name: str) -> str:
        """Abbreviate verbose CPU names."""
        # Apple Silicon is already short
        if "Apple" in name:
            return name.replace("Apple ", "")
        # Intel — keep core model
        for prefix in ("Intel(R) Core(TM) ", "Intel(R) Xeon(R) ", "AMD Ryzen "):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if len(name) > 20:
            name = name[:18] + "…"
        return name

    # ── ExpandLayout fix ──────────────────────────────────────────────

    def _adjustHeight(self):
        self.setFixedHeight(self.layout().sizeHint().height())
        QTimer.singleShot(0, self._updateParentLayout)

    def _updateParentLayout(self):
        parent = self.parentWidget()
        if parent is not None:
            parent.adjustSize()
            parent.updateGeometry()

    # ── Live refresh ────────────────────────────────────────────────

    def start_refresh(self):
        """Start periodic GPU stats refresh (call when training starts)."""
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def stop_refresh(self):
        """Stop periodic GPU stats refresh (call when training ends)."""
        self._refresh_timer.stop()
        # Do one final refresh
        self._refresh_gpu_stats()

    def _refresh_gpu_stats(self):
        """Refresh GPU memory statistics."""
        self._update_gpu_memory()

    # ── Custom painting ─────────────────────────────────────────────

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

        # Top accent bar (4 px) — teal/cyan to distinguish from ProgressCard
        accent = _accent_color()
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
        p.setBrush(accent)
        p.drawPath(path)

        p.end()
