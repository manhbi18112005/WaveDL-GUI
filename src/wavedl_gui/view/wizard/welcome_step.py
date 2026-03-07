"""
WaveDL GUI - Wizard Step 1: Welcome & Mode Selection

Visually polished welcome screen with two large interactive cards
for choosing between Basic (guided wizard) and Advanced (full UI) modes.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon as FIF,
    IconWidget,
    SubtitleLabel,
    TitleLabel,
    isDarkTheme,
    setFont,
)


# ─── Mode card ────────────────────────────────────────────────────────────────


class _ModeCard(QWidget):
    """Large, clickable card representing a user mode."""

    clicked = Signal()

    def __init__(
        self,
        icon: FIF,
        title: str,
        description: str,
        accent: QColor,
        parent=None,
    ):
        super().__init__(parent)
        self._accent = accent
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # ── layout ───
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(12)

        # Icon row
        ic = IconWidget(icon, self)
        ic.setFixedSize(40, 40)
        root.addWidget(ic)

        # Title
        lbl = SubtitleLabel(title, self)
        setFont(lbl, 20, QFont.Weight.DemiBold)
        root.addWidget(lbl)

        # Description — theme-aware color
        desc = BodyLabel(description, self)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {'#b4b4b4' if isDarkTheme() else '#646464'};")
        root.addWidget(desc)

        root.addStretch()

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Theme-aware card background
        bg = QColor(39, 39, 41) if isDarkTheme() else QColor(255, 255, 255)
        border = (
            self._accent
            if self._hovered
            else (QColor(55, 55, 58) if isDarkTheme() else QColor(225, 225, 230))
        )

        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, 12, 12)

        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawPath(path)

        p.setPen(QPen(border, 2 if self._hovered else 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def enterEvent(self, _):
        self._hovered = True
        self.update()

    def leaveEvent(self, _):
        self._hovered = False
        self.update()

    def mousePressEvent(self, _):
        self.clicked.emit()


# ─── Welcome step widget ─────────────────────────────────────────────────────


class WelcomeStep(QWidget):
    """Step 1: mode selection."""

    modeSelected = Signal(str)  # "basic" or "advanced"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 48, 60, 48)
        root.setSpacing(0)

        # Title
        title = TitleLabel("Welcome to WaveDL", self)
        setFont(title, 32, QFont.Weight.Bold)
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        root.addSpacing(8)

        # Subtitle — use setTextColor for proper dark/light support
        sub = SubtitleLabel("Deep learning training made simple", self)
        setFont(sub, 16, QFont.Weight.Normal)
        sub.setAlignment(Qt.AlignCenter)
        sub.setTextColor(QColor(120, 120, 120), QColor(160, 160, 160))
        root.addWidget(sub)

        root.addSpacing(12)

        # Description
        desc = CaptionLabel(
            "Choose how you'd like to get started. You can always switch later.",
            self,
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setTextColor(QColor(140, 140, 140), QColor(140, 140, 140))
        root.addWidget(desc)

        root.addSpacing(40)

        # ── Cards ────
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)

        basic_card = _ModeCard(
            FIF.EDUCATION,
            "Basic Mode",
            "Step-by-step guided wizard. Perfect for beginners — "
            "we'll walk you through data selection, model choosing, "
            "and training with smart defaults.",
            QColor(0, 159, 170),
            self,
        )
        advanced_card = _ModeCard(
            FIF.DEVELOPER_TOOLS,
            "Advanced Mode",
            "Full control over every hyperparameter. "
            "For experienced users who want direct access "
            "to the complete training configuration.",
            QColor(138, 101, 212),
            self,
        )

        basic_card.clicked.connect(lambda: self.modeSelected.emit("basic"))
        advanced_card.clicked.connect(lambda: self.modeSelected.emit("advanced"))

        cards_layout.addWidget(basic_card)
        cards_layout.addWidget(advanced_card)
        root.addLayout(cards_layout)

        root.addStretch()
