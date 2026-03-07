"""
WaveDL GUI - Wizard Step 5: Training Ready / Summary

Summary of all wizard choices with a prominent "Start Training" button.
Uses a custom-painted summary card with proper dark/light theme support.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    FluentIcon as FIF,
    IconWidget,
    PrimaryPushButton,
    SubtitleLabel,
    isDarkTheme,
    setFont,
)

from ...common.constants.models import MODEL_INFO
from ...common.wizard_state import PRESETS, WizardState
from ...components.shared import add_step_header


# ─── Summary card with custom rounded painting ───────────────────────────────


class _SummaryCard(QWidget):
    """Rounded card with proper dark/light background for the config summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(24, 20, 24, 20)
        self._layout.setSpacing(6)

        header = SubtitleLabel("Configuration Summary", self)
        setFont(header, 14, QFont.Weight.DemiBold)
        self._layout.addWidget(header)
        self._layout.addSpacing(8)

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._layout

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg = QColor(39, 39, 41) if isDarkTheme() else QColor(248, 250, 252)
        border = QColor(55, 55, 58) if isDarkTheme() else QColor(225, 225, 230)
        path = QPainterPath()
        path.addRoundedRect(
            0.5,
            0.5,
            self.width() - 1,
            self.height() - 1,
            10,
            10,
        )
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawPath(path)
        p.setPen(QPen(border, 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)


# ─── Summary row ──────────────────────────────────────────────────────────────


class _SummaryRow(QWidget):
    """Single key-value row in the summary card."""

    def __init__(self, icon: FIF, label: str, value: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 4, 0, 4)
        h.setSpacing(10)

        ic = IconWidget(icon, self)
        ic.setFixedSize(18, 18)
        h.addWidget(ic)

        lbl = BodyLabel(label, self)
        setFont(lbl, 13, QFont.Weight.DemiBold)
        lbl.setFixedWidth(120)
        h.addWidget(lbl)

        val = BodyLabel(value, self)
        val.setStyleSheet(f"color: {'#b4b4b4' if isDarkTheme() else '#505050'};")
        val.setWordWrap(True)
        h.addWidget(val, 1)


# ─── Training step widget ────────────────────────────────────────────────────


class TrainingStep(QWidget):
    """Step 5: review & start training."""

    startClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_SummaryRow] = []
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 36, 60, 36)
        root.setSpacing(0)

        add_step_header(
            root,
            "Ready to Train",
            "Review your configuration below, then click Start Training.",
            self,
            bottom_spacing=28,
        )

        # Summary card — properly painted with rounded corners
        self._card = _SummaryCard(self)
        root.addWidget(self._card)

        root.addSpacing(32)

        # Start button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._start_btn = PrimaryPushButton(FIF.PLAY, "Start Training", self)
        self._start_btn.setFixedSize(220, 44)
        setFont(self._start_btn, 15, QFont.Weight.DemiBold)
        self._start_btn.clicked.connect(self.startClicked.emit)
        btn_row.addWidget(self._start_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

    def populate(self, state: WizardState):
        """Fill the summary card from the wizard state."""
        # Clear old rows
        for row in self._rows:
            row.deleteLater()
        self._rows.clear()

        data_name = os.path.basename(state.data_path) if state.data_path else "—"
        model_info = MODEL_INFO.get(state.selected_model, {})
        model_name = model_info.get("display_name", state.selected_model)
        preset_info = PRESETS.get(state.preset, {})
        preset_name = preset_info.get("label", state.preset)

        entries = [
            (FIF.DOCUMENT, "Data File", data_name),
            (FIF.FOLDER, "Output", state.output_dir or "Default"),
            (FIF.ROBOT, "Model", model_name),
            (FIF.SPEED_HIGH, "Preset", f"{preset_info.get('icon', '')}  {preset_name}"),
            (FIF.DATE_TIME, "Est. Time", preset_info.get("badge", "—")),
        ]

        for icon, label, value in entries:
            row = _SummaryRow(icon, label, value, self._card)
            self._card.content_layout.addWidget(row)
            self._rows.append(row)
