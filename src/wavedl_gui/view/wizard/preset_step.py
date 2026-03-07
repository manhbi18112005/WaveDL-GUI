"""
WaveDL GUI - Wizard Step 4: Configuration Presets

Apple-inspired preset selection with a DataInfoCard-style detail card
that appears below the preset list to show hyperparameter details.
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
    SimpleCardWidget,
    StrongBodyLabel,
    isDarkTheme,
    setFont,
)

from ...common.theme_colors import (
    accent_color as _accent_color,
    muted_text_color as _muted_text_color,
)
from ...common.wizard_state import PRESETS
from ...components.shared import Separator as _Separator, add_step_header
from ...components.statistic_widget import StatisticsWidget


# ─── Preset card (compact, no shadow effect to avoid QPainter conflicts) ──────


class _PresetCard(QWidget):
    """Clean, compact preset selector card."""

    clicked = Signal(str)  # preset key

    def __init__(self, key: str, preset: dict, parent=None):
        super().__init__(parent)
        self._key = key
        self._selected = False
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(72)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 12)
        root.setSpacing(12)

        # Icon
        icon_lbl = BodyLabel(preset["icon"], self)
        setFont(icon_lbl, 22)
        root.addWidget(icon_lbl)

        # Name + short desc
        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        name = StrongBodyLabel(preset["label"], self)
        setFont(name, 14, QFont.Weight.DemiBold)
        text_col.addWidget(name)

        hint = CaptionLabel(preset["badge"], self)
        hint.setTextColor(_muted_text_color(), _muted_text_color())
        text_col.addWidget(hint)

        root.addLayout(text_col, 1)

    @property
    def key(self) -> str:
        return self._key

    def set_selected(self, selected: bool):
        self._selected = selected
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        dark = isDarkTheme()
        bg = QColor(44, 44, 47) if dark else QColor(255, 255, 255)
        accent = QColor(0, 122, 255)

        if self._selected:
            border_color = accent
            border_w = 2.0
        elif self._hovered:
            border_color = QColor(70, 70, 74) if dark else QColor(200, 200, 205)
            border_w = 1.0
        else:
            border_color = QColor(50, 50, 53) if dark else QColor(230, 230, 232)
            border_w = 0.5

        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, self.width() - 1, self.height() - 1, 10, 10)

        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawPath(path)

        p.setPen(QPen(border_color, border_w))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def enterEvent(self, _):
        self._hovered = True
        self.update()

    def leaveEvent(self, _):
        self._hovered = False
        self.update()

    def mousePressEvent(self, _):
        self.clicked.emit(self._key)


# ─── Preset detail card (DataInfoCard style) ─────────────────────────────────


class _PropertyRow(QWidget):
    """Label → value row."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(12)

        self.label = CaptionLabel(label, self)
        self.label.setTextColor(_muted_text_color(), _muted_text_color())
        self.label.setFixedWidth(110)

        self.value = BodyLabel("", self)
        setFont(self.value, 12, QFont.Weight.Normal)

        h.addWidget(self.label)
        h.addWidget(self.value, 1)

    def setValue(self, v: str):
        self.value.setText(v)


class _PresetInfoCard(SimpleCardWidget):
    """Rich detail card showing preset hyperparameters, styled like DataInfoCard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._accent = _accent_color()
        self._init_ui()
        self.hide()

    def _init_ui(self):
        self.setBorderRadius(10)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QVBoxLayout()
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(0)
        root.addLayout(content)

        # ── Header: preset name + description ──
        self._name_label = StrongBodyLabel("", self)
        setFont(self._name_label, 15, QFont.Weight.DemiBold)
        content.addWidget(self._name_label)

        content.addSpacing(4)

        self._desc_label = CaptionLabel("", self)
        self._desc_label.setWordWrap(True)
        self._desc_label.setTextColor(_muted_text_color(), _muted_text_color())
        content.addWidget(self._desc_label)

        content.addSpacing(14)

        # ── Stat boxes ──
        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self._epochs_stat = StatisticsWidget("Epochs", self)
        self._lr_stat = StatisticsWidget("Learn Rate", self)
        self._batch_stat = StatisticsWidget("Batch Size", self)
        self._time_stat = StatisticsWidget("Est. Time", self)
        for s in (self._epochs_stat, self._lr_stat, self._batch_stat, self._time_stat):
            stats_row.addWidget(s)
        content.addLayout(stats_row)

        content.addSpacing(14)
        content.addWidget(_Separator(self))
        content.addSpacing(10)

        # ── Detail rows ──
        self._optimizer_row = _PropertyRow("Optimizer", self)
        self._scheduler_row = _PropertyRow("Scheduler", self)
        self._precision_row = _PropertyRow("Precision", self)
        self._patience_row = _PropertyRow("Patience", self)
        self._decay_row = _PropertyRow("Weight Decay", self)

        for row in (
            self._optimizer_row,
            self._scheduler_row,
            self._precision_row,
            self._patience_row,
            self._decay_row,
        ):
            content.addWidget(row)

        self._content_layout = content

    def set_preset(self, key: str):
        """Update the card for a given preset key."""
        preset = PRESETS.get(key)
        if not preset:
            self.hide()
            return

        self._name_label.setText(f"{preset['icon']}  {preset['label']}")
        self._desc_label.setText(preset["description"])

        self._epochs_stat.setValue(f"{preset['epochs']:,}")
        self._lr_stat.setValue(str(preset["lr"]))
        self._batch_stat.setValue(str(preset["batch_size"]))
        self._time_stat.setValue(preset["badge"])

        self._optimizer_row.setValue(preset["optimizer"].upper())
        self._scheduler_row.setValue(preset["scheduler"].replace("_", " ").title())
        self._precision_row.setValue(preset["precision"].upper())
        self._patience_row.setValue(str(preset["patience"]))
        self._decay_row.setValue(str(preset["weight_decay"]))

        self.show()

    # ── Custom painting (same as DataInfoCard) ───────────────────────

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
        p.setBrush(self._accent)
        p.drawPath(path)

        p.end()


# ─── Preset step widget ──────────────────────────────────────────────────────


class PresetStep(QWidget):
    """Step 4: training preset selection with detail card."""

    presetSelected = Signal(str)  # preset key

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected: str = "balanced"
        self._cards: list[_PresetCard] = []
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(60, 36, 60, 36)
        root.setSpacing(0)

        add_step_header(
            root,
            "Training Configuration",
            "Choose a training configuration that fits your needs.",
            self,
        )

        # Preset cards (compact)
        for key, preset in PRESETS.items():
            card = _PresetCard(key, preset, self)
            card.clicked.connect(self._on_card_clicked)
            root.addWidget(card)
            root.addSpacing(6)
            self._cards.append(card)

        root.addSpacing(16)

        # Detail card (DataInfoCard style)
        self._info_card = _PresetInfoCard(self)
        root.addWidget(self._info_card)

        root.addStretch()

        # Default selection
        self._select("balanced")

    def _on_card_clicked(self, key: str):
        self._select(key)

    def _select(self, key: str):
        self._selected = key
        for card in self._cards:
            card.set_selected(card.key == key)
        self._info_card.set_preset(key)
        self.presetSelected.emit(key)

    def get_selected_preset(self) -> str:
        return self._selected

    def is_valid(self) -> bool:
        return bool(self._selected)
