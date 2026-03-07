"""
WaveDL GUI - Onboarding Wizard

Frameless dialog containing a QStackedWidget with animated transitions
between the 5 wizard steps.  Provides Back/Next navigation with
per-step validation gating.
"""

from __future__ import annotations

from typing import ClassVar

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    FluentIcon as FIF,
    PrimaryPushButton,
    PushButton,
    isDarkTheme,
)

from ...common.wizard_state import WizardState
from .data_step import DataStep
from .model_step import ModelStep
from .preset_step import PresetStep
from .training_step import TrainingStep
from .welcome_step import WelcomeStep


# ─── Step dot indicator ──────────────────────────────────────────────────────


class _DotIndicator(QWidget):
    """Horizontal row of dots showing the current wizard step."""

    def __init__(self, count: int, parent=None):
        super().__init__(parent)
        self._count = count
        self._current = 0
        self.setFixedHeight(20)

    def set_current(self, index: int):
        self._current = index
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        dot_r = 4
        spacing = 16
        total_w = self._count * (dot_r * 2) + (self._count - 1) * (spacing - dot_r * 2)
        start_x = (self.width() - total_w) / 2

        accent = QColor(0, 159, 170)
        inactive = QColor(80, 80, 80) if isDarkTheme() else QColor(200, 200, 200)

        for i in range(self._count):
            cx = start_x + i * spacing + dot_r
            cy = self.height() / 2
            color = accent if i == self._current else inactive
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(int(cx - dot_r), int(cy - dot_r), dot_r * 2, dot_r * 2)


# ─── Onboarding wizard dialog ────────────────────────────────────────────────


class OnboardingWizard(QWidget):
    """Full-screen overlay wizard with 5 steps and animated transitions."""

    wizardCompleted = Signal(object)  # WizardState
    wizardCancelled = Signal()

    STEP_LABELS: ClassVar[list[str]] = ["Welcome", "Data", "Model", "Preset", "Train"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = WizardState()

        self._init_steps()
        self._init_ui()
        self._connect_signals()

    def _init_steps(self):
        self._welcome = WelcomeStep()
        self._data = DataStep()
        self._model = ModelStep()
        self._preset = PresetStep()
        self._training = TrainingStep()

    def _init_ui(self):
        self.setObjectName("onboardingWizard")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Stacked widget ──
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._welcome)  # 0
        self._stack.addWidget(self._data)  # 1
        self._stack.addWidget(self._model)  # 2
        self._stack.addWidget(self._preset)  # 3
        self._stack.addWidget(self._training)  # 4
        root.addWidget(self._stack, 1)

        # ── Bottom nav bar ──
        nav = QWidget(self)
        nav.setFixedHeight(64)
        nav_bg = "rgba(30,30,32,0.95)" if isDarkTheme() else "rgba(245,245,248,0.95)"
        nav.setStyleSheet(f"background: {nav_bg};")

        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(32, 0, 32, 0)

        self._back_btn = PushButton(FIF.LEFT_ARROW, "Back", nav)
        self._back_btn.setFixedWidth(100)
        self._back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self._back_btn)

        nav_layout.addStretch()

        self._dots = _DotIndicator(len(self.STEP_LABELS), nav)
        self._dots.setFixedWidth(160)
        nav_layout.addWidget(self._dots)

        nav_layout.addStretch()

        self._next_btn = PrimaryPushButton(FIF.RIGHT_ARROW, "Next", nav)
        self._next_btn.setFixedWidth(100)
        self._next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self._next_btn)

        root.addWidget(nav)

        # Initial state — hide nav on welcome screen
        self._update_nav()

    def _connect_signals(self):
        # Step 1 — mode selected
        self._welcome.modeSelected.connect(self._on_mode_selected)

        # Step 2 — data validated
        self._data.dataValidated.connect(self._on_data_validated)
        self._data.dataValidated.connect(lambda _: self._update_nav())

        # Step 3 — model selected
        self._model.modelSelected.connect(self._on_model_selected)

        # Step 4 — preset selected
        self._preset.presetSelected.connect(self._on_preset_selected)

        # Step 5 — start training
        self._training.startClicked.connect(self._on_start_training)

    # ── Signal handlers ───────────────────────────────────────────────────

    def _on_mode_selected(self, mode: str):
        self.state.user_mode = mode
        if mode == "advanced":
            self.wizardCompleted.emit(self.state)
        else:
            self._animate_to(1)

    def _on_data_validated(self, info):
        self.state.data_info = info

    def _on_model_selected(self, model_id: str):
        self.state.selected_model = model_id

    def _on_preset_selected(self, preset: str):
        self.state.preset = preset

    def _on_start_training(self):
        self.state.data_path = self._data.get_data_path()
        self.state.output_dir = self._data.get_output_dir()
        self.state.selected_model = self._model.get_selected_model()
        self.state.preset = self._preset.get_selected_preset()
        self.wizardCompleted.emit(self.state)

    # ── Navigation ────────────────────────────────────────────────────────

    def _go_back(self):
        idx = self._stack.currentIndex()
        if idx > 1:  # Can't go back past data step (step 0 is welcome)
            self._animate_to(idx - 1)

    def _go_next(self):
        idx = self._stack.currentIndex()
        if not self._validate_step(idx):
            return

        # Populate next step if needed
        if idx == 1:  # data → model
            self._model.populate(self.state.data_info)
        elif idx == 2:  # model → preset
            self.state.selected_model = self._model.get_selected_model()
        elif idx == 3:  # preset → training
            self.state.data_path = self._data.get_data_path()
            self.state.output_dir = self._data.get_output_dir()
            self.state.preset = self._preset.get_selected_preset()
            self._training.populate(self.state)

        if idx < self._stack.count() - 1:
            self._animate_to(idx + 1)

    def _validate_step(self, idx: int) -> bool:
        """Return True if the current step is valid and can proceed."""
        if idx == 1:
            if not self._data.is_valid():
                from qfluentwidgets import InfoBar, InfoBarPosition

                w = self.window()
                InfoBar.warning(
                    title="Data Required",
                    content="Please select a valid data file before continuing.",
                    parent=w if w else self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
                return False
        elif idx == 2:
            if not self._model.is_valid():
                from qfluentwidgets import InfoBar, InfoBarPosition

                w = self.window()
                InfoBar.warning(
                    title="Model Required",
                    content="Please select a model before continuing.",
                    parent=w if w else self,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                )
                return False
        return True

    # ── Animated transition ───────────────────────────────────────────────

    def _animate_to(self, target_idx: int):
        """Slide + fade transition between wizard steps."""
        current = self._stack.currentWidget()
        target = self._stack.widget(target_idx)

        if current is target:
            return

        w = self._stack.width()
        going_forward = target_idx > self._stack.currentIndex()

        # Position target off-screen
        target.setGeometry(
            QRect(w if going_forward else -w, 0, w, self._stack.height())
        )
        target.show()
        target.raise_()

        # Opacity effects
        current_opacity = QGraphicsOpacityEffect(current)
        current.setGraphicsEffect(current_opacity)
        current_opacity.setOpacity(1.0)

        target_opacity = QGraphicsOpacityEffect(target)
        target.setGraphicsEffect(target_opacity)
        target_opacity.setOpacity(0.0)

        duration = 300

        group = QParallelAnimationGroup(self)

        # Current slide out
        anim_current_pos = QPropertyAnimation(current, b"geometry", self)
        anim_current_pos.setDuration(duration)
        anim_current_pos.setStartValue(QRect(0, 0, w, self._stack.height()))
        anim_current_pos.setEndValue(
            QRect(-w if going_forward else w, 0, w, self._stack.height())
        )
        anim_current_pos.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(anim_current_pos)

        # Current fade out
        anim_current_fade = QPropertyAnimation(current_opacity, b"opacity", self)
        anim_current_fade.setDuration(duration)
        anim_current_fade.setStartValue(1.0)
        anim_current_fade.setEndValue(0.0)
        group.addAnimation(anim_current_fade)

        # Target slide in
        anim_target_pos = QPropertyAnimation(target, b"geometry", self)
        anim_target_pos.setDuration(duration)
        anim_target_pos.setStartValue(
            QRect(w if going_forward else -w, 0, w, self._stack.height())
        )
        anim_target_pos.setEndValue(QRect(0, 0, w, self._stack.height()))
        anim_target_pos.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(anim_target_pos)

        # Target fade in
        anim_target_fade = QPropertyAnimation(target_opacity, b"opacity", self)
        anim_target_fade.setDuration(duration)
        anim_target_fade.setStartValue(0.0)
        anim_target_fade.setEndValue(1.0)
        group.addAnimation(anim_target_fade)

        def on_finished():
            self._stack.setCurrentIndex(target_idx)
            # Clean up opacity effects
            current.setGraphicsEffect(None)
            target.setGraphicsEffect(None)
            self._update_nav()

        group.finished.connect(on_finished)
        group.start()

    # ── Nav state ─────────────────────────────────────────────────────────

    def _update_nav(self):
        idx = self._stack.currentIndex()
        self._dots.set_current(idx)

        # Hide entire nav on welcome (step 0)
        nav_visible = idx > 0
        self._back_btn.setVisible(nav_visible and idx > 1)
        self._dots.setVisible(nav_visible)

        if idx == self._stack.count() - 1:
            # Last step — hide Next (Start Training button is in the step)
            self._next_btn.setVisible(False)
        else:
            self._next_btn.setVisible(nav_visible)
            self._next_btn.setText("Next")

            # Disable Next when the current step is not valid
            if idx == 1:  # Data step
                self._next_btn.setEnabled(self._data.is_valid())
            else:
                self._next_btn.setEnabled(True)

    # ── Paint background ──────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        bg = QColor(32, 32, 35) if isDarkTheme() else QColor(249, 249, 252)
        p.fillRect(self.rect(), bg)
