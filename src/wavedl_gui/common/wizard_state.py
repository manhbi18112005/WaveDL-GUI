"""
WaveDL GUI - Wizard State

Centralised dataclass that holds all user selections made during the
onboarding wizard.  Kept intentionally decoupled from UI views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .utils import DataInfo


# ── Preset definitions (hidden from the user) ────────────────────────────────

PRESETS = {
    "fast": {
        "label": "Fast Training",
        "icon": "⚡",
        "badge": "~5-15 min",
        "description": "Quick iteration for exploration and prototyping. "
        "Lower accuracy but very fast turnaround.",
        "epochs": 100,
        "lr": 0.003,
        "batch_size": 256,
        "patience": 10,
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "optimizer": "adamw",
        "loss": "mse",
        "precision": "bf16",
    },
    "balanced": {
        "label": "Balanced",
        "icon": "⚖️",
        "badge": "~30-60 min",
        "description": "Recommended default. Good balance between accuracy "
        "and training time for most datasets.",
        "epochs": 500,
        "lr": 0.001,
        "batch_size": 128,
        "patience": 20,
        "weight_decay": 0.0001,
        "scheduler": "plateau",
        "optimizer": "adamw",
        "loss": "mse",
        "precision": "bf16",
    },
    "high_accuracy": {
        "label": "High Accuracy",
        "icon": "🎯",
        "badge": "~2-4 hrs",
        "description": "Maximum model quality. Trains longer with finer "
        "learning rate and larger patience for the best results.",
        "epochs": 2000,
        "lr": 0.0005,
        "batch_size": 64,
        "patience": 50,
        "weight_decay": 0.00005,
        "scheduler": "cosine_restarts",
        "optimizer": "adamw",
        "loss": "mse",
        "precision": "bf16",
    },
}


@dataclass
class WizardState:
    """Persistent state collected across the wizard steps."""

    # Step 1 — mode
    user_mode: str = "basic"  # "basic" | "advanced"

    # Step 2 — data
    data_path: str = ""
    output_dir: str = ""
    data_info: DataInfo | None = field(default=None, repr=False)

    # Step 3 — model
    selected_model: str = "cnn"

    # Step 4 — preset
    preset: str = "balanced"

    # ── helpers ───────────────────────────────────────────────────────────

    def to_training_config(self):
        """Convert wizard state into a TrainingConfig for the training service."""
        from .constants.index import TrainingConfig

        p = PRESETS.get(self.preset, PRESETS["balanced"])

        return TrainingConfig(
            data_path=self.data_path,
            output_dir=self.output_dir,
            model=self.selected_model,
            pretrained=self._should_use_pretrained(),
            batch_size=p["batch_size"],
            lr=p["lr"],
            epochs=p["epochs"],
            patience=p["patience"],
            loss=p["loss"],
            optimizer=p["optimizer"],
            scheduler=p["scheduler"],
            precision=p["precision"],
            weight_decay=p["weight_decay"],
        )

    def _should_use_pretrained(self) -> bool:
        from .constants.models import MODEL_INFO

        info = MODEL_INFO.get(self.selected_model, {})
        return bool(info.get("is_pretrained"))
