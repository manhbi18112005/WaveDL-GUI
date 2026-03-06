"""
WaveDL GUI - Training Service

Manages training subprocesses with real-time output parsing and progress tracking.
Provides clean process isolation to prevent GUI freezing and enable graceful termination.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal

from ..common.signal_bus import signalBus
from ..common.utils import get_python_executable


if TYPE_CHECKING:
    from ..common.constants.index import TestConfig, TrainingConfig


class ProcessState(Enum):
    """States for training/test processes."""

    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class TrainingProgress:
    """Real-time training progress information."""

    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    max_patience: int = 0
    r2_score: float = 0.0
    pearson: float = 0.0
    grad_norm: float = 0.0
    mae_avg: float = 0.0
    mae_per_param: list[float] = field(default_factory=list)
    time_per_epoch: float = 0.0
    total_time: float = 0.0
    eta_seconds: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Overall progress percentage."""
        if self.total_epochs == 0:
            return 0.0
        return (self.epoch / self.total_epochs) * 100


_METRICS_PREFIX = "##METRICS##"


class OutputParser:
    """Parses structured ##METRICS## JSON lines emitted by train.py."""

    def __init__(self):
        self.progress = TrainingProgress()

    def parse_line(self, line: str) -> tuple[TrainingProgress, bool]:
        """Parse a single line of output and update progress.

        Returns:
            Tuple of (progress, is_metrics_line). When is_metrics_line is True,
            the caller should suppress this line from the visible log.
        """
        stripped = line.strip()
        if not stripped:
            return self.progress, False

        if stripped.startswith(_METRICS_PREFIX):
            self._parse_metrics_json(stripped[len(_METRICS_PREFIX) :])
            return self.progress, True

        return self.progress, False

    def _parse_metrics_json(self, json_str: str) -> None:
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return

        p = self.progress
        p.epoch = data.get("epoch", p.epoch)
        p.total_epochs = data.get("total_epochs", p.total_epochs)
        p.train_loss = data.get("train_loss", p.train_loss)
        p.val_loss = data.get("val_loss", p.val_loss)
        p.best_val_loss = data.get("best_val_loss", p.best_val_loss)
        p.r2_score = data.get("r2", p.r2_score)
        p.pearson = data.get("pearson", p.pearson)
        p.grad_norm = data.get("grad_norm", p.grad_norm)
        p.learning_rate = data.get("lr", p.learning_rate)
        p.mae_avg = data.get("mae_avg", p.mae_avg)
        p.mae_per_param = data.get("mae_per_param", p.mae_per_param)
        p.time_per_epoch = data.get("epoch_time", p.time_per_epoch)
        p.total_time = data.get("total_time", p.total_time)
        p.patience_counter = data.get("patience_counter", p.patience_counter)
        p.max_patience = data.get("max_patience", p.max_patience)

        remaining = p.total_epochs - p.epoch
        p.eta_seconds = remaining * p.time_per_epoch if p.time_per_epoch > 0 else 0.0


class TrainingWorker(QThread):
    """Worker thread for running training subprocess."""

    progressSig = Signal(object)  # TrainingProgress
    outputSig = Signal(str)  # Log line
    stateSig = Signal(object)  # ProcessState
    finishedSig = Signal(bool, str)  # success, message

    def __init__(self, config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.process = None
        self._cancelled = False
        self._parser = OutputParser()

    def run(self):
        self._cancelled = False
        self.stateSig.emit(ProcessState.STARTING)

        try:
            if self.config.no_cache:
                self._delete_cache_files()

            cmd = self._build_command()
            self.outputSig.emit(f"$ {' '.join(cmd)}")

            # Validate and ensure cwd exists
            cwd = self.config.output_dir or None
            if cwd and not os.path.isdir(cwd):
                os.makedirs(cwd, exist_ok=True)

            # Start process in its own session to isolate from parent signals
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=self._get_env(),
                cwd=cwd,
                start_new_session=True,
            )

            self.stateSig.emit(ProcessState.RUNNING)

            # Read output line by line
            for line in iter(self.process.stdout.readline, ""):
                if self._cancelled:
                    break

                line = line.rstrip()

                progress, is_metrics = self._parser.parse_line(line)

                if is_metrics:
                    self.progressSig.emit(progress)
                else:
                    self.outputSig.emit(line)

            # Wait for process to complete
            return_code = self.process.wait()

            if self._cancelled:
                self.stateSig.emit(ProcessState.CANCELLED)
                self.finishedSig.emit(False, "Training cancelled by user")
            elif return_code == 0:
                self.stateSig.emit(ProcessState.COMPLETED)
                self.finishedSig.emit(True, "Training completed successfully")
            else:
                self.stateSig.emit(ProcessState.FAILED)
                self.finishedSig.emit(False, f"Training failed with code {return_code}")

        except Exception as e:
            self.stateSig.emit(ProcessState.FAILED)
            self.finishedSig.emit(False, str(e))

    def _build_command(self) -> list[str]:
        python = get_python_executable()
        cmd = [python, "-u", "-m", "wavedl.train"]
        cmd.extend(self.config.to_cli_args())
        return cmd

    _CACHE_FILES = ("train_data_cache.dat", "scaler.pkl", "data_metadata.pkl")

    def _delete_cache_files(self):
        cache_dir = self.config.output_dir
        if not cache_dir or not os.path.isdir(cache_dir):
            return
        for name in self._CACHE_FILES:
            path = os.path.join(cache_dir, name)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    self.outputSig.emit(f"Removed cache file: {name}")
                except OSError:
                    pass

    def _get_env(self) -> dict:
        """Get environment variables for the subprocess."""
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            if sys.platform != "darwin":
                env["CUDA_VISIBLE_DEVICES"] = "0"
        return env

    def stop(self):
        """Stop the training process."""
        self._cancelled = True
        if self.process and self.process.poll() is None:
            self.stateSig.emit(ProcessState.STOPPING)
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
            except (ProcessLookupError, OSError):
                pass


class TestWorker(QThread):
    """Worker thread for running inference/test subprocess."""

    outputSig = Signal(str)
    stateSig = Signal(object)
    finishedSig = Signal(bool, str, dict)  # success, message, results

    def __init__(self, config: TestConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.process = None
        self._cancelled = False

    def run(self):
        """Run the test process."""
        self._cancelled = False
        self.stateSig.emit(ProcessState.STARTING)

        try:
            cmd = self._build_command()
            self.outputSig.emit(f"$ {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            self.stateSig.emit(ProcessState.RUNNING)

            output_lines = []
            for line in iter(self.process.stdout.readline, ""):
                if self._cancelled:
                    break
                line = line.rstrip()
                output_lines.append(line)
                self.outputSig.emit(line)

            return_code = self.process.wait()

            # Parse results from output
            results = self._parse_results(output_lines)

            if self._cancelled:
                self.stateSig.emit(ProcessState.CANCELLED)
                self.finishedSig.emit(False, "Test cancelled by user", {})
            elif return_code == 0:
                self.stateSig.emit(ProcessState.COMPLETED)
                self.finishedSig.emit(True, "Test completed successfully", results)
            else:
                self.stateSig.emit(ProcessState.FAILED)
                self.finishedSig.emit(False, f"Test failed with code {return_code}", {})

        except Exception as e:
            self.stateSig.emit(ProcessState.FAILED)
            self.finishedSig.emit(False, str(e), {})

    def _build_command(self) -> list[str]:
        """Build the test command."""
        python = get_python_executable()
        cmd = [python, "-u", "-m", "wavedl.test"]
        cmd.extend(self.config.to_cli_args())
        return cmd

    def _parse_results(self, lines: list[str]) -> dict:
        """Parse test results from output."""
        results = {}

        for line in lines:
            # Parse MAE values
            if "MAE:" in line or "Mean Absolute Error:" in line:
                match = re.search(r"[\d.e+-]+", line)
                if match:
                    results["mae"] = float(match.group())

            # Parse R² values
            if "R²" in line or "R2" in line:
                match = re.search(r"[\d.e+-]+", line)
                if match:
                    results["r2"] = float(match.group())

            # Parse correlation
            if "Pearson" in line:
                match = re.search(r"[\d.e+-]+", line)
                if match:
                    results["pearson"] = float(match.group())

        return results

    def stop(self):
        """Stop the test process."""
        self._cancelled = True
        if self.process and self.process.poll() is None:
            self.stateSig.emit(ProcessState.STOPPING)
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
            except (ProcessLookupError, OSError):
                pass


class TrainingService(QObject):
    """Service for managing training and test workflows.

    Provides a high-level interface for starting/stopping training
    and integrates with the SignalBus for application-wide events.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_worker: TrainingWorker | None = None
        self._test_worker: TestWorker | None = None
        self._current_config: TrainingConfig | None = None

    @property
    def is_training(self) -> bool:
        """Check if training is currently running."""
        return self._training_worker is not None and self._training_worker.isRunning()

    @property
    def is_testing(self) -> bool:
        """Check if testing is currently running."""
        return self._test_worker is not None and self._test_worker.isRunning()

    def start_training(self, config: TrainingConfig):
        """Start a training run.

        Args:
            config: Training configuration
        """
        if self.is_training:
            signalBus.appErrorSig.emit("Training already in progress")
            return

        self._current_config = config
        self._training_worker = TrainingWorker(config)

        # Connect signals
        self._training_worker.progressSig.connect(signalBus.trainingProgressSig.emit)
        self._training_worker.outputSig.connect(signalBus.trainingOutputSig.emit)
        self._training_worker.stateSig.connect(signalBus.trainingStateChangedSig.emit)
        self._training_worker.finishedSig.connect(self._on_training_finished)

        self._training_worker.start()

    def stop_training(self):
        """Stop the current training run."""
        if self._training_worker and self._training_worker.isRunning():
            self._training_worker.stop()

    def start_test(self, config: TestConfig):
        """Start a test/inference run.

        Args:
            config: Test configuration
        """
        if self.is_testing:
            signalBus.appErrorSig.emit("Test already in progress")
            return

        self._test_worker = TestWorker(config)

        self._test_worker.outputSig.connect(signalBus.trainingOutputSig.emit)
        self._test_worker.stateSig.connect(signalBus.trainingStateChangedSig.emit)
        self._test_worker.finishedSig.connect(self._on_test_finished)

        self._test_worker.start()

    def stop_test(self):
        """Stop the current test run."""
        if self._test_worker and self._test_worker.isRunning():
            self._test_worker.stop()

    def _on_training_finished(self, success: bool, message: str):
        """Handle training completion."""
        signalBus.trainingCompletedSig.emit(success, message)
        self._training_worker = None

    def _on_test_finished(self, success: bool, message: str, results: dict):
        """Handle test completion."""
        signalBus.testCompletedSig.emit(success, message, results)
        self._test_worker = None


# Global training service singleton
trainingService = TrainingService()
