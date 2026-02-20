# coding: utf-8
import os
from pathlib import Path
import re
import sys
from typing import Union
from json import loads

from PySide6.QtCore import QFile, QUrl, QFileInfo, QDir, QProcess, QStandardPaths
from PySide6.QtGui import QDesktopServices


def adjustFileName(name: str):
    """ adjust file name

    Returns
    -------
    name: str
        file name after adjusting
    """
    name = re.sub(r'[\\/:*?"<>|\r\n\s]+', "_", name.strip()).strip()
    return name.rstrip(".")


def readFile(filePath: str):
    """ load json data from file """
    file = QFile(filePath)
    file.open(QFile.OpenModeFlag.ReadOnly)
    data = str(file.readAll(), encoding='utf-8')
    file.close()
    return data


def loadJsonData(filePath: str):
    """ load json data from file """
    return loads(readFile(filePath))


def removeFile(filePath: str | Path):
    try:
        os.remove(filePath)
    except:
        pass


def openUrl(url: str):
    if not url.startswith("http"):
        if not os.path.exists(url):
            return False

        QDesktopServices.openUrl(QUrl.fromLocalFile(url))
    else:
        QDesktopServices.openUrl(QUrl(url))

    return True


def showInFolder(path: Union[str, Path]):
    """ show file in file explorer """
    if not os.path.exists(path):
        return False

    if isinstance(path, Path):
        path = str(path.absolute())

    if not path or path.lower().startswith('http'):
        return False

    info = QFileInfo(path)   # type:QFileInfo
    if sys.platform == "win32":
        args = [QDir.toNativeSeparators(path)]
        if not info.isDir():
            args.insert(0, '/select,')

        QProcess.startDetached('explorer', args)
    elif sys.platform == "darwin":
        args = [
            "-e", 'tell application "Finder"', "-e", "activate",
            "-e", f'select POSIX file "{path}"', "-e", "end tell",
            "-e", "return"
        ]
        QProcess.execute("/usr/bin/osascript", args)
    else:
        url = QUrl.fromLocalFile(path if info.isDir() else info.path())
        QDesktopServices.openUrl(url)

    return True


def runProcess(executable: Union[str, Path], args=None, timeout=5000, cwd=None) -> str:
    process = QProcess()

    if cwd:
        process.setWorkingDirectory(str(cwd))

    process.start(str(executable).replace("\\", "/"), args or [])
    process.waitForFinished(timeout)
    return process.readAllStandardOutput().toStdString()


def runDetachedProcess(executable: Union[str, Path], args=None, cwd=None):
    process = QProcess()

    if cwd:
        process.setWorkingDirectory(str(cwd))

    process.startDetached(str(executable).replace("\\", "/"), args or [])

def getSystemProxy():
    """ get system proxy """
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:
                enabled, _ = winreg.QueryValueEx(key, 'ProxyEnable')

                if enabled:
                    return "http://" + winreg.QueryValueEx(key, 'ProxyServer')
        except:
            pass
    elif sys.platform == "darwin":
        s = os.popen('scutil --proxy').read()
        info = dict(re.findall('(?m)^\s+([A-Z]\w+)\s+:\s+(\S+)', s))

        if info.get('HTTPEnable') == '1':
            return f"http://{info['HTTPProxy']}:{info['HTTPPort']}"
        elif info.get('ProxyAutoConfigEnable') == '1':
            return info['ProxyAutoConfigURLString']

    return os.environ.get("http_proxy")


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================
import platform
import shutil
import subprocess
from dataclasses import dataclass


def get_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return result.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Windows":
            return platform.processor() or "Unknown CPU"
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_system_memory_mb() -> int:
    """Get total system RAM in MB."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return int(result.stdout.strip()) // (1024 * 1024)
        elif system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) // 1024  # kB -> MB
        elif system == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    *[(f"ull{i}", ctypes.c_ulonglong) for i in range(6)],
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys // (1024 * 1024)
    except Exception:
        pass
    return 0


def get_os_display_name() -> str:
    """Get a human-readable OS name string."""
    system = platform.system()
    if system == "Darwin":
        ver = platform.mac_ver()[0]
        return f"macOS {ver}" if ver else "macOS"
    elif system == "Linux":
        try:
            import distro  # type: ignore
            return f"{distro.name()} {distro.version()}"
        except ImportError:
            return f"Linux {platform.release()}"
    elif system == "Windows":
        ver = platform.version()
        return f"Windows {platform.win32_ver()[1]}" if platform.win32_ver()[1] else f"Windows {ver}"
    return platform.platform()


# =============================================================================
# GPU DETECTION
# =============================================================================


@dataclass
class GPUInfo:
    """Information about a detected GPU."""

    index: int
    name: str
    memory_total: int  # MB
    memory_free: int  # MB
    compute_capability: str = ""


def detect_gpus() -> list:
    """Detect available GPUs across platforms.

    Returns:
        List of GPUInfo objects for each detected GPU.
    """
    gpus = []

    # Try NVIDIA GPUs first
    nvidia_gpus = _detect_nvidia_gpus()
    if nvidia_gpus:
        return nvidia_gpus

    # Try Apple Silicon (MPS)
    if platform.system() == "Darwin":
        mps_gpu = _detect_mps()
        if mps_gpu:
            return [mps_gpu]

    # No GPU found - return empty list
    return gpus


def _detect_nvidia_gpus() -> list:
    """Detect NVIDIA GPUs using nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total=int(float(parts[2])),
                        memory_free=int(float(parts[3])),
                    )
                )
        return gpus
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return []


def _detect_mps() -> GPUInfo | None:
    """Detect Apple Silicon GPU (MPS)."""
    try:
        import torch

        if torch.backends.mps.is_available():
            # Get approximate memory from system
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            total_mem = int(result.stdout.strip()) // (1024 * 1024)  # Convert to MB
            # MPS uses unified memory, estimate ~50% available for GPU
            return GPUInfo(
                index=0,
                name="Apple Silicon (MPS)",
                memory_total=total_mem // 2,
                memory_free=total_mem // 4,
            )
    except (ImportError, subprocess.CalledProcessError, ValueError):
        pass
    return None


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    return len(detect_gpus())


def get_gpu_summary() -> str:
    """Get a human-readable GPU summary."""
    gpus = detect_gpus()

    if not gpus:
        return "No GPU detected (CPU training only)"

    if len(gpus) == 1:
        gpu = gpus[0]
        return f"{gpu.name} ({gpu.memory_total} MB)"

    return f"{len(gpus)} GPUs: " + ", ".join(
        f"{gpu.name} ({gpu.memory_total} MB)" for gpu in gpus
    )


# =============================================================================
# DATA INSPECTION - Uses WaveDL library for consistent data loading
# =============================================================================
import numpy as np

# Import WaveDL data utilities for consistent key detection and loading
from wavedl.utils.data import (
    DataSource,
    get_data_source,
    INPUT_KEYS,
    OUTPUT_KEYS,
)


@dataclass
class DataInfo:
    """Information about a dataset file."""

    path: str
    format: str  # npz, mat, hdf5, unknown
    file_size: int  # bytes
    num_samples: int
    input_shape: tuple
    output_shape: tuple
    input_dtype: str
    output_dtype: str
    input_key: str = "X"
    output_key: str = "Y"
    dimensionality: str = ""  # 1D, 2D, 3D
    error: str = ""

    @property
    def file_size_str(self) -> str:
        """Human-readable file size."""
        size = self.file_size
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def num_outputs(self) -> int:
        """Number of output targets."""
        if len(self.output_shape) >= 1:
            return self.output_shape[-1]
        return 0


def inspect_data_file(path: str | Path) -> DataInfo:
    """Inspect a data file and extract metadata using WaveDL library.

    Uses the same data loading logic as WaveDL CLI to ensure consistency.

    Args:
        path: Path to the data file.

    Returns:
        DataInfo object with file metadata.
    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        return DataInfo(
            path=str(path),
            format="unknown",
            file_size=0,
            num_samples=0,
            input_shape=(),
            output_shape=(),
            input_dtype="",
            output_dtype="",
            error=f"File not found: {path}",
        )

    file_size = path.stat().st_size

    # Detect format using WaveDL's format detection
    try:
        format_str = DataSource.detect_format(str(path))
    except ValueError as e:
        return DataInfo(
            path=str(path),
            format="unknown",
            file_size=file_size,
            num_samples=0,
            input_shape=(),
            output_shape=(),
            input_dtype="",
            output_dtype="",
            error=str(e),
        )

    # Use WaveDL's data source to load and inspect data
    try:
        source = get_data_source(format_str)
        with source.load_mmap(str(path)) as (X, Y):
            input_shape = tuple(X.shape)
            output_shape = tuple(Y.shape)
            input_dtype = str(X.dtype)
            output_dtype = str(Y.dtype)

        # Determine which keys were used
        input_key, output_key = _detect_keys(str(path), format_str)

        return DataInfo(
            path=str(path),
            format=format_str,
            file_size=file_size,
            num_samples=input_shape[0] if input_shape else 0,
            input_shape=input_shape[1:] if len(input_shape) > 1 else (),
            output_shape=output_shape[1:] if len(output_shape) > 1 else (1,),
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            input_key=input_key,
            output_key=output_key,
            dimensionality=_get_dimensionality(input_shape),
        )
    except Exception as e:
        return DataInfo(
            path=str(path),
            format=format_str,
            file_size=file_size,
            num_samples=0,
            input_shape=(),
            output_shape=(),
            input_dtype="",
            output_dtype="",
            error=str(e),
        )

def _detect_keys(path: str, format_str: str) -> tuple[str, str]:
    """Detect the input/output keys used in a data file."""
    import h5py

    if format_str == "npz":
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.keys())
    elif format_str in ("hdf5", "mat"):
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
    else:
        return "X", "Y"

    input_key = DataSource._find_key(keys, INPUT_KEYS) or "X"
    output_key = DataSource._find_key(keys, OUTPUT_KEYS) or "Y"
    return input_key, output_key


def _get_dimensionality(shape: tuple) -> str:
    """Determine data dimensionality from input shape."""
    # Exclude batch dimension
    spatial_dims = len(shape) - 1

    # Account for channel dimension
    if spatial_dims <= 2:
        return "1D"
    elif spatial_dims == 3:
        return "2D"
    elif spatial_dims >= 4:
        return "3D"
    return "Unknown"


# =============================================================================
# TRAINING HISTORY PARSING
# =============================================================================
@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    r2_score: float = 0.0
    pearson: float = 0.0
    mae: list | None = None
    best_val_loss: float = 0.0
    patience_counter: int = 0


def parse_training_history(csv_path: str | Path) -> list:
    """Parse training_history.csv file.

    Args:
        csv_path: Path to training_history.csv

    Returns:
        List of TrainingMetrics for each epoch.
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    try:
        import pandas as pd

        df = pd.read_csv(path)

        metrics = []
        for _, row in df.iterrows():
            metrics.append(
                TrainingMetrics(
                    epoch=int(row.get("epoch", 0)),
                    train_loss=float(row.get("train_loss", 0)),
                    val_loss=float(row.get("val_loss", 0)),
                    learning_rate=float(row.get("learning_rate", row.get("lr", 0))),
                    r2_score=float(row.get("r2", row.get("r2_score", 0))),
                    pearson=float(row.get("pearson", 0)),
                )
            )
        return metrics
    except Exception:
        return []


# =============================================================================
# PYTHON/WAVEDL ENVIRONMENT DETECTION
# =============================================================================
def check_wavedl_installation() -> tuple:
    """Check if wavedl is installed and accessible.

    Returns:
        Tuple of (is_installed, version_or_error)
    """
    try:
        import wavedl

        return True, getattr(wavedl, "__version__", "unknown")
    except ImportError as e:
        return False, str(e)


def check_pytorch_installation() -> tuple:
    """Check PyTorch installation and CUDA availability.

    Returns:
        Tuple of (is_installed, version_or_error, cuda_available)
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
        return True, torch.__version__, cuda
    except ImportError as e:
        return False, str(e), False


def get_python_executable() -> str:
    """Get the path to the current Python executable."""
    return sys.executable

