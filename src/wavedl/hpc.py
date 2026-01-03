#!/usr/bin/env python
"""
WaveDL HPC Training Launcher.

This module provides a Python-based HPC training launcher that wraps accelerate
for distributed training on High-Performance Computing clusters.

Usage:
    wavedl-hpc --model cnn --data_path train.npz --num_gpus 4

Example SLURM script:
    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --gpus-per-node=4
    #SBATCH --time=12:00:00

    wavedl-hpc --model cnn --data_path /scratch/data.npz --compile

Author: Ductho Le (ductho.le@outlook.com)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def detect_gpus() -> int:
    """Auto-detect available GPUs using nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        print("Warning: nvidia-smi not found, defaulting to NUM_GPUS=1")
        return 1

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_count = len(result.stdout.strip().split("\n"))
        if gpu_count > 0:
            print(f"Auto-detected {gpu_count} GPU(s)")
            return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("Warning: No GPUs detected, defaulting to NUM_GPUS=1")
    return 1


def setup_hpc_environment() -> None:
    """Configure environment variables for HPC systems.

    Handles restricted home directories (e.g., Compute Canada) and
    offline logging configurations.
    """
    # Use SLURM_TMPDIR if available, otherwise system temp
    tmpdir = os.environ.get(
        "SLURM_TMPDIR", os.environ.get("TMPDIR", tempfile.gettempdir())
    )

    # Configure directories for systems with restricted home directories
    os.environ.setdefault("MPLCONFIGDIR", f"{tmpdir}/matplotlib")
    os.environ.setdefault(
        "FONTCONFIG_PATH", os.environ.get("FONTCONFIG_PATH", "/etc/fonts")
    )
    os.environ.setdefault("XDG_CACHE_HOME", f"{tmpdir}/.cache")

    # Ensure matplotlib config dir exists
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    # WandB configuration (offline by default for HPC)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", f"{tmpdir}/wandb")
    os.environ.setdefault("WANDB_CACHE_DIR", f"{tmpdir}/wandb_cache")
    os.environ.setdefault("WANDB_CONFIG_DIR", f"{tmpdir}/wandb_config")

    # Suppress non-critical warnings
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse HPC-specific arguments, pass remaining to wavedl.train."""
    parser = argparse.ArgumentParser(
        description="WaveDL HPC Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with auto-detected GPUs
  wavedl-hpc --model cnn --data_path train.npz --epochs 100

  # Specify GPU count and mixed precision
  wavedl-hpc --model cnn --data_path train.npz --num_gpus 4 --mixed_precision bf16

  # Full configuration
  wavedl-hpc --model resnet18 --data_path train.npz --num_gpus 8 \\
             --batch_size 256 --lr 1e-3 --compile --output_dir ./results

Environment Variables:
  WANDB_MODE          WandB mode: offline|online (default: offline)
  SLURM_TMPDIR        Temp directory for HPC systems
""",
    )

    # HPC-specific arguments
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )
    parser.add_argument(
        "--num_machines",
        type=int,
        default=1,
        help="Number of machines for multi-node training (default: 1)",
    )
    parser.add_argument(
        "--machine_rank",
        type=int,
        default=0,
        help="Rank of this machine in multi-node setup (default: 0)",
    )
    parser.add_argument(
        "--main_process_ip",
        type=str,
        default=None,
        help="IP address of the main process for multi-node training",
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=None,
        help="Port for multi-node communication (default: accelerate auto-selects)",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["bf16", "fp16", "no"],
        default="bf16",
        help="Mixed precision mode (default: bf16)",
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="no",
        help="PyTorch dynamo backend (default: no)",
    )

    # Parse known args, pass rest to wavedl.train
    args, remaining = parser.parse_known_args()
    return args, remaining


def print_summary(exit_code: int, wandb_mode: str, wandb_dir: str) -> None:
    """Print post-training summary and instructions."""
    print()
    print("=" * 40)

    if exit_code == 0:
        print("✅ Training completed successfully!")
        print("=" * 40)

        if wandb_mode == "offline":
            print()
            print("📊 WandB Sync Instructions:")
            print("   From the login node, run:")
            print(f"   wandb sync {wandb_dir}/wandb/offline-run-*")
            print()
            print("   This will upload your training logs to wandb.ai")
    else:
        print(f"❌ Training failed with exit code: {exit_code}")
        print("=" * 40)
        print()
        print("Common issues:")
        print("  - Missing data file (check --data_path)")
        print("  - Insufficient GPU memory (reduce --batch_size)")
        print("  - Invalid model name (run: python train.py --list_models)")
        print()

    print("=" * 40)
    print()


def main() -> int:
    """Main entry point for wavedl-hpc command."""
    # Parse arguments
    args, train_args = parse_args()

    # Setup HPC environment
    setup_hpc_environment()

    # Check if wavedl package is importable
    try:
        import wavedl  # noqa: F401
    except ImportError:
        print("Error: wavedl package not found. Run: pip install -e .", file=sys.stderr)
        return 1

    # Auto-detect GPUs if not specified
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
        print(f"Using NUM_GPUS={num_gpus} (set via command line)")
    else:
        num_gpus = detect_gpus()

    # Build accelerate launch command
    cmd = [
        "accelerate",
        "launch",
        f"--num_processes={num_gpus}",
        f"--num_machines={args.num_machines}",
        f"--machine_rank={args.machine_rank}",
        f"--mixed_precision={args.mixed_precision}",
        f"--dynamo_backend={args.dynamo_backend}",
    ]

    # Add multi-node networking args if specified (required for some clusters)
    if args.main_process_ip:
        cmd.append(f"--main_process_ip={args.main_process_ip}")
    if args.main_process_port:
        cmd.append(f"--main_process_port={args.main_process_port}")

    cmd += ["-m", "wavedl.train"] + train_args

    # Create output directory if specified
    for i, arg in enumerate(train_args):
        if arg == "--output_dir" and i + 1 < len(train_args):
            Path(train_args[i + 1]).mkdir(parents=True, exist_ok=True)
            break
        if arg.startswith("--output_dir="):
            Path(arg.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
            break

    # Launch training
    try:
        result = subprocess.run(cmd, check=False)
        exit_code = result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        exit_code = 130

    # Print summary
    print_summary(
        exit_code,
        os.environ.get("WANDB_MODE", "offline"),
        os.environ.get("WANDB_DIR", "/tmp/wandb"),
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
