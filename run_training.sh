#!/bin/bash
################################################################################
# WaveDL Training Launcher for HPC Environments
################################################################################
#
# Description:
#   Configures environment variables and launches distributed training with
#   Accelerate on High-Performance Computing clusters (e.g., Compute Canada).
#   Automatically handles system-specific quirks like restricted home directories
#   and offline WandB logging.
#
# Author: Ductho Le (ductho.le@outlook.com)
#
#
# Usage:
#   ./run_training.sh [OPTIONS]
#
# Examples:
#   # Basic training
#   ./run_training.sh --model cnn --batch_size 128 --epochs 500
#
#   # Full configuration
#   ./run_training.sh --model cnn --data_path train_data.npz \
#     --batch_size 128 --lr 1e-3 --patience 30 --compile \
#     --output_dir ./results
#
#   # Custom GPU configuration
#   NUM_GPUS=2 MIXED_PRECISION=fp16 ./run_training.sh --model cnn
#
#   # K-fold cross-validation
#   ./run_training.sh --model cnn --cv 5 --data_path train_data.npz
#
# Environment Variables:
#   NUM_GPUS          Number of GPUs to use (default: auto-detected)
#   NUM_MACHINES      Number of machines in distributed setup (default: 1)
#   MIXED_PRECISION   Precision mode: bf16|fp16|no (default: bf16)
#   DYNAMO_BACKEND    PyTorch dynamo backend (default: no)
#   WANDB_MODE        WandB logging mode: offline|online (default: offline)
#
################################################################################

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

# HPC-specific paths (Compute Canada uses SLURM_TMPDIR, fallback to /tmp)
TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

# Configure directories for systems with restricted home directories
export MPLCONFIGDIR="${TMPDIR}/matplotlib"
export FONTCONFIG_PATH="${FONTCONFIG_PATH:-/etc/fonts}"
export XDG_CACHE_HOME="${TMPDIR}/.cache"

# WandB configuration
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${TMPDIR}/wandb"
export WANDB_CACHE_DIR="${TMPDIR}/wandb_cache"
export WANDB_CONFIG_DIR="${TMPDIR}/wandb_config"

# Suppress non-critical Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

# ==============================================================================
# ACCELERATE CONFIGURATION
# ==============================================================================

# Auto-detect available GPUs if NUM_GPUS is not set
if [[ -z "${NUM_GPUS:-}" ]]; then
    # Try to detect GPUs using nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$DETECTED_GPUS" -gt 0 ]]; then
            NUM_GPUS="$DETECTED_GPUS"
            echo "Auto-detected $NUM_GPUS GPU(s)"
        else
            NUM_GPUS=1
            echo "Warning: No GPUs detected, defaulting to NUM_GPUS=1"
        fi
    else
        NUM_GPUS=1
        echo "Warning: nvidia-smi not found, defaulting to NUM_GPUS=1"
    fi
else
    echo "Using NUM_GPUS=$NUM_GPUS (set via environment variable)"
fi

# Set other defaults (can be overridden via environment variables)
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DYNAMO_BACKEND="${DYNAMO_BACKEND:-no}"

# Validate mixed precision setting
if [[ ! "$MIXED_PRECISION" =~ ^(bf16|fp16|no)$ ]]; then
    echo "Error: MIXED_PRECISION must be 'bf16', 'fp16', or 'no'. Got: $MIXED_PRECISION" >&2
    exit 1
fi

# ==============================================================================
# LAUNCH TRAINING
# ==============================================================================

# Check if wavedl package is importable
if ! python -c "import wavedl" 2>/dev/null; then
    echo "Error: wavedl package not found. Run: pip install -e ." >&2
    exit 1
fi

# Parse output_dir from arguments and create it if specified
prev_arg=""
for i in "$@"; do
    if [[ "$i" == --output_dir=* ]]; then
        OUTPUT_DIR="${i#*=}"
        mkdir -p "$OUTPUT_DIR"
        break
    elif [[ "$prev_arg" == "--output_dir" ]]; then
        OUTPUT_DIR="$i"
        mkdir -p "$OUTPUT_DIR"
        break
    fi
    prev_arg="$i"
done

# Launch training with Accelerate
# Temporarily disable exit-on-error to capture exit code for post-training summary
set +e
accelerate launch \
  --num_processes="$NUM_GPUS" \
  --num_machines="$NUM_MACHINES" \
  --machine_rank=0 \
  --mixed_precision="$MIXED_PRECISION" \
  --dynamo_backend="$DYNAMO_BACKEND" \
  -m wavedl.train \
  "$@"
EXIT_CODE=$?
set -e

# ==============================================================================
# POST-TRAINING SUMMARY
# ==============================================================================

echo ""
echo "========================================"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Training completed successfully!"
    echo "========================================"

    # WandB sync instructions for offline mode
    if [[ "$WANDB_MODE" == "offline" ]]; then
        echo ""
        echo "📊 WandB Sync Instructions:"
        echo "   From the login node, run:"
        echo "   wandb sync ${WANDB_DIR}/wandb/offline-run-*"
        echo ""
        echo "   This will upload your training logs to wandb.ai"
    fi
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - Missing data file (check --data_path)"
    echo "  - Insufficient GPU memory (reduce --batch_size)"
    echo "  - Invalid model name (run: python train.py --list_models)"
    echo ""
fi

echo "========================================"
echo ""

exit $EXIT_CODE
