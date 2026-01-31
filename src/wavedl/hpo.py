"""
WaveDL - Hyperparameter Optimization with Optuna
=================================================
Automated hyperparameter search for finding optimal training configurations.

Usage:
    # Basic HPO (50 trials)
    wavedl-hpo --data_path train.npz --n_trials 50

    # Quick search (fewer parameters)
    wavedl-hpo --data_path train.npz --n_trials 30 --quick

    # Full search with specific models
    wavedl-hpo --data_path train.npz --n_trials 100 --models cnn resnet18 efficientnet_b0

    # Parallel trials on multiple GPUs
    wavedl-hpo --data_path train.npz --n_trials 100 --n_jobs 4

Author: Ductho Le (ductho.le@outlook.com)
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    print("Error: Optuna not installed. Run: pip install wavedl")
    sys.exit(1)


# =============================================================================
# DEFAULT SEARCH SPACES
# =============================================================================

DEFAULT_MODELS = ["cnn", "resnet18", "resnet34"]
QUICK_MODELS = ["cnn"]

# All 6 optimizers
DEFAULT_OPTIMIZERS = ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]
QUICK_OPTIMIZERS = ["adamw"]

# All 8 schedulers
DEFAULT_SCHEDULERS = [
    "plateau",
    "cosine",
    "cosine_restarts",
    "onecycle",
    "step",
    "multistep",
    "exponential",
    "linear_warmup",
]
QUICK_SCHEDULERS = ["plateau"]

# All 6 losses
DEFAULT_LOSSES = ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]
QUICK_LOSSES = ["mse"]


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================


def create_objective(args):
    """Create Optuna objective function with configurable search space."""

    def objective(trial):
        # Select search space based on mode
        # CLI arguments always take precedence over defaults
        if args.quick:
            models = args.models or QUICK_MODELS
            optimizers = args.optimizers or QUICK_OPTIMIZERS
            schedulers = args.schedulers or QUICK_SCHEDULERS
            losses = args.losses or QUICK_LOSSES
        else:
            models = args.models or DEFAULT_MODELS
            optimizers = args.optimizers or DEFAULT_OPTIMIZERS
            schedulers = args.schedulers or DEFAULT_SCHEDULERS
            losses = args.losses or DEFAULT_LOSSES

        # Suggest hyperparameters
        model = trial.suggest_categorical("model", models)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_sizes = args.batch_sizes or [16, 32, 64, 128]
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)
        optimizer = trial.suggest_categorical("optimizer", optimizers)
        scheduler = trial.suggest_categorical("scheduler", schedulers)
        loss = trial.suggest_categorical("loss", losses)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        patience = trial.suggest_int("patience", 10, 30, step=5)

        # Conditional hyperparameters
        if loss == "huber":
            huber_delta = trial.suggest_float("huber_delta", 0.1, 2.0)
        else:
            huber_delta = None

        if optimizer == "sgd":
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
        else:
            momentum = None

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "wavedl.train",
            "--data_path",
            str(args.data_path),
            "--model",
            model,
            "--lr",
            str(lr),
            "--batch_size",
            str(batch_size),
            "--optimizer",
            optimizer,
            "--scheduler",
            scheduler,
            "--loss",
            loss,
            "--weight_decay",
            str(weight_decay),
            "--patience",
            str(patience),
            "--epochs",
            str(args.max_epochs),
            "--seed",
            str(args.seed),
        ]

        # Add conditional args
        if huber_delta:
            cmd.extend(["--huber_delta", str(huber_delta)])
        if momentum:
            cmd.extend(["--momentum", str(momentum)])

        # Use temporary directory for trial output
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd.extend(["--output_dir", tmpdir])
            history_file = Path(tmpdir) / "training_history.csv"

            # GPU isolation for parallel trials: assign each trial to a specific GPU
            # This prevents multiple trials from competing for all GPUs
            env = None
            if args.n_jobs > 1:
                import os

                # Detect available GPUs
                n_gpus = 1
                try:
                    import subprocess as sp

                    result_gpu = sp.run(
                        ["nvidia-smi", "--list-gpus"],
                        capture_output=True,
                        text=True,
                    )
                    if result_gpu.returncode == 0:
                        n_gpus = len(result_gpu.stdout.strip().split("\n"))
                except Exception:
                    pass

                # Assign trial to a specific GPU (round-robin)
                gpu_id = trial.number % n_gpus
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Run training
            # Note: We inherit the user's cwd instead of setting cwd=Path(__file__).parent
            # because site-packages may be read-only and train.py creates cache directories
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    env=env,
                )

                # Read best val_loss from training_history.csv (reliable machine-readable)
                val_loss = None
                if history_file.exists():
                    try:
                        import csv

                        with open(history_file) as f:
                            reader = csv.DictReader(f)
                            val_losses = []
                            for row in reader:
                                if "val_loss" in row:
                                    try:
                                        val_losses.append(float(row["val_loss"]))
                                    except (ValueError, TypeError):
                                        pass
                            if val_losses:
                                val_loss = min(val_losses)  # Best (minimum) val_loss
                    except Exception as e:
                        print(f"Trial {trial.number}: Error reading history: {e}")

                if val_loss is None:
                    # Fallback: parse stdout for training log format
                    # Pattern: "epoch | train_loss | val_loss | ..."
                    # Use regex to avoid false positives from unrelated lines
                    import re

                    # Match lines like: "  42  | 0.0123   | 0.0156   | ..."
                    log_pattern = re.compile(
                        r"^\s*\d+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
                    )
                    val_losses_stdout = []
                    for line in result.stdout.split("\n"):
                        match = log_pattern.match(line)
                        if match:
                            try:
                                val_losses_stdout.append(float(match.group(1)))
                            except ValueError:
                                continue
                    if val_losses_stdout:
                        val_loss = min(val_losses_stdout)

                if val_loss is None:
                    # Training failed or no loss found
                    print(f"Trial {trial.number}: Training failed (no val_loss found)")
                    if result.returncode != 0:
                        # Show last few lines of stderr for debugging
                        stderr_lines = result.stderr.strip().split("\n")[-3:]
                        for line in stderr_lines:
                            print(f"  stderr: {line}")
                    return float("inf")

                print(f"Trial {trial.number}: val_loss={val_loss:.6f}")
                return val_loss

            except subprocess.TimeoutExpired:
                print(f"Trial {trial.number}: Timeout after {args.timeout}s")
                return float("inf")
            except Exception as e:
                print(f"Trial {trial.number}: Error - {e}")
                return float("inf")

    return objective


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="WaveDL Hyperparameter Optimization with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    wavedl-hpo --data_path train.npz --n_trials 50
    wavedl-hpo --data_path train.npz --n_trials 30 --quick
    wavedl-hpo --data_path train.npz --n_trials 100 --models cnn resnet18
        """,
    )

    # Required
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data"
    )

    # HPO settings
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of HPO trials (default: 50)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel trials (-1 = auto-detect GPUs, default: -1)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: search fewer parameters",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per trial in seconds (default: 3600)",
    )

    # Search space customization
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to search (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        help=f"Optimizers to search (default: {DEFAULT_OPTIMIZERS})",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=None,
        help=f"Schedulers to search (default: {DEFAULT_SCHEDULERS})",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=None,
        help=f"Losses to search (default: {DEFAULT_LOSSES})",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to search (default: 16 32 64 128)",
    )

    # Training settings for each trial
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Max epochs per trial (default: 50, use early stopping)",
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed (default: 2025)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="hpo_results.json",
        help="Output file for best params (default: hpo_results.json)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="wavedl_hpo",
        help="Optuna study name (default: wavedl_hpo)",
    )

    args = parser.parse_args()

    # Convert to absolute path (child processes may run in different cwd)
    args.data_path = str(Path(args.data_path).resolve())

    # Validate data path
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)

    # Auto-detect GPUs for n_jobs if not specified
    if args.n_jobs == -1:
        try:
            result_gpu = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
            )
            if result_gpu.returncode == 0:
                args.n_jobs = max(1, len(result_gpu.stdout.strip().split("\n")))
            else:
                args.n_jobs = 1
        except Exception:
            args.n_jobs = 1
        print(f"Auto-detected {args.n_jobs} GPU(s) for parallel trials")

    # Create study
    print("=" * 60)
    print("WaveDL Hyperparameter Optimization")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Trials: {args.n_trials}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Parallel jobs: {args.n_jobs}")
    print("=" * 60)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Run optimization
    objective = create_objective(args)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    # Filter completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed_trials:
        print("No trials completed successfully.")
        sys.exit(1)

    print(f"\nCompleted trials: {len(completed_trials)}/{args.n_trials}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.6f}")

    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(completed_trials),
        "study_name": args.study_name,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Print command to train with best params
    print("\n" + "=" * 60)
    print("TO TRAIN WITH BEST PARAMETERS:")
    print("=" * 60)
    cmd_parts = ["wavedl-train"]
    cmd_parts.append(f"--data_path {args.data_path}")
    for key, value in study.best_params.items():
        cmd_parts.append(f"--{key} {value}")
    print(" \\\n    ".join(cmd_parts))


if __name__ == "__main__":
    main()
