"""
Checkpoint Crash-Recovery Tests for WaveDL
===========================================

Regression tests for progress loss on preemptible machines (Colab, SLURM,
spot instances), where the process is killed without unwinding.

The original failure: `best_checkpoint` only advances when val loss improves
and periodic checkpoints defaulted to every 50 epochs, so a plateaued run that
was hard-killed rewound to the last *improvement* — in the reported case,
9 epochs (~5 GPU-hours) discarded on resume.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import pickle

import pandas as pd
import pytest

from wavedl.train import (
    CHECKPOINT_SENTINEL,
    _finalize_checkpoint,
    checkpoint_is_usable,
    resolve_checkpoint,
    write_csv_atomic,
)


# ==============================================================================
# HELPERS
# ==============================================================================
def make_checkpoint(root, name, epoch, *, sealed=True, meta=True):
    """Create a checkpoint directory mimicking an accelerate save_state dump."""
    ckpt = os.path.join(root, name)
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, "model.bin"), "wb") as f:
        f.write(b"weights")
    if meta:
        with open(os.path.join(ckpt, "training_meta.pkl"), "wb") as f:
            pickle.dump({"epoch": epoch, "best_val_loss": 1.0, "patience_ctr": 0}, f)
    if sealed:
        with open(os.path.join(ckpt, CHECKPOINT_SENTINEL), "w") as f:
            f.write("ok\n")
    return ckpt


def select_resume_checkpoint(output_dir):
    """Mirror of train.py's auto-resume selection: highest complete epoch wins.

    Kept in sync with the scan in `train_worker`; exercised here without
    standing up an Accelerator and a full training loop.
    """
    candidates = [
        os.path.join(output_dir, n)
        for n in ("last_checkpoint", "best_checkpoint", "interrupted_checkpoint")
    ]
    candidates += [
        os.path.join(output_dir, e)
        for e in os.listdir(output_dir)
        if e.startswith("epoch_") and e.endswith("_checkpoint")
    ]

    best_path, best_epoch = None, -1
    for candidate in candidates:
        resolved = resolve_checkpoint(candidate)
        if resolved is None:
            continue
        meta_file = os.path.join(resolved, "training_meta.pkl")
        epoch = 0
        if os.path.exists(meta_file):
            with open(meta_file, "rb") as f:
                epoch = pickle.load(f).get("epoch", 0)
        if epoch > best_epoch:
            best_path, best_epoch = resolved, epoch
    return best_path, best_epoch


# ==============================================================================
# ROOT-CAUSE REGRESSION
# ==============================================================================
class TestPlateauedRunRecovery:
    """The reported bug: plateau + hard kill = rewind to last improvement."""

    def test_resumes_from_rolling_checkpoint_not_stale_best(self, tmp_path):
        """Reproduces the reported loss exactly.

        best_checkpoint is pinned at epoch 6 (val loss never improved after
        that); the rolling checkpoint reached epoch 15. Resume must pick 15.
        Before the fix, last_checkpoint did not exist and resume yielded 6.
        """
        root = str(tmp_path)
        make_checkpoint(root, "best_checkpoint", epoch=6)
        make_checkpoint(root, "last_checkpoint", epoch=15)

        resumed, epoch = select_resume_checkpoint(root)

        assert epoch == 15, f"Rewound to epoch {epoch}, losing 9 epochs of training"
        assert os.path.basename(resumed) == "last_checkpoint"

    def test_prefers_best_when_it_is_the_newer_one(self, tmp_path):
        """Right after an improvement, best_checkpoint is ahead of the rolling one."""
        root = str(tmp_path)
        make_checkpoint(root, "best_checkpoint", epoch=20)
        make_checkpoint(root, "last_checkpoint", epoch=18)

        resumed, epoch = select_resume_checkpoint(root)

        assert epoch == 20
        assert os.path.basename(resumed) == "best_checkpoint"

    def test_emergency_checkpoint_wins_when_furthest_along(self, tmp_path):
        """A SIGTERM-triggered save lands past both other checkpoints."""
        root = str(tmp_path)
        make_checkpoint(root, "best_checkpoint", epoch=6)
        make_checkpoint(root, "last_checkpoint", epoch=15)
        make_checkpoint(root, "interrupted_checkpoint", epoch=16)

        _resumed, epoch = select_resume_checkpoint(root)

        assert epoch == 16


# ==============================================================================
# CORRUPTION RESISTANCE
# ==============================================================================
class TestIncompleteCheckpointHandling:
    """Killed mid-write on a FUSE mount must not destroy the recovery point."""

    def test_unsealed_checkpoint_with_no_metadata_is_rejected(self, tmp_path):
        ckpt = make_checkpoint(tmp_path, "last_checkpoint", 9, sealed=False, meta=False)
        assert checkpoint_is_usable(ckpt) is False
        assert resolve_checkpoint(ckpt) is None

    def test_falls_back_to_prev_generation(self, tmp_path):
        """A crash during the final swap leaves only `.prev` intact."""
        root = str(tmp_path)
        make_checkpoint(root, "last_checkpoint", 9, sealed=False, meta=False)
        make_checkpoint(root, "last_checkpoint.prev", epoch=8)

        resolved = resolve_checkpoint(os.path.join(root, "last_checkpoint"))

        assert resolved is not None
        assert resolved.endswith(".prev")

    def test_incomplete_checkpoint_does_not_shadow_older_good_one(self, tmp_path):
        """A truncated newer checkpoint must not be preferred over a valid older one."""
        root = str(tmp_path)
        make_checkpoint(root, "best_checkpoint", epoch=6)
        make_checkpoint(root, "last_checkpoint", 15, sealed=False, meta=False)

        resumed, epoch = select_resume_checkpoint(root)

        assert epoch == 6
        assert os.path.basename(resumed) == "best_checkpoint"

    def test_legacy_checkpoint_without_sentinel_still_resumable(self, tmp_path):
        """Backward compat: checkpoints written before sentinels existed.

        Users mid-run when upgrading must not have their existing
        best_checkpoint on Drive rejected as incomplete.
        """
        ckpt = make_checkpoint(tmp_path, "best_checkpoint", 6, sealed=False, meta=True)
        assert checkpoint_is_usable(ckpt) is True

    def test_tmp_staging_dir_is_never_selected(self, tmp_path):
        """`.tmp` leftovers from a killed save must be invisible to resume."""
        root = str(tmp_path)
        make_checkpoint(root, "best_checkpoint", epoch=6)
        make_checkpoint(root, "epoch_99_checkpoint.tmp", epoch=99)

        _resumed, epoch = select_resume_checkpoint(root)

        assert epoch == 6, "Half-written staging directory was resumed from"


class TestFinalizeCheckpoint:
    """The atomic swap itself."""

    def test_swap_seals_and_replaces(self, tmp_path):
        root = str(tmp_path)
        final = os.path.join(root, "last_checkpoint")
        make_checkpoint(root, "last_checkpoint", epoch=4)
        tmp_dir = make_checkpoint(root, "last_checkpoint.tmp", 5, sealed=False)

        _finalize_checkpoint(tmp_dir, final)

        assert checkpoint_is_usable(final)
        with open(os.path.join(final, "training_meta.pkl"), "rb") as f:
            assert pickle.load(f)["epoch"] == 5
        assert not os.path.exists(tmp_dir), "staging dir left behind"
        assert not os.path.exists(f"{final}.prev"), "prev generation left behind"

    def test_swap_works_with_no_existing_checkpoint(self, tmp_path):
        root = str(tmp_path)
        final = os.path.join(root, "last_checkpoint")
        tmp_dir = make_checkpoint(root, "last_checkpoint.tmp", 1, sealed=False)

        _finalize_checkpoint(tmp_dir, final)

        assert checkpoint_is_usable(final)

    def test_sentinel_written_last(self, tmp_path):
        """Ordering is what makes the sentinel meaningful as a completeness mark."""
        root = str(tmp_path)
        tmp_dir = make_checkpoint(root, "c.tmp", 1, sealed=False)
        sentinel = os.path.join(tmp_dir, CHECKPOINT_SENTINEL)
        assert not os.path.exists(sentinel)

        _finalize_checkpoint(tmp_dir, os.path.join(root, "c"))

        assert os.path.exists(os.path.join(root, "c", CHECKPOINT_SENTINEL))


class TestAtomicCsv:
    def test_history_survives_and_replaces(self, tmp_path):
        path = os.path.join(str(tmp_path), "training_history.csv")
        write_csv_atomic(pd.DataFrame([{"epoch": 1}, {"epoch": 2}]), path)
        write_csv_atomic(pd.DataFrame([{"epoch": i} for i in range(1, 16)]), path)

        assert len(pd.read_csv(path)) == 15
        assert not os.path.exists(f"{path}.tmp"), "temp file left behind"


# ==============================================================================
# ARGUMENT DEFAULTS
# ==============================================================================
class TestSaveEveryDefault:
    def test_default_bounds_loss_to_one_epoch(self):
        """A 50-epoch default meant a ~25h exposure window at 28min/epoch."""
        import sys

        from wavedl.train import parse_args

        argv = sys.argv
        try:
            sys.argv = ["wavedl-train", "--model", "cnn", "--data_path", "x.h5"]
            args, _parser = parse_args()
        finally:
            sys.argv = argv

        assert args.save_every == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
