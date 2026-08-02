import copy
import json

from wavedl.runtime_protocol import LEGACY_METRICS_PREFIX, encode_event
from wavedl_gui.service.training_service import OutputParser


RUN_ID = "123e4567-e89b-12d3-a456-426614174000"


def metric_payload():
    return {
        "epoch": 3,
        "total_epochs": 10,
        "train_loss": 0.2,
        "val_loss": 0.25,
        "best_val_loss": 0.18,
        "r2_score": 0.9,
        "pearson": 0.95,
        "grad_norm": 0.4,
        "learning_rate": 0.001,
        "mae_avg": 0.12,
        "mae_per_param": [0.1, 0.14],
        "time_per_epoch": 2.5,
        "total_time": 7.5,
        "patience_counter": 2,
        "max_patience": 5,
    }


def v1_metric_line(payload):
    return encode_event(
        "metric",
        payload,
        run_id=RUN_ID,
        seq=3,
        ts="2026-01-02T03:04:05Z",
    )


def test_v1_metric_event_updates_progress_and_eta():
    parser = OutputParser()

    progress, is_metrics = parser.parse_line(v1_metric_line(metric_payload()))

    assert is_metrics is True
    assert progress.epoch == 3
    assert progress.total_epochs == 10
    assert progress.r2_score == 0.9
    assert progress.learning_rate == 0.001
    assert progress.time_per_epoch == 2.5
    assert progress.eta_seconds == 17.5


def test_legacy_metric_line_is_normalized_and_updates_identically():
    payload = metric_payload()
    payload["r2"] = payload.pop("r2_score")
    payload["lr"] = payload.pop("learning_rate")
    payload["epoch_time"] = payload.pop("time_per_epoch")
    parser = OutputParser()

    progress, is_metrics = parser.parse_line(
        LEGACY_METRICS_PREFIX + json.dumps(payload)
    )

    assert is_metrics is True
    assert progress.r2_score == 0.9
    assert progress.learning_rate == 0.001
    assert progress.time_per_epoch == 2.5
    assert progress.eta_seconds == 17.5


def test_blank_and_ordinary_lines_remain_visible_and_preserve_progress():
    parser = OutputParser()
    parser.progress.epoch = 4

    for line in ("", "   ", "ordinary training log"):
        progress, is_metrics = parser.parse_line(line)
        assert is_metrics is False
        assert progress.epoch == 4


def test_malformed_v1_and_legacy_lines_are_non_fatal_visible_logs():
    parser = OutputParser()
    parser.progress.epoch = 4

    for line in (
        '{"protocol":"wavedl-jsonl","version":1}',
        LEGACY_METRICS_PREFIX + '{"epoch":',
    ):
        progress, is_metrics = parser.parse_line(line)
        assert is_metrics is False
        assert progress.epoch == 4


def test_valid_non_metric_protocol_event_is_visible_and_does_not_update():
    parser = OutputParser()
    parser.progress.epoch = 4

    progress, is_metrics = parser.parse_line(
        encode_event("log", {"message": "hello"}, run_id=RUN_ID, seq=4)
    )

    assert is_metrics is False
    assert progress.epoch == 4


def test_metric_payload_is_not_mutated():
    payload = metric_payload()
    original = copy.deepcopy(payload)
    parser = OutputParser()

    parser.parse_line(v1_metric_line(payload))

    assert payload == original
    parser.progress.mae_per_param.append(99.0)
    assert payload["mae_per_param"] == original["mae_per_param"]
