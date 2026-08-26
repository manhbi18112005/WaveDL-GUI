# ruff: noqa: I001

import copy
import json
import math
from dataclasses import asdict

import pytest

pytest.importorskip("PySide6")

from wavedl.runtime_protocol import (
    LEGACY_METRICS_PREFIX,
    encode_event,
    parse_jsonl_line,
)
from wavedl_gui.service.training_service import (
    OutputKind,
    OutputParseResult,
    OutputParser,
    TrainingWorker,
    TrainingProgress,
)


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


def v1_metric_line(payload, *, seq=0, run_id=RUN_ID):
    return encode_event(
        "metric",
        payload,
        run_id=run_id,
        seq=seq,
        ts="2026-01-02T03:04:05Z",
    )


def protocol_line(event_type, payload, *, seq=0, run_id=RUN_ID):
    return encode_event(
        event_type,
        payload,
        run_id=run_id,
        seq=seq,
        ts="2026-01-02T03:04:05Z",
    )


def test_v1_and_legacy_metrics_produce_equivalent_full_progress():
    canonical_payload = metric_payload()
    legacy_payload = canonical_payload.copy()
    legacy_payload["r2"] = legacy_payload.pop("r2_score")
    legacy_payload["lr"] = legacy_payload.pop("learning_rate")
    legacy_payload["epoch_time"] = legacy_payload.pop("time_per_epoch")

    v1_parser = OutputParser()
    legacy_parser = OutputParser()
    v1_progress, v1_is_metrics = v1_parser.parse_line(v1_metric_line(canonical_payload))
    legacy_progress, legacy_is_metrics = legacy_parser.parse_line(
        LEGACY_METRICS_PREFIX + json.dumps(legacy_payload)
    )

    assert v1_is_metrics is True
    assert legacy_is_metrics is True
    expected = TrainingProgress(
        epoch=3,
        total_epochs=10,
        train_loss=0.2,
        val_loss=0.25,
        learning_rate=0.001,
        best_val_loss=0.18,
        patience_counter=2,
        max_patience=5,
        r2_score=0.9,
        pearson=0.95,
        grad_norm=0.4,
        mae_avg=0.12,
        mae_per_param=[0.1, 0.14],
        time_per_epoch=2.5,
        total_time=7.5,
        eta_seconds=17.5,
    )
    assert asdict(v1_progress) == asdict(legacy_progress)
    assert asdict(v1_progress) == asdict(expected)


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


def test_output_classification_distinguishes_logs_protocol_events_and_metrics():
    parser = OutputParser()

    assert parser._parse_output("ordinary training log").kind is OutputKind.ORDINARY_LOG
    assert (
        parser._parse_output('{"protocol":"wavedl-jsonl","version":1}').kind
        is OutputKind.MALFORMED_PROTOCOL
    )
    assert (
        parser._parse_output(
            encode_event("log", {"message": "hello"}, run_id=RUN_ID, seq=0)
        ).kind
        is OutputKind.PROTOCOL_EVENT
    )
    assert (
        parser._parse_output(v1_metric_line(metric_payload(), seq=1)).kind
        is OutputKind.METRIC
    )


def test_public_parse_result_retains_protocol_event_and_raw_line():
    line = protocol_line("warning", {"message": "watch out", "code": 7})
    parser = OutputParser()

    result = parser.parse_result(line)

    assert isinstance(result, OutputParseResult)
    assert result.kind is OutputKind.PROTOCOL_EVENT
    assert result.raw_line == line
    assert result.event == parse_jsonl_line(line)
    assert result.event is not None
    assert result.event.payload == {"message": "watch out", "code": 7}


def test_unknown_protocol_event_is_retained_and_routed_as_raw_warning():
    envelope = json.loads(protocol_line("hello", {"message": "future"}))
    envelope["type"] = "future_event"
    line = json.dumps(envelope)
    parser = OutputParser()

    result = parser.parse_result(line)

    assert result.kind is OutputKind.PROTOCOL_EVENT
    assert result.event is not None
    assert result.event.type == "future_event"
    assert result.raw_line == line

    worker = TrainingWorker(None)
    output_lines = []
    worker.outputSig.connect(output_lines.append)
    worker._route_output(line)
    assert len(output_lines) == 1
    assert "Unhandled protocol event 'future_event'" in output_lines[0]
    assert line in output_lines[0]


@pytest.mark.parametrize(
    ("event_type", "payload", "expected"),
    [
        ("log", {"message": "training started"}, "training started"),
        ("warning", {"message": "low memory"}, "low memory"),
        ("error", {"message": "failed batch"}, "failed batch"),
        ("state", {"state": "running"}, None),
        ("artifact", {"path": "model.pt"}, None),
        ("exit", {"code": 0}, None),
    ],
)
def test_worker_routes_protocol_events_without_a_subprocess(
    event_type, payload, expected
):
    line = protocol_line(event_type, payload)
    worker = TrainingWorker(None)
    output_lines = []
    worker.outputSig.connect(output_lines.append)

    worker._route_output(line)

    assert output_lines == [expected if expected is not None else line]


def test_worker_routes_metric_events_to_progress_signal():
    worker = TrainingWorker(None)
    progress_updates = []
    output_lines = []
    worker.progressSig.connect(progress_updates.append)
    worker.outputSig.connect(output_lines.append)

    worker._route_output(v1_metric_line(metric_payload()))

    assert len(progress_updates) == 1
    assert progress_updates[0].epoch == 3
    assert output_lines == []


def test_protocol_stream_requires_contiguous_identity_and_sequence():
    parser = OutputParser()

    first, accepted = parser.parse_line(
        v1_metric_line({"epoch": 1, "total_epochs": 10}, seq=0)
    )
    assert accepted is True
    assert first.epoch == 1

    for line in (
        v1_metric_line({"epoch": 2, "total_epochs": 10}, seq=0),
        v1_metric_line({"epoch": 3, "total_epochs": 10}, seq=2),
        v1_metric_line(
            {"epoch": 4, "total_epochs": 10},
            seq=1,
            run_id="123e4567-e89b-12d3-a456-426614174001",
        ),
    ):
        progress, accepted = parser.parse_line(line)
        assert accepted is False
        assert progress.epoch == 1

    progress, accepted = parser.parse_line(
        v1_metric_line({"epoch": 5, "total_epochs": 10}, seq=1)
    )
    assert accepted is True
    assert progress.epoch == 5


def test_hello_sequence_zero_then_metric_sequence_one_is_accepted():
    parser = OutputParser()

    _, hello_is_metric = parser.parse_line(protocol_line("hello", {}, seq=0))
    progress, metric_is_metric = parser.parse_line(
        v1_metric_line({"epoch": 1, "total_epochs": 10}, seq=1)
    )

    assert hello_is_metric is False
    assert metric_is_metric is True
    assert progress.epoch == 1


def test_invalid_metric_consumes_sequence_and_allows_next_protocol_event():
    parser = OutputParser()
    worker = TrainingWorker(None)
    worker._parser = parser
    output_lines = []
    worker.outputSig.connect(output_lines.append)

    _, hello_is_metric = parser.parse_line(protocol_line("hello", {}, seq=0))
    progress, invalid_is_metric = parser.parse_line(
        v1_metric_line({"epoch": True}, seq=1)
    )
    worker._route_output(protocol_line("log", {"message": "recovered"}, seq=2))

    assert hello_is_metric is False
    assert invalid_is_metric is False
    assert progress.epoch == 0
    assert parser._last_seq == 2
    assert output_lines == ["recovered"]


def test_wrong_metric_types_and_domains_are_non_fatal_and_atomic():
    parser = OutputParser()
    parser.parse_line(v1_metric_line(metric_payload()))
    before = asdict(parser.progress)

    invalid_payloads = [
        {"epoch": True},
        {"train_loss": "not-a-number"},
        {"mae_per_param": {"value": 0.1}},
        {"time_per_epoch": -1.0},
        {"epoch": 11, "total_epochs": 10},
    ]
    for invalid_payload in invalid_payloads:
        parser = OutputParser()
        parser.parse_line(v1_metric_line(metric_payload()))
        result = parser.parse_result(v1_metric_line(invalid_payload, seq=1))
        progress, is_metrics = result.progress, result.kind is OutputKind.METRIC
        assert is_metrics is False
        assert asdict(progress) == before
        assert result.kind is OutputKind.MALFORMED_PROTOCOL
        assert result.event is not None
        assert result.event.type == "metric"
        assert parser._last_seq == 1

    result = parser.parse_result(
        LEGACY_METRICS_PREFIX + json.dumps({"mae_per_param": {"value": 0.1}})
    )
    assert result.kind is OutputKind.MALFORMED_PROTOCOL
    assert result.event is None
    assert result.progress.epoch == before["epoch"]
    assert asdict(result.progress) == before


@pytest.mark.parametrize(
    "line",
    [
        v1_metric_line({"learning_rate": 10**1000}, seq=1),
        LEGACY_METRICS_PREFIX + json.dumps({"lr": 10**1000}),
        v1_metric_line(
            {"epoch": 0, "total_epochs": 10**1000, "time_per_epoch": 1.0},
            seq=1,
        ),
        LEGACY_METRICS_PREFIX
        + json.dumps({"epoch": 0, "total_epochs": 10**1000, "epoch_time": 1.0}),
    ],
)
def test_huge_metric_numbers_are_visible_logs_without_state_corruption(line):
    parser = OutputParser()
    parser.parse_line(v1_metric_line(metric_payload()))
    before = asdict(parser.progress)

    result = parser.parse_result(line)
    progress, is_metrics = result.progress, result.kind is OutputKind.METRIC

    assert is_metrics is False
    assert result.kind is OutputKind.MALFORMED_PROTOCOL
    if result.event is not None:
        assert result.event.type == "metric"
        assert result.event.seq == 1
        assert parser._last_seq == 1
    assert asdict(progress) == before
    assert math.isfinite(progress.eta_seconds)


def test_none_metric_values_are_tolerated_as_absent():
    parser = OutputParser()
    parser.parse_line(v1_metric_line(metric_payload()))
    before = asdict(parser.progress)

    progress, is_metrics = parser.parse_line(
        v1_metric_line(
            {"r2_score": None, "mae_per_param": [0.1, None], "total_time": None},
            seq=1,
        )
    )

    assert is_metrics is True
    assert asdict(progress) == before


def test_metric_snapshots_are_independent_between_parses():
    parser = OutputParser()

    first, first_is_metrics = parser.parse_line(v1_metric_line(metric_payload()))
    second_payload = metric_payload()
    second_payload["epoch"] = 4
    second_payload["mae_per_param"] = [0.3, 0.4]
    second, second_is_metrics = parser.parse_line(v1_metric_line(second_payload, seq=1))

    assert first_is_metrics is True
    assert second_is_metrics is True
    assert first.epoch == 3
    assert first.mae_per_param == [0.1, 0.14]
    assert second.epoch == 4
    assert second.mae_per_param == [0.3, 0.4]


def test_metric_application_does_not_retain_caller_owned_values():
    payload = metric_payload()
    payload["nested"] = {"values": [1, {"kept": True}]}
    original = copy.deepcopy(payload)
    parser = OutputParser()

    parser._apply_metrics(payload)

    assert payload == original
    payload["mae_per_param"].append(99.0)
    assert parser.progress.mae_per_param == original["mae_per_param"]
