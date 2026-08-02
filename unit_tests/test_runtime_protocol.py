import json
import math
from datetime import datetime, tzinfo
from fractions import Fraction

import pytest

from wavedl.runtime_protocol import (
    EVENT_TYPES,
    LEGACY_METRICS_PREFIX,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ProtocolEncodeError,
    ProtocolParseError,
    RuntimeEvent,
    canonicalize_metric_keys,
    encode_event,
    normalize_legacy_metrics_line,
    parse_jsonl_line,
)


RUN_ID = "123e4567-e89b-12d3-a456-426614174000"


def _json_line_with_payload(payload_json):
    return (
        '{"protocol":"wavedl-jsonl","version":1,"run_id":"'
        f'{RUN_ID}","seq":0,"ts":"2026-01-02T03:04:05Z",'
        f'"type":"hello","payload":{payload_json}}}'
    )


def _json_line_with_version(version_json):
    return (
        '{"protocol":"wavedl-jsonl","version":'
        f'{version_json},"run_id":"{RUN_ID}","seq":0,'
        '"ts":"2026-01-02T03:04:05Z","type":"hello","payload":{}}'
    )


def test_encode_event_is_compact_strict_json_and_converts_nonfinite_values():
    line = encode_event(
        "metric",
        {"nan": math.nan, "positive": math.inf, "negative": -math.inf},
        run_id=RUN_ID,
        seq=0,
        ts="2026-01-02T03:04:05.000Z",
    )

    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert " " not in line
    assert json.loads(line) == {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "seq": 0,
        "ts": "2026-01-02T03:04:05Z",
        "type": "metric",
        "payload": {"nan": None, "positive": None, "negative": None},
    }


def test_encode_and_parse_round_trip():
    payload = {"values": [1, "ok", (True, None)]}
    event = parse_jsonl_line(
        encode_event(
            "state",
            payload,
            run_id=RUN_ID,
            seq=4,
            ts="2026-01-02T03:04:05Z",
        )
    )

    assert event == RuntimeEvent(
        protocol=PROTOCOL_NAME,
        version=PROTOCOL_VERSION,
        run_id=RUN_ID,
        seq=4,
        ts="2026-01-02T03:04:05Z",
        type="state",
        payload={"values": [1, "ok", [True, None]]},
    )


def test_encode_normalizes_supplied_timestamp_to_utc():
    line = encode_event(
        "state",
        {},
        run_id=RUN_ID,
        seq=0,
        ts="2026-01-02T03:04:05+07:00",
    )

    assert json.loads(line)["ts"] == "2026-01-01T20:04:05Z"


def test_encode_rejects_naive_supplied_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        encode_event(
            "state",
            {},
            run_id=RUN_ID,
            seq=0,
            ts="2026-01-02T03:04:05",
        )


def test_parse_canonicalizes_offset_timestamp_to_utc():
    envelope = _envelope()
    envelope["ts"] = "2026-01-02T03:04:05+07:00"

    event = parse_jsonl_line(json.dumps(envelope))

    assert event.ts == "2026-01-01T20:04:05Z"


def test_timestamp_rejects_more_than_six_fractional_digits():
    timestamp = "2026-01-02T03:04:05.1234567Z"

    with pytest.raises(ProtocolParseError, match="invalid field: ts"):
        parse_jsonl_line(json.dumps({**_envelope(), "ts": timestamp}))
    with pytest.raises(ProtocolEncodeError, match="RFC3339"):
        encode_event("state", {}, run_id=RUN_ID, seq=0, ts=timestamp)


def test_timestamp_rejects_tzinfo_without_offset():
    class NoOffset(tzinfo):
        def utcoffset(self, _dt):
            return None

        def dst(self, _dt):
            return None

    with pytest.raises(ProtocolEncodeError, match="timezone"):
        encode_event(
            "state",
            {},
            run_id=RUN_ID,
            seq=0,
            ts=datetime(2026, 1, 2, tzinfo=NoOffset()),
        )


def test_encode_rejects_timestamp_utc_overflow_as_timestamp_error():
    with pytest.raises(ProtocolEncodeError, match="timestamp"):
        encode_event(
            "state",
            {},
            run_id=RUN_ID,
            seq=0,
            ts="9999-12-31T23:59:59-23:59",
        )


def test_parse_rejects_timestamp_utc_overflow_as_timestamp_error():
    envelope = _envelope()
    envelope["ts"] = "9999-12-31T23:59:59-23:59"

    with pytest.raises(ProtocolParseError, match="invalid field: ts"):
        parse_jsonl_line(json.dumps(envelope))


def test_encode_rejects_unknown_negative_zero_offset():
    with pytest.raises(ProtocolEncodeError, match="timestamp"):
        encode_event(
            "state",
            {},
            run_id=RUN_ID,
            seq=0,
            ts="2026-01-02T03:04:05-00:00",
        )


def test_parse_rejects_unknown_negative_zero_offset():
    envelope = _envelope()
    envelope["ts"] = "2026-01-02T03:04:05-00:00"

    with pytest.raises(ProtocolParseError, match="invalid field: ts"):
        parse_jsonl_line(json.dumps(envelope))


@pytest.mark.parametrize("line", ["", "   ", "not json", "null", "[]"])
def test_parse_rejects_invalid_json_blank_and_non_object(line):
    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(line)


@pytest.mark.parametrize("payload", [[], "text", None])
def test_parse_rejects_non_object_payload(payload):
    envelope = _envelope()
    envelope["payload"] = payload

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


def test_parse_rejects_duplicate_top_level_key():
    line = (
        '{"protocol":"wavedl-jsonl","version":1,"run_id":"'
        f'{RUN_ID}","seq":0,"ts":"2026-01-02T03:04:05Z",'
        '"type":"hello","payload":{},"seq":1}'
    )

    with pytest.raises(ProtocolParseError, match="duplicate key"):
        parse_jsonl_line(line)


def test_parse_rejects_decoded_nonfinite_number():
    line = _json_line_with_payload('{"nested":[1e999]}')

    with pytest.raises(ProtocolParseError, match="non-finite"):
        parse_jsonl_line(line)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            _json_line_with_payload('{"x":1e999}'),
            "non-finite number at $.payload.x",
        ),
        (
            _json_line_with_version("1e999"),
            "non-finite number at $.version",
        ),
    ],
)
def test_parse_reports_exact_nonfinite_paths(line, message):
    with pytest.raises(ProtocolParseError) as exc_info:
        parse_jsonl_line(line)
    assert str(exc_info.value) == message


def test_parse_translates_decoder_recursion_failure():
    nested = "[" * 1200 + "0" + "]" * 1200
    line = _json_line_with_payload(nested)

    with pytest.raises(ProtocolParseError, match="Invalid JSONL line"):
        parse_jsonl_line(line)


def test_parse_rejects_wrong_protocol():
    envelope = _envelope()
    envelope["protocol"] = "other"

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


def test_parse_ignores_unknown_top_level_fields():
    envelope = _envelope()
    envelope["future_field"] = {"ignored": True}

    event = parse_jsonl_line(json.dumps(envelope))

    assert event.type == "hello"
    assert not hasattr(event, "future_field")


@pytest.mark.parametrize(
    "missing", ["protocol", "version", "run_id", "seq", "ts", "type", "payload"]
)
def test_parse_rejects_missing_required_key(missing):
    envelope = _envelope()
    del envelope[missing]

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


def test_parse_rejects_wrong_version_and_type():
    for key, value in (("version", 2), ("type", "unknown")):
        envelope = _envelope()
        envelope[key] = value
        with pytest.raises(ProtocolParseError):
            parse_jsonl_line(json.dumps(envelope))


@pytest.mark.parametrize("version", [True, False, 1.0, "1"])
def test_parse_rejects_noninteger_protocol_version(version):
    envelope = _envelope()
    envelope["version"] = version

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


@pytest.mark.parametrize("seq", [-1, 1.5, True])
def test_parse_rejects_negative_or_noninteger_sequence(seq):
    envelope = _envelope()
    envelope["seq"] = seq

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("run_id", "not-a-uuid"),
        ("ts", 123),
        ("ts", "2026-01-02T03:04:05"),
        ("ts", "not-a-timestamp"),
    ],
)
def test_parse_rejects_invalid_uuid_or_timestamp(key, value):
    envelope = _envelope()
    envelope[key] = value

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


def test_encode_rejects_unsupported_event_type():
    assert {
        "hello",
        "state",
        "metric",
        "log",
        "artifact",
        "warning",
        "error",
        "exit",
    } == EVENT_TYPES

    with pytest.raises(ValueError):
        encode_event("unsupported", {}, run_id=RUN_ID, seq=0)


def test_encode_rejects_non_string_mapping_keys():
    with pytest.raises(ProtocolEncodeError, match="mapping keys must be strings"):
        encode_event("state", {"nested": {1: "bad"}}, run_id=RUN_ID, seq=0)


@pytest.mark.parametrize("container", ["dict", "list", "tuple"])
def test_encode_rejects_cyclic_payload(container):
    if container == "dict":
        payload = {}
        payload["self"] = payload
    elif container == "list":
        cyclic = []
        cyclic.append(cyclic)
        payload = {"value": cyclic}
    else:
        values = []
        cyclic = (values,)
        values.append(cyclic)
        payload = {"value": cyclic}

    with pytest.raises(ProtocolEncodeError, match="cyclic"):
        encode_event("state", payload, run_id=RUN_ID, seq=0)


def test_encode_converts_stdlib_real_scalar():
    line = encode_event(
        "metric",
        {"fraction": Fraction(1, 2)},
        run_id=RUN_ID,
        seq=0,
    )

    assert json.loads(line)["payload"]["fraction"] == 0.5


def test_encode_preserves_bool_scalar():
    line = encode_event("state", {"enabled": True}, run_id=RUN_ID, seq=0)

    assert json.loads(line)["payload"]["enabled"] is True


def test_parse_canonicalizes_uuid_identity():
    envelope = _envelope()
    envelope["run_id"] = "123E4567E89B12D3A456426614174000"

    event = parse_jsonl_line(json.dumps(envelope))

    assert event.run_id == RUN_ID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("run_id", "invalid"),
        ("seq", -1),
        ("ts", "invalid"),
        ("payload", []),
    ],
)
def test_parse_reports_stable_invalid_field_diagnostics(field, value):
    envelope = _envelope()
    envelope[field] = value

    with pytest.raises(ProtocolParseError, match=f"invalid field: {field}"):
        parse_jsonl_line(json.dumps(envelope))


def test_nested_nonfinite_values_become_null():
    line = encode_event(
        "log",
        {"outer": [{"nan": math.nan}, (math.inf, -math.inf)]},
        run_id=RUN_ID,
        seq=1,
    )

    assert json.loads(line)["payload"] == {
        "outer": [{"nan": None}, [None, None]],
    }


def test_normalize_legacy_metrics_returns_none_for_non_prefixed_line():
    assert normalize_legacy_metrics_line("ordinary training log") is None


def test_legacy_metrics_prefix_is_stable():
    assert LEGACY_METRICS_PREFIX == "##METRICS##"


def test_normalize_legacy_metrics_remaps_legacy_keys_and_preserves_fields():
    line = LEGACY_METRICS_PREFIX + json.dumps(
        {
            "r2": 0.9,
            "lr": 0.001,
            "epoch_time": 2.5,
            "epoch": 3,
            "nested": {"values": [1, "kept", True]},
        }
    )

    metrics = normalize_legacy_metrics_line(line)

    assert metrics == {
        "r2_score": 0.9,
        "learning_rate": 0.001,
        "time_per_epoch": 2.5,
        "epoch": 3,
        "nested": {"values": [1, "kept", True]},
    }


def test_normalize_legacy_metrics_allows_surrounding_whitespace():
    line = f'  {LEGACY_METRICS_PREFIX} {{"r2": 0.5}}  \n'

    assert normalize_legacy_metrics_line(line) == {"r2_score": 0.5}


def test_normalize_legacy_metrics_sanitizes_producer_nonfinite_values():
    line = (
        LEGACY_METRICS_PREFIX
        + '{"r2":NaN,"history":[1,Infinity,{"loss":-Infinity}],"overflow":1e999}'
    )

    assert normalize_legacy_metrics_line(line) == {
        "r2_score": None,
        "history": [1, None, {"loss": None}],
        "overflow": None,
    }


@pytest.mark.parametrize(
    "payload",
    [
        '{"r2":',
        "[]",
        '"metrics"',
        "null",
    ],
)
def test_normalize_legacy_metrics_rejects_invalid_or_non_object_payload(payload):
    with pytest.raises(ProtocolParseError):
        normalize_legacy_metrics_line(LEGACY_METRICS_PREFIX + payload)


def test_normalize_legacy_metrics_rejects_duplicate_keys():
    line = LEGACY_METRICS_PREFIX + '{"r2":0.1,"r2":0.2}'

    with pytest.raises(ProtocolParseError, match="duplicate key"):
        normalize_legacy_metrics_line(line)


@pytest.mark.parametrize(
    ("legacy_key", "canonical_key"),
    [
        ("r2", "r2_score"),
        ("lr", "learning_rate"),
        ("epoch_time", "time_per_epoch"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_normalize_legacy_metrics_rejects_alias_conflicts_in_both_orders(
    legacy_key, canonical_key, reverse
):
    pairs = [(legacy_key, 1), (canonical_key, 2)]
    if reverse:
        pairs.reverse()
    payload = ",".join(f'"{key}":{value}' for key, value in pairs)

    with pytest.raises(ProtocolParseError, match="conflicting metric keys"):
        normalize_legacy_metrics_line(LEGACY_METRICS_PREFIX + "{" + payload + "}")


@pytest.mark.parametrize(
    ("legacy_key", "canonical_key"),
    [
        ("r2", "r2_score"),
        ("lr", "learning_rate"),
        ("epoch_time", "time_per_epoch"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_canonicalize_metric_keys_rejects_alias_conflicts_in_both_orders(
    legacy_key, canonical_key, reverse
):
    """Canonical metric output must not discard conflicting values."""
    pairs = [(legacy_key, 1), (canonical_key, 2)]
    if reverse:
        pairs.reverse()

    with pytest.raises(ProtocolParseError, match="conflicting metric keys"):
        canonicalize_metric_keys(dict(pairs))


def test_normalize_legacy_metrics_preserves_unrelated_nested_and_list_values():
    line = LEGACY_METRICS_PREFIX + json.dumps(
        {"r2": 0.9, "nested": {"values": [1, {"kept": True}]}}
    )

    metrics = normalize_legacy_metrics_line(line)

    assert metrics == {
        "r2_score": 0.9,
        "nested": {"values": [1, {"kept": True}]},
    }


def _envelope():
    return {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "seq": 0,
        "ts": "2026-01-02T03:04:05Z",
        "type": "hello",
        "payload": {},
    }
