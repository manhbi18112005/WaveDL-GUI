import json
import math

import pytest

from wavedl.runtime_protocol import (
    EVENT_TYPES,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ProtocolParseError,
    RuntimeEvent,
    encode_event,
    parse_jsonl_line,
)


RUN_ID = "123e4567-e89b-12d3-a456-426614174000"


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
        "ts": "2026-01-02T03:04:05.000Z",
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


@pytest.mark.parametrize("line", ["", "   ", "not json", "null", "[]"])
def test_parse_rejects_invalid_json_blank_and_non_object(line):
    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(line)


def test_parse_rejects_wrong_protocol():
    envelope = _envelope()
    envelope["protocol"] = "other"

    with pytest.raises(ProtocolParseError):
        parse_jsonl_line(json.dumps(envelope))


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
