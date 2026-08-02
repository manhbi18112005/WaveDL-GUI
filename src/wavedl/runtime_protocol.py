"""JSON Lines protocol used at the runtime boundary."""

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


PROTOCOL_NAME = "wavedl-jsonl"
PROTOCOL_VERSION = 1
EVENT_TYPES = {
    "hello",
    "state",
    "metric",
    "log",
    "artifact",
    "warning",
    "error",
    "exit",
}
_REQUIRED_KEYS = {"protocol", "version", "run_id", "seq", "ts", "type", "payload"}
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class ProtocolParseError(ValueError):
    """Raised when a JSONL line is not a valid protocol v1 envelope."""


@dataclass(frozen=True)
class RuntimeEvent:
    """A decoded runtime protocol event."""

    protocol: str
    version: int
    run_id: str
    seq: int
    ts: str
    type: str
    payload: dict[str, Any]


def encode_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    run_id: str | UUID,
    seq: int,
    ts: str | datetime | None = None,
) -> str:
    """Encode one protocol event as compact, strict JSONL."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type!r}")
    normalized_run_id = _parse_uuid(run_id)
    _validate_seq(seq)
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    timestamp = _encode_timestamp(ts)
    envelope = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "run_id": str(normalized_run_id),
        "seq": seq,
        "ts": timestamp,
        "type": event_type,
        "payload": _sanitize(payload),
    }
    return json.dumps(envelope, separators=(",", ":"), allow_nan=False) + "\n"


def parse_jsonl_line(line: str) -> RuntimeEvent:
    """Parse and validate a single protocol v1 JSONL envelope."""
    if not isinstance(line, str) or not line.strip():
        raise ProtocolParseError("JSONL line must not be blank")
    try:
        envelope = json.loads(line, parse_constant=_reject_json_constant)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ProtocolParseError("Invalid JSONL line") from exc
    if not isinstance(envelope, dict):
        raise ProtocolParseError("Envelope must be a JSON object")
    missing = _REQUIRED_KEYS - envelope.keys()
    if missing:
        raise ProtocolParseError(f"Missing required envelope keys: {sorted(missing)}")
    if envelope["protocol"] != PROTOCOL_NAME:
        raise ProtocolParseError("Wrong protocol")
    if envelope["version"] != PROTOCOL_VERSION:
        raise ProtocolParseError("Wrong protocol version")
    if not isinstance(envelope["type"], str) or envelope["type"] not in EVENT_TYPES:
        raise ProtocolParseError("Unsupported event type")
    try:
        _parse_uuid(envelope["run_id"])
        _validate_seq(envelope["seq"])
        _parse_timestamp(envelope["ts"])
    except (TypeError, ValueError) as exc:
        raise ProtocolParseError("Invalid envelope field") from exc
    if not isinstance(envelope["payload"], dict):
        raise ProtocolParseError("Payload must be a JSON object")
    return RuntimeEvent(
        protocol=envelope["protocol"],
        version=envelope["version"],
        run_id=envelope["run_id"],
        seq=envelope["seq"],
        ts=envelope["ts"],
        type=envelope["type"],
        payload=envelope["payload"],
    )


def _parse_uuid(value: str | UUID) -> UUID:
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str):
        raise TypeError("run_id must be a UUID")
    return UUID(value)


def _validate_seq(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seq must be a nonnegative integer")


def _encode_timestamp(value: str | datetime | None) -> str:
    if value is None:
        return (
            datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return (
            value.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    _parse_timestamp(value)
    return value


def _parse_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339.fullmatch(value):
        raise ValueError("timestamp must be RFC3339")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")
