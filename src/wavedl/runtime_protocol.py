"""JSON Lines protocol used at the runtime boundary."""

import json
import math
import numbers
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


PROTOCOL_NAME = "wavedl-jsonl"
PROTOCOL_VERSION = 1
LEGACY_METRICS_PREFIX = "##METRICS##"
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
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


class ProtocolParseError(ValueError):
    """Raised when a JSONL line is not a valid protocol v1 envelope."""


class ProtocolEncodeError(ValueError):
    """Raised when a value cannot be encoded as a protocol v1 event."""


def normalize_legacy_metrics_line(line: str) -> dict[str, Any] | None:
    """Decode and canonicalize a legacy metrics log line."""
    if not isinstance(line, str):
        raise ProtocolParseError("Metrics line must be a string")
    stripped_line = line.strip()
    if not stripped_line.startswith(LEGACY_METRICS_PREFIX):
        return None
    metrics = _load_json_object(
        stripped_line[len(LEGACY_METRICS_PREFIX) :],
        nonfinite_policy="sanitize",
        object_name="Legacy metrics",
        invalid_message="Invalid legacy metrics line",
    )

    legacy_names = {
        "r2": "r2_score",
        "lr": "learning_rate",
        "epoch_time": "time_per_epoch",
    }
    for legacy_name, canonical_name in legacy_names.items():
        if legacy_name in metrics and canonical_name in metrics:
            raise ProtocolParseError(
                f"conflicting metric keys: {legacy_name}, {canonical_name}"
            )
    return {legacy_names.get(key) or key: value for key, value in metrics.items()}


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
    try:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {event_type!r}")
        normalized_run_id = _parse_uuid(run_id)
        _validate_seq(seq)
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        envelope = {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "run_id": str(normalized_run_id),
            "seq": seq,
            "ts": _canonical_timestamp(ts),
            "type": event_type,
            "payload": _sanitize(payload),
        }
        return json.dumps(envelope, separators=(",", ":"), allow_nan=False) + "\n"
    except ProtocolEncodeError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolEncodeError(str(exc)) from exc


def parse_jsonl_line(line: str) -> RuntimeEvent:
    """Parse and validate a single protocol v1 JSONL envelope."""
    if not isinstance(line, str) or not line.strip():
        raise ProtocolParseError("JSONL line must not be blank")
    envelope = _load_json_object(
        line,
        nonfinite_policy="reject",
        object_name="Envelope",
        invalid_message="Invalid JSONL line",
    )
    missing = _REQUIRED_KEYS - envelope.keys()
    if missing:
        missing_field = sorted(missing)[0]
        raise ProtocolParseError(f"invalid field: {missing_field}")
    if envelope["protocol"] != PROTOCOL_NAME:
        raise ProtocolParseError("Wrong protocol")
    if (
        isinstance(envelope["version"], bool)
        or not isinstance(envelope["version"], int)
        or envelope["version"] != PROTOCOL_VERSION
    ):
        raise ProtocolParseError("invalid field: version")
    if not isinstance(envelope["type"], str) or envelope["type"] not in EVENT_TYPES:
        raise ProtocolParseError("Unsupported event type")
    try:
        normalized_run_id = str(_parse_uuid(envelope["run_id"]))
    except (TypeError, ValueError) as exc:
        raise ProtocolParseError("invalid field: run_id") from exc
    try:
        _validate_seq(envelope["seq"])
    except (TypeError, ValueError) as exc:
        raise ProtocolParseError("invalid field: seq") from exc
    try:
        canonical_timestamp = _canonical_timestamp(envelope["ts"])
    except (TypeError, ValueError) as exc:
        raise ProtocolParseError("invalid field: ts") from exc
    if not isinstance(envelope["payload"], dict):
        raise ProtocolParseError("invalid field: payload")
    return RuntimeEvent(
        protocol=envelope["protocol"],
        version=envelope["version"],
        run_id=normalized_run_id,
        seq=envelope["seq"],
        ts=canonical_timestamp,
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


def _canonical_timestamp(value: str | datetime | None) -> str:
    if value is None:
        value = datetime.now(UTC)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        parsed = value
    else:
        if not isinstance(value, str):
            raise ValueError("timestamp must be RFC3339")
        if not _RFC3339.fullmatch(value):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("timestamp must be RFC3339") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timestamp must be timezone-aware")
            raise ValueError("timestamp must be RFC3339")
        if value.endswith("-00:00"):
            raise ValueError("timestamp has unknown offset")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    try:
        return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    except OverflowError as exc:
        raise ValueError("timestamp UTC conversion overflow") from exc


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_json_object(
    text: str,
    *,
    nonfinite_policy: str,
    object_name: str,
    invalid_message: str,
) -> dict[str, Any]:
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_parse_json_constant,
        )
        if nonfinite_policy == "reject":
            _reject_nonfinite(decoded)
        else:
            decoded = _sanitize_nonfinite(decoded)
    except RecursionError as exc:
        raise ProtocolParseError(invalid_message) from exc
    except (ValueError, TypeError) as exc:
        if isinstance(exc, (_DuplicateKeyError, _NonFiniteError)):
            raise ProtocolParseError(str(exc)) from exc
        raise ProtocolParseError(invalid_message) from exc
    if not isinstance(decoded, dict):
        raise ProtocolParseError(f"{object_name} must be a JSON object")
    return decoded


def _reject_nonfinite(value, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        raise _NonFiniteError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _sanitize_nonfinite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_nonfinite(item) for item in value]
    return value


def _sanitize(value: Any, active: set[int] | None = None) -> Any:
    if active is None:
        active = set()
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise ProtocolEncodeError("cyclic payload")
        active.add(identity)
        try:
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ProtocolEncodeError("mapping keys must be strings")
                result[key] = _sanitize(item, active)
            return result
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ProtocolEncodeError("cyclic payload")
        active.add(identity)
        try:
            return [_sanitize(item, active) for item in value]
        finally:
            active.remove(identity)
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return value


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteError(ValueError):
    pass


def _parse_json_constant(value: str) -> float:
    return float(value)
