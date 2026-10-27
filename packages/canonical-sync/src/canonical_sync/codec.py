from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

_RecordT = TypeVar("_RecordT", bound=BaseModel)


def _canonical_scalar(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Enum):
        return _canonical_scalar(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetime must be timezone-aware")
        normalized = value.astimezone(UTC)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, (tuple, list)):
        return [_canonical_scalar(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical record mappings require string keys")
        return {key: _canonical_scalar(item) for key, item in value.items()}
    if isinstance(value, float):
        raise TypeError("canonical record identity does not support floating-point scalars")
    raise TypeError(f"unsupported canonical record scalar: {type(value).__name__}")


def canonical_record_bytes(record: BaseModel) -> bytes:
    """Return the versioned record's canonical UTF-8 JSON bytes with no trailing newline."""

    normalized = _canonical_scalar(record.model_dump(mode="python"))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_record_hash(record: BaseModel) -> str:
    return hashlib.sha256(canonical_record_bytes(record)).hexdigest()


def parse_canonical_record(data: bytes, record_type: type[_RecordT]) -> _RecordT:
    if type(data) is not bytes:
        raise TypeError("canonical record input must be bytes")
    record = record_type.model_validate_json(data, strict=True)
    if canonical_record_bytes(record) != data:
        raise ValueError("record is valid but not canonical JSON")
    return record
