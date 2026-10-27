from __future__ import annotations

import hashlib
import json
from typing import TypeVar

from pydantic import BaseModel

_RecordT = TypeVar("_RecordT", bound=BaseModel)


def canonical_record_bytes(record: BaseModel) -> bytes:
    return json.dumps(
        record.model_dump(mode="json"),
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
