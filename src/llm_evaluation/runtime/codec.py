"""Canonical finite JSON and ambiguity-rejecting provider decoding."""

import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType


def freeze(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON keys must be strings")
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("finite JSON-like value required")


def json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def canonical_json_bytes(value: object) -> bytes:
    freeze(value)
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json_unique(data: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out

    def invalid(value: str) -> object:
        raise ValueError("non-finite JSON number")

    try:
        result = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid
        )
        canonical_json_bytes(result)  # rejects overflowed floats and invalid Unicode
        return result
    except (UnicodeError, RecursionError) as e:
        raise ValueError("invalid JSON encoding/depth") from e
