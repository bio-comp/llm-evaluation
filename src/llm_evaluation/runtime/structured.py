"""Conservative exact schema adapter: structure is not semantic correctness."""

from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel
from .codec import canonical_json_bytes, parse_json_unique, sha256_hex

_ALLOWED = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "title",
    "description",
    "enum",
    "const",
    "items",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
}


@dataclass(frozen=True)
class StructuredOutputContract:
    output_model: type[BaseModel]
    schema_bytes: bytes
    domain_schema_sha256: str
    effective_schema_sha256: str
    adapter_id: str = "llamacpp-json-schema-v1"

    def effective_schema_json(self) -> Any:
        return parse_json_unique(self.schema_bytes)


def build_structured_output_contract(
    model: type[BaseModel],
) -> StructuredOutputContract:
    schema = model.model_json_schema()

    def check(s: Any) -> None:
        if not isinstance(s, dict):
            raise ValueError("schema object required")
        if set(s) - _ALLOWED:
            raise ValueError("unsupported schema keyword")
        if (
            any(key in s for key in ("minimum", "maximum"))
            and s.get("type") != "integer"
        ):
            raise ValueError("exact build supports bounds only for integer schemas")
        if s.get("type") == "object" and s.get("additionalProperties") is not False:
            raise ValueError("closed objects required")
        for key in ("properties", "$defs"):
            for child in s.get(key, {}).values():
                check(child)
        if "items" in s:
            check(s["items"])
        for key in ("anyOf", "oneOf"):
            for child in s.get(key, []):
                check(child)

    check(schema)
    wire = canonical_json_bytes(schema)
    digest = sha256_hex(wire)
    return StructuredOutputContract(model, wire, digest, digest)


def parse_structured_output(
    contract: StructuredOutputContract, data: bytes
) -> BaseModel:
    return contract.output_model.model_validate(parse_json_unique(data), strict=True)
