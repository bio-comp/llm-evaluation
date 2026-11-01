import pytest
from pydantic import BaseModel, ConfigDict
from llm_evaluation.runtime.structured import (
    build_structured_output_contract,
    parse_structured_output,
)


class Answer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: int


def test_schema_identity_and_strict_structure_without_semantic_oracle():
    c = build_structured_output_contract(Answer)
    assert c.domain_schema_sha256 == c.effective_schema_sha256
    assert parse_structured_output(c, b'{"value":999}').value == 999
    for raw in (b'{"value":"1"}', b'{"value":1,"value":2}', b'{"value":true}', b"{}"):
        with pytest.raises(ValueError):
            parse_structured_output(c, raw)


def test_unsupported_schema_keyword_rejected():
    class Unsupported(BaseModel):
        values: dict[str, int]

    with pytest.raises(ValueError):
        build_structured_output_contract(Unsupported)


def test_exact_build_unsupported_schema_combinations_fail_closed():
    from pydantic import Field

    class FloatBound(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        value: float = Field(ge=0)

    class Child(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        value: int

    class Referenced(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        child: Child

    for model in (FloatBound, Referenced):
        with pytest.raises(ValueError):
            build_structured_output_contract(model)
