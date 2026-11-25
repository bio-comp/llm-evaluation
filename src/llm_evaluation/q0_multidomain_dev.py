"""Standalone, synthetic-only E-002 Q0 DEVELOPMENT packet and reference checks.

Distinct from the frozen five-case Qwen3/llama-server technical preflight.
No model code, tool calls, inference runtime or downloaded artifacts are executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScoringMethod = Literal[
    "INTEGER",
    "JSON_STRUCTURAL",
    "RATIONAL",
    "EXACT_TEXT",
    "SANDBOX_DEFERRED",
    "EXPERT_DEFERRED",
]
Category = Literal[
    "sanity", "coding", "scientific_reasoning", "general_instruction_following"
]
Disposition = Literal["PASS", "FAIL", "NOT_AUTOMATICALLY_SCORABLE"]
MAX_ANSWER_CHARS = 8192
PACKET_ID = "e002-q0-multidomain-dev-v1"
PACKET_SHA256 = "b2749f6dc23ac42d4632cf1fe1d1c44150a3215486109e5ca82929406c7518d9"

CONTROL_SOURCE_SHA256 = (
    "2371f2e019f3b9862c7ec28a75c5486888efdf9b31c297348f5dd883db3b306a"
)
# Frozen v1 synthetic controls after R1 qualification correction. IDs and source
# bytes are both independently pinned; retaining only superficial coverage
# cannot promote a weakened/replaced instrument to "qualified".
CONTROL_INVENTORY: tuple[str, ...] = (
    "sanity-arithmetic-01",
    "sanity-arithmetic-02",
    "sanity-arithmetic-03",
    "sanity-record-map-01",
    "sanity-record-map-02",
    "sanity-record-map-03",
    "sanity-record-map-04",
    "sanity-record-map-05",
    "sanity-tag-extraction-01",
    "sanity-tag-extraction-02",
    "sanity-tag-extraction-03",
    "sanity-tag-extraction-04",
    "coding-median-01",
    "coding-median-02",
    "coding-clamp-debug-01",
    "coding-clamp-debug-02",
    "coding-tool-call-01",
    "coding-tool-call-02",
    "coding-tool-call-03",
    "coding-tool-call-04",
    "coding-tool-call-05",
    "science-pooled-rate-01",
    "science-pooled-rate-02",
    "science-pooled-rate-03",
    "science-pooled-rate-04",
    "science-pooled-rate-05",
    "science-pooled-rate-06",
    "science-pooled-rate-07",
    "science-pooled-rate-08",
    "science-ci-decision-01",
    "science-ci-decision-02",
    "science-ci-decision-03",
    "science-causal-claim-01",
    "science-causal-claim-02",
    "general-exact-steps-01",
    "general-exact-steps-02",
    "general-exact-steps-03",
    "general-faithful-summary-01",
    "general-faithful-summary-02",
    "general-multistep-followup-01",
    "general-multistep-followup-02",
)

# These are independently recorded reference-content fingerprints, not
# dynamically accepted hashes from the same untrusted packet being validated.
# Changing a source, criterion, or expected value requires a new version/anchors.
REFERENCE_SHA256: dict[str, str] = {
    "sanity-arithmetic": "fc6f5c6851efbf90c6fadab3998753dc1b09e7dcf931085817da1ae8387297f2",
    "sanity-record-map": "829f5550dd8836729c35bb9d8170262d989b2f3683ea7d9ba71ba00b12008c5e",
    "sanity-tag-extraction": "fb8f2b8d85b1e489577bb3bc625e78b4c114559adc9e9631bdfca103c8573413",
    "coding-median": "7cf5c555d04add1cf2e0414036598931aad336d760fe843325afdd30ea643c48",
    "coding-clamp-debug": "51b5f2f9f9fa7034d74727d062c6acbc5edad79db6f56ee8f57f9fd923ac0026",
    "coding-tool-call": "42daca88914206db1037183fbe004d13a9011e13848470919b1c93b7e1d64d0e",
    "science-pooled-rate": "d0b8d5639217ed29104c450284c9e6c6829ce60409c5b835674676568265aca2",
    "science-ci-decision": "f162c0bdd37869e4c544cad6238851db78766307153c55d2941027b93546d4be",
    "science-causal-claim": "41f112d1212a8a54f00be02da6835735b5ba8303a57b1cf50bd2b6d828bde83f",
    "general-exact-steps": "fdb1826c23ea092a636f559cc1f10cb084b0b7530b7a5dd4a53eb1a4718ccabd",
    "general-faithful-summary": "1ee41874dc1ab289c880d67dd34222036b9ae8d95a3bc87176842435b538b631",
    "general-multistep-followup": "26f00179a14f82e3eb5723555fc8dcf7298e9ecd944e427c268cd2ada1f21734",
}
# A fixed 12-case design, not a generic benchmark DSL or model-family runner.
CASE_CONTRACT: tuple[tuple[str, Category, ScoringMethod], ...] = (
    ("sanity-arithmetic", "sanity", "INTEGER"),
    ("sanity-record-map", "sanity", "JSON_STRUCTURAL"),
    ("sanity-tag-extraction", "sanity", "JSON_STRUCTURAL"),
    ("coding-median", "coding", "SANDBOX_DEFERRED"),
    ("coding-clamp-debug", "coding", "SANDBOX_DEFERRED"),
    ("coding-tool-call", "coding", "JSON_STRUCTURAL"),
    ("science-pooled-rate", "scientific_reasoning", "RATIONAL"),
    ("science-ci-decision", "scientific_reasoning", "EXACT_TEXT"),
    ("science-causal-claim", "scientific_reasoning", "EXPERT_DEFERRED"),
    ("general-exact-steps", "general_instruction_following", "EXACT_TEXT"),
    ("general-faithful-summary", "general_instruction_following", "EXPERT_DEFERRED"),
    ("general-multistep-followup", "general_instruction_following", "EXPERT_DEFERRED"),
)


def sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def canonical_reference_sha256(reference: Q0Reference) -> str:
    return sha256_hex(
        json.dumps(
            reference.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Q0Reference(ClosedModel):
    reference_id: str = Field(min_length=5, max_length=100)
    source: str = Field(min_length=5, max_length=1000)
    rationale: str = Field(min_length=5, max_length=1000)
    expected: str | None
    unit_test_ref: str | None


class Q0Case(ClosedModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9-]+$", max_length=80)
    category: Category
    scoring_method: ScoringMethod
    prompt: str = Field(min_length=10, max_length=3000)
    reference: Q0Reference

    @model_validator(mode="after")
    def valid_reference(self) -> Q0Case:
        ref = self.reference
        if ref.reference_id != f"{self.case_id}-ref-v1":
            raise ValueError("reference identity differs from case")
        deferred = self.scoring_method in ("SANDBOX_DEFERRED", "EXPERT_DEFERRED")
        if (ref.expected is None) != deferred:
            raise ValueError("reference expected value conflicts with scoring method")
        if (ref.unit_test_ref is not None) != (
            self.scoring_method == "SANDBOX_DEFERRED"
        ):
            raise ValueError("unit-test reference exists only for sandbox tasks")
        anchor = REFERENCE_SHA256.get(self.case_id)
        if anchor is None or canonical_reference_sha256(ref) != anchor:
            raise ValueError("reference content contradicts independent anchor")
        if self.scoring_method == "JSON_STRUCTURAL":
            assert ref.expected is not None
            _json_value(ref.expected)
        return self


class Q0Packet(ClosedModel):
    schema_version: Literal[1]
    packet_id: Literal["e002-q0-multidomain-dev-v1"]
    phase: Literal["DEVELOPMENT"]
    purpose: str = Field(min_length=30, max_length=500)
    cases: tuple[Q0Case, ...] = Field(min_length=12, max_length=12)

    @model_validator(mode="after")
    def fixed_inventory(self) -> Q0Packet:
        actual = tuple((c.case_id, c.category, c.scoring_method) for c in self.cases)
        if actual != CASE_CONTRACT:
            raise ValueError("case inventory/order/category/scoring contract mismatch")
        return self


class LoadedPacket(ClosedModel):
    packet: Q0Packet
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_size_bytes: int = Field(ge=1)


class Q0Control(ClosedModel):
    control_id: str = Field(min_length=4, max_length=100)
    case_id: str
    response: str = Field(max_length=MAX_ANSWER_CHARS)
    expected_disposition: Disposition
    control_purpose: str = Field(min_length=8)


class Q0ControlSet(ClosedModel):
    schema_version: Literal[1]
    packet_id: Literal["e002-q0-multidomain-dev-v1"]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    controls: tuple[Q0Control, ...] = Field(min_length=41, max_length=41)

    @model_validator(mode="after")
    def unique_controls(self) -> Q0ControlSet:
        ids = [c.control_id for c in self.controls]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate control identity")
        if tuple(ids) != CONTROL_INVENTORY:
            raise ValueError("control inventory differs from qualified v1 instrument")
        return self


class Q0Outcome(ClosedModel):
    case_id: str
    category: Category
    scoring_method: ScoringMethod
    reference_id: str
    reference_sha256: str
    packet_id: str
    packet_version: int
    packet_sha256: str
    packet_source_size_bytes: int
    answer_sha256: str
    disposition: Disposition
    reason_code: str


class Q0Readiness(ClosedModel):
    packet_id: str
    packet_version: int
    packet_sha256: str
    packet_source_size_bytes: int
    phase: Literal["DEVELOPMENT"]
    objective_case_ids: tuple[str, ...]
    sandbox_case_ids: tuple[str, ...]
    expert_case_ids: tuple[str, ...]
    q1_dependencies: tuple[str, ...]


class Q0QualificationReport(ClosedModel):
    readiness: Q0Readiness
    controls_sha256: str
    controls_count: int
    controls_qualified: bool


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _json_value(raw: str) -> object:
    return json.loads(
        raw, object_pairs_hook=_no_duplicate_object, parse_constant=_reject_constant
    )


def load_packet(path: Path, *, source_bytes: bytes | None = None) -> LoadedPacket:
    blob = path.read_bytes() if source_bytes is None else source_bytes
    digest = sha256_hex(blob)
    if digest != PACKET_SHA256:
        raise ValueError(
            "packet source bytes differ from versioned development reference"
        )
    packet = Q0Packet.model_validate_json(blob)
    return LoadedPacket(
        packet=packet, packet_sha256=digest, packet_size_bytes=len(blob)
    )


def _validate_control_source(source: bytes) -> None:
    """Require reviewed v1 control source bytes, not merely plausible labels."""
    if sha256_hex(source) != CONTROL_SOURCE_SHA256:
        raise ValueError("control source differs from qualified v1 instrument")


def load_controls(
    path: Path, packet: LoadedPacket, *, source_bytes: bytes | None = None
) -> Q0ControlSet:
    blob = path.read_bytes() if source_bytes is None else source_bytes
    _validate_control_source(blob)
    control_set = Q0ControlSet.model_validate_json(blob)
    if (
        control_set.packet_sha256 != packet.packet_sha256
        or control_set.packet_id != packet.packet.packet_id
    ):
        raise ValueError("control packet identity mismatch")
    ids = {c.case_id for c in packet.packet.cases}
    if any(c.case_id not in ids for c in control_set.controls):
        raise ValueError("control references an unknown case")
    return control_set


def _parse_rational(raw: str) -> Fraction:
    if (
        len(raw) > MAX_ANSWER_CHARS
        or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*|\.[0-9]+)?", raw) is None
    ):
        raise ValueError("invalid rational syntax")
    if "/" in raw:
        numerator, denominator = raw.split("/")
        value = Fraction(int(numerator), int(denominator))
        if value.numerator != int(numerator) or value.denominator != int(denominator):
            raise ValueError("fraction is not simplified")
        return value
    return Fraction(Decimal(raw))


def _objective_match(method: ScoringMethod, answer: str, expected: str) -> bool:
    if method == "EXACT_TEXT":
        return answer == expected
    if method == "INTEGER":
        return re.fullmatch(r"-?(?:0|[1-9][0-9]*)", answer) is not None and (
            int(answer) == int(expected)
        )
    if method == "JSON_STRUCTURAL":
        left = json.dumps(
            _json_value(answer),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        right = json.dumps(
            _json_value(expected),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return left == right
    if method == "RATIONAL":
        return _parse_rational(answer) == _parse_rational(expected)
    raise ValueError("non-objective scoring method")


def evaluate_response(packet: LoadedPacket, case_id: str, answer: str) -> Q0Outcome:
    case = next((c for c in packet.packet.cases if c.case_id == case_id), None)
    if case is None:
        raise ValueError("unknown case identity")
    if not isinstance(answer, str):
        raise TypeError("answer must be text")
    try:
        answer_bytes = answer.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        # Reject before any scoring or digest: an unpaired surrogate has no
        # genuine UTF-8 source-byte identity. Never normalize it or invent bytes.
        raise ValueError("answer is not valid UTF-8 text") from None
    if case.scoring_method == "SANDBOX_DEFERRED":
        disposition: Disposition = "NOT_AUTOMATICALLY_SCORABLE"
        reason = "requires_qualified_code_sandbox"
    elif case.scoring_method == "EXPERT_DEFERRED":
        disposition = "NOT_AUTOMATICALLY_SCORABLE"
        reason = "requires_expert_reference_assessment"
    elif len(answer) > MAX_ANSWER_CHARS:
        disposition = "FAIL"
        reason = "answer_over_limit"
    else:
        assert case.reference.expected is not None
        try:
            passed = _objective_match(
                case.scoring_method, answer, case.reference.expected
            )
        except (ValueError, InvalidOperation, OverflowError, json.JSONDecodeError):
            passed = False
        disposition = "PASS" if passed else "FAIL"
        reason = "objective_match" if passed else "objective_mismatch"
    return Q0Outcome(
        case_id=case.case_id,
        category=case.category,
        scoring_method=case.scoring_method,
        reference_id=case.reference.reference_id,
        reference_sha256=canonical_reference_sha256(case.reference),
        packet_id=packet.packet.packet_id,
        packet_version=packet.packet.schema_version,
        packet_sha256=packet.packet_sha256,
        packet_source_size_bytes=packet.packet_size_bytes,
        answer_sha256=sha256_hex(answer_bytes),
        disposition=disposition,
        reason_code=reason,
    )


def evaluate_all_controls(
    packet: LoadedPacket, controls: Q0ControlSet
) -> tuple[Q0Outcome, ...]:
    if controls.packet_sha256 != packet.packet_sha256:
        raise ValueError("control packet identity mismatch")
    results: list[Q0Outcome] = []
    for control in controls.controls:
        result = evaluate_response(packet, control.case_id, control.response)
        if result.disposition != control.expected_disposition:
            raise ValueError("control expectation contradicts scorer outcome")
        results.append(result)
    # This is an evaluator qualification, not evidence that expert/deferred
    # answers are valid. Negative and valid-alternative coverage is explicit.
    for c in packet.packet.cases:
        labels = {
            t.expected_disposition for t in controls.controls if t.case_id == c.case_id
        }
        required: set[str] = (
            {"NOT_AUTOMATICALLY_SCORABLE"}
            if c.scoring_method in ("SANDBOX_DEFERRED", "EXPERT_DEFERRED")
            else {"PASS", "FAIL"}
        )
        if not required.issubset(labels):
            raise ValueError("control set lacks positive/negative case controls")
    return tuple(results)


def readiness_report(packet: LoadedPacket) -> Q0Readiness:
    cases = packet.packet.cases
    return Q0Readiness(
        packet_id=packet.packet.packet_id,
        packet_version=packet.packet.schema_version,
        packet_sha256=packet.packet_sha256,
        packet_source_size_bytes=packet.packet_size_bytes,
        phase=packet.packet.phase,
        objective_case_ids=tuple(
            c.case_id
            for c in cases
            if c.scoring_method not in ("SANDBOX_DEFERRED", "EXPERT_DEFERRED")
        ),
        sandbox_case_ids=tuple(
            c.case_id for c in cases if c.scoring_method == "SANDBOX_DEFERRED"
        ),
        expert_case_ids=tuple(
            c.case_id for c in cases if c.scoring_method == "EXPERT_DEFERRED"
        ),
        q1_dependencies=(
            "D-017 exactly user-approved, hash-verified GGUF in per-host model registry",
            "Separately qualified pinned llama-server/host/runtime condition and resource isolation",
            "Prospective Q1 freeze of settings, prompts, sampling, reruns, reference and budgets",
            "Qualified sandbox for code execution and human/expert reference protocol",
            "Held-out cases and uncertainty methodology before inferential E-002 conclusions",
        ),
    )


def qualification_report(
    packet: LoadedPacket, controls: Q0ControlSet, controls_source: bytes
) -> Q0QualificationReport:
    _validate_control_source(controls_source)
    source_controls = Q0ControlSet.model_validate_json(controls_source)
    if source_controls != controls:
        raise ValueError("control source and evaluated typed controls differ")
    results = evaluate_all_controls(packet, source_controls)
    return Q0QualificationReport(
        readiness=readiness_report(packet),
        controls_sha256=sha256_hex(controls_source),
        controls_count=len(results),
        controls_qualified=True,
    )
