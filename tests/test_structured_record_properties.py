"""Bounded property checks for versioned structured-diagnostic readback invariants."""

import json
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import ValidationError

from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
from llm_evaluation.local_runtime_preflight_records import StructuredDiagnosticRecord
from llm_evaluation.runtime.llamacpp.qualification import ExternalConflictStatus

# Immutable fixture records are reused; each generated example mutates only a fresh JSON copy.
pytest_plugins = ("test_runtime_preflight_harness",)

_SUCCESS_PROVENANCE = (
    "executable",
    "artifact",
    "structured_domain_sha256",
    "structured_effective_sha256",
)


@pytest.fixture
def genuine_success(fake: dict[str, Any]) -> StructuredDiagnosticRecord:
    result = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert result.status == "passed"
    assert StructuredDiagnosticRecord.model_validate_json(fake["output"].read_bytes()) == result
    return result


@pytest.fixture
def genuine_unfinished(
    fake: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> dict[str, StructuredDiagnosticRecord]:
    failed = run_structured_diagnostic(
        **(fake | {
            "executable": str(tmp_path / "nonexistent-executable"),
            "output": tmp_path / "failed.json",
        }),
        reasoning_mode="auto",
    )
    assert failed.status == "failed"
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.CONFLICT_DETECTED,
    )
    incomplete = run_structured_diagnostic(
        **(fake | {"output": tmp_path / "incomplete.json"}),
        reasoning_mode="auto",
    )
    assert incomplete.status == "incomplete"
    return {"failed": failed, "incomplete": incomplete}


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(missing=st.sets(st.sampled_from(_SUCCESS_PROVENANCE), min_size=1))
def test_all_missing_success_provenance_subsets_fail_readback(
    genuine_success: StructuredDiagnosticRecord, missing: set[str]
) -> None:
    """All nonempty subsets are forbidden, not only single missing fields."""
    corrupt = genuine_success.model_dump(mode="json")
    for field in missing:
        corrupt[field] = None
    with pytest.raises(ValidationError, match="passed structured diagnostic"):
        StructuredDiagnosticRecord.model_validate_json(json.dumps(corrupt))


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(present=st.booleans(), count=st.integers(min_value=0, max_value=2048))
def test_reasoning_presence_byte_count_consistency(
    genuine_success: StructuredDiagnosticRecord, present: bool, count: int
) -> None:
    """Auto reasoning permits genuine content but never contradictory metadata."""
    record = genuine_success.model_dump(mode="json")
    record["summary"]["reasoning_present"] = present
    record["summary"]["reasoning_bytes"] = count
    encoded = json.dumps(record)
    if count > 0 and not present:
        with pytest.raises(ValidationError, match="reasoning"):
            StructuredDiagnosticRecord.model_validate_json(encoded)
    else:
        restored = StructuredDiagnosticRecord.model_validate_json(encoded)
        assert restored.status == "passed"
        assert restored.summary is not None
        assert restored.summary.reasoning_present is present
        assert restored.summary.reasoning_bytes == count


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    status=st.sampled_from(("failed", "incomplete")),
    missing=st.sets(st.sampled_from(_SUCCESS_PROVENANCE), min_size=1),
)
def test_unfinished_records_allow_unavailable_provenance(
    genuine_unfinished: dict[str, StructuredDiagnosticRecord],
    status: str,
    missing: set[str],
) -> None:
    """Do not impose success-only observations on failures or incomplete runs."""
    original = genuine_unfinished[status]
    assert original.summary is None
    record = original.model_dump(mode="json")
    for field in missing:
        record[field] = None
    restored = StructuredDiagnosticRecord.model_validate_json(json.dumps(record))
    assert restored.status == status
    assert restored.summary is None
