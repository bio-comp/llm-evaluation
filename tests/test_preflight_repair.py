"""Regression controls for versioned, safe local preflight evidence."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_evaluation.local_runtime_preflight_records import (
    PreflightRecord,
    read_preflight_record,
)

pytest_plugins = ("test_runtime_preflight_harness",)


def test_old_unversioned_record_read_without_rewriting(tmp_path):
    source = Path("runs/task1-llamacpp-preflight-20261026T005055-0600.json")
    if not source.exists():
        pytest.skip("historical private record not installed")
    before = source.read_bytes()
    old = read_preflight_record(source)
    assert old.status == "failed"
    assert old.budget.server_launches == 2
    assert old.gates.structured_generation is None
    assert source.read_bytes() == before


def test_v2_evidence_closed_failure_vocabulary_and_completion(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    record = run_preflight(**fake)
    assert record.schema_version == 2
    assert record.cleanup_status == "verified"
    assert record.failure_stage is None
    assert record.failure_code is None
    data = record.model_dump()
    data["failure_stage"] = "secret-looking arbitrary diagnostic"
    with pytest.raises(ValidationError):
        PreflightRecord.model_validate(data)
    data = record.model_dump()
    data["gates"]["structured_generation"] = None
    with pytest.raises(ValidationError):
        PreflightRecord.model_validate(data)


def test_identity_mismatch_retains_failed_record_before_launch(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    result = run_preflight(**(fake | {"expected_sha256": "0" * 64}))
    assert result.status == "failed"
    assert result.failure_stage == "identity"
    assert result.budget.server_launches == 0
    assert result.budget.generation_submissions == 0
    assert read_preflight_record(fake["output"]) == result


def test_qualification_failure_retains_cleanup(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    executable = Path(fake["executable"])
    executable.write_text(executable.read_text().replace("'total_slots':1", "'total_slots':2"))
    result = run_preflight(**fake)
    assert result.status == "failed"
    assert result.cleanup_status == "verified"
    assert result.gates.cleanup is True
    assert result.failure_stage == "qualification"


def test_structured_failure_retains_stage_and_cleanup(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    executable = Path(fake["executable"])
    executable.write_text(executable.read_text().replace(
        "'choices':[{'message':{'content':'{\"value\":1}'",
        "'choices':[{'message':{'content':'not json'"
    ))
    result = run_preflight(**fake)
    assert result.status == "failed"
    assert result.gates.tiny_generation is True
    assert result.gates.structured_generation is False
    assert result.failure_stage == "structured_generation"
    assert result.failure_code == "structured_parse"
    assert result.cleanup_status == "verified"



def test_generation_summary_distinguishes_reasoning_from_visible_content():
    import json
    from llm_evaluation.local_runtime_preflight_records import summarize_generation
    from llm_evaluation.runtime.evidence import GenerationObservation

    observation = GenerationObservation(
        raw_output="", backend_model_id="preflight", finish_reason="length",
        termination_reason="length", prompt_tokens=8,
        completion_tokens=128, total_tokens=136,
        reported_model_matches_condition=True,
    )
    body = json.dumps({"choices": [{"message": {
        "content": "", "reasoning_content": "hidden reasoning"
    }, "finish_reason": "length"}]}).encode()
    summary = summarize_generation(observation, body)
    assert summary.visible_output_bytes == 0
    assert summary.reasoning_present is True
    assert summary.reasoning_bytes == len("hidden reasoning".encode())
    assert summary.completion_tokens == 128
    assert summary.finish_reason == "length"
    assert "hidden" not in summary.model_dump_json()
    with pytest.raises(ValidationError):
        summary.model_copy(update={"finish_reason": "secret-text"}).__class__.model_validate(
            summary.model_dump() | {"finish_reason": "secret-text"}
        )



def test_preflight_record_retains_safe_probe_summaries(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    record = run_preflight(**fake)
    assert len(record.generation_summaries) == 4
    assert record.generation_summaries[0].visible_output_bytes == len('{"value":1}')
    assert record.generation_summaries[1].reported_model_matches_condition is True
    assert 'reasoning_content' not in fake["output"].read_text()



def test_fake_rejects_known_bad_structured_wire(fake, monkeypatch):
    import json
    import llm_evaluation.local_runtime_preflight as preflight
    original = preflight.prepare_generation

    def wrong_wrapper(binding, request, structured=None):
        prepared = original(binding, request, structured)
        if structured is None:
            return prepared
        body = json.loads(prepared.wire_bytes)
        body["response_format"] = {"type": "unknown"}
        return prepared.model_copy(update={"wire_bytes": json.dumps(body).encode()})

    monkeypatch.setattr(preflight, "prepare_generation", wrong_wrapper)
    record = preflight.run_preflight(**fake)
    assert record.status == "failed"
    assert record.gates.structured_generation is False
    assert record.failure_stage == "structured_generation"
    assert record.failure_code == "http_status"


def test_preflight_binds_qualified_chat_template_identity(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    import hashlib
    first = run_preflight(**fake)
    assert first.chat_template_sha256 == hashlib.sha256(b"synthetic-template").hexdigest()
    executable = Path(fake["executable"])
    executable.write_text(executable.read_text().replace(
        "'chat_template':'synthetic-template'", "'chat_template':'changed-template'"
    ))
    second = run_preflight(**(fake | {"output": fake["output"].with_name("second.json")}))
    assert second.chat_template_sha256 != first.chat_template_sha256
    assert second.qualified_condition_sha256 != first.qualified_condition_sha256
    assert second.generation_summaries[0].reported_model_matches_condition is True


def test_structured_only_diagnostic_counts_one_launch_one_submission(fake):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    output = fake["output"].with_name("structured-diagnostic.json")
    result = run_structured_diagnostic(**(fake | {"output": output}), reasoning_mode="auto")
    assert result.status == "passed"
    assert result.server_launches == 1
    assert result.generation_submissions == 1
    assert result.cleanup_status == "verified"
    assert result.structured_generation is True
    assert result.summary.visible_output_bytes == len('{"value":1}')
    assert result.reasoning_profile == "auto"
    assert "value" not in output.read_text()


def test_structured_only_diagnostic_retains_reasoning_exhaustion(fake):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    executable = Path(fake["executable"])
    executable.write_text(executable.read_text().replace(
        "'content':'{\"value\":1}'",
        "'content':'','reasoning_content':'private reasoning'"
    ).replace("'finish_reason':'stop'", "'finish_reason':'length'"))
    result = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert result.status == "failed"
    assert result.failure_stage == "structured_generation"
    assert result.failure_code == "output_exhausted"
    assert result.summary.visible_output_bytes == 0
    assert result.summary.reasoning_present is True
    assert result.cleanup_status == "verified"
    assert result.server_launches == 1
    assert result.generation_submissions == 1
    assert "private reasoning" not in fake["output"].read_text()



def test_reasoning_profile_changes_qualified_condition(fake):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    auto = run_structured_diagnostic(**fake, reasoning_mode="auto")
    off = run_structured_diagnostic(
        **(fake | {"output": fake["output"].with_name("off.json")}),
        reasoning_mode="off",
    )
    assert auto.qualified_condition_sha256 != off.qualified_condition_sha256
    assert auto.chat_template_sha256 == off.chat_template_sha256


def test_formal_prompt_cache_policy_is_bound_into_condition(fake, monkeypatch):
    import llm_evaluation.local_runtime_preflight as preflight

    original = preflight.prepare_generation
    environments = []

    def capture(binding, request, structured=None):
        environments.append(binding.condition.runtime_environment)
        return original(binding, request, structured)

    monkeypatch.setattr(preflight, "prepare_generation", capture)
    result = preflight.run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert result.status == "passed"
    assert any(env.get("cache_prompt") is False for env in environments)


def test_preflight_retains_listener_and_agent_isolation(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    result = run_preflight(**fake)
    assert result.gates.listener_loopback is True
    assert result.gates.agent_tools_disabled is True


def test_wildcard_listener_cannot_pass_formal_preflight(fake, monkeypatch):
    from llm_evaluation.local_runtime_preflight import run_preflight
    from llm_evaluation.runtime.llamacpp.qualification import ListenerStatus
    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.observe_owned_listener",
        lambda pid, port: ListenerStatus.WILDCARD_DETECTED,
    )
    result = run_preflight(**fake)
    assert result.status == "failed"
    assert result.gates.listener_loopback is False
    assert result.cleanup_status == "verified"



def test_concurrency_retains_bounded_timestamped_slot_samples(fake):
    from llm_evaluation.local_runtime_preflight import run_preflight
    result = run_preflight(**fake)
    assert result.gates.concurrent_requests is True
    assert 1 <= len(result.slot_samples) <= 4000
    assert result.slot_poll_count >= len(result.slot_samples)
    assert all(s.slot_count == 1 and s.processing_count <= 1 for s in result.slot_samples)
    assert all(0 <= s.monotonic_offset_seconds <= result.slot_observation_seconds
               for s in result.slot_samples)



def test_historical_passed_gate_shape_reads_without_new_v2_gates(fake, tmp_path):
    import json
    from llm_evaluation.local_runtime_preflight import run_preflight
    record = run_preflight(**fake)
    historical = record.model_dump(mode="json")
    for key in (
        "schema_version", "failure_stage", "failure_code", "cleanup_status",
        "generation_summaries", "slot_samples", "chat_template_sha256",
        "chat_template_caps", "build_info_sha256", "qualified_condition_sha256",
        "reasoning_profile",
    ):
        historical.pop(key)
    historical["gates"].pop("listener_loopback")
    historical["gates"].pop("agent_tools_disabled")
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical))
    old = read_preflight_record(path)
    assert old.status == "passed"
    assert not hasattr(old.gates, "listener_loopback")


def test_reasoning_off_rejects_reasoning_with_valid_visible_answer(fake):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic

    executable = Path(fake["executable"])
    original = executable.read_text()
    changed = original.replace(
        "'content':'{\"value\":1}'",
        "'content':'{\"value\":1}','reasoning_content':'private reasoning'",
    )
    assert changed != original  # control must actually affect the synthetic server
    executable.write_text(changed)
    result = run_structured_diagnostic(**fake, reasoning_mode="off")
    assert result.status == "failed"
    assert result.failure_stage == "structured_generation"
    assert result.failure_code == "invalid_response"
    assert result.summary is not None and result.summary.reasoning_present
    assert result.cleanup_status == "verified"
    assert "private reasoning" not in fake["output"].read_text()



@pytest.mark.parametrize(
    ("finish_reason", "failure_code"),
    [("length", "output_exhausted"), ("surprise", "invalid_response"), (None, "invalid_response")],
)
def test_structured_diagnostic_rejects_unreliable_termination_even_with_valid_json(
    fake, finish_reason, failure_code
):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic

    executable = Path(fake["executable"])
    original = executable.read_text()
    changed = original.replace(
        "'finish_reason':'stop'",
        "'finish_reason':" + repr(finish_reason),
    )
    assert changed != original
    executable.write_text(changed)
    record = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert record.status == "failed"
    assert record.failure_stage == "structured_generation"
    assert record.failure_code == failure_code
    assert record.summary is not None and record.summary.visible_output_bytes > 0
    assert record.cleanup_status == "verified"



def test_structured_diagnostic_readback_rejects_contradictory_passed_evidence(fake):
    from copy import deepcopy
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    from llm_evaluation.local_runtime_preflight_records import StructuredDiagnosticRecord

    passed = run_structured_diagnostic(**fake, reasoning_mode="off")
    assert passed.status == "passed"
    source = passed.model_dump(mode="json")
    for field, value in (
        ("failure_code", "invalid_response"),
        ("remote_completion_unknown", True),
    ):
        tampered = deepcopy(source)
        tampered[field] = value
        with pytest.raises(ValidationError, match="passed"):
            StructuredDiagnosticRecord.model_validate(tampered)
    for field, value in (
        ("termination_reason", "length"),
        ("finish_reason", "unknown"),
        ("reported_model_matches_condition", False),
        ("visible_output_bytes", 0),
        ("reasoning_bytes", 10),
        ("completion_tokens", 129),
        ("total_tokens", 2049),
    ):
        tampered = deepcopy(source)
        tampered["summary"][field] = value
        if field == "reasoning_bytes":
            tampered["summary"]["reasoning_present"] = True
        with pytest.raises(ValidationError, match="passed"):
            StructuredDiagnosticRecord.model_validate(tampered)



@pytest.mark.parametrize(
    ("completion", "total"),
    [(129, 139), (5, 2049)],
)
def test_structured_diagnostic_cannot_pass_with_reported_budget_violation(
    fake, completion, total
):
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic

    executable = Path(fake["executable"])
    original = executable.read_text()
    needle = "'finish_reason':'stop'}]})"
    replacement = (
        "'finish_reason':'stop'}],'usage':"
        "{'prompt_tokens':10,'completion_tokens':" + str(completion)
        + ",'total_tokens':" + str(total) + "}})"
    )
    assert original.count(needle) == 1
    executable.write_text(original.replace(needle, replacement))
    result = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert result.status == "failed"
    assert result.failure_stage == "structured_generation"
    assert result.failure_code == "invalid_response"
    assert result.summary is not None
    assert result.summary.completion_tokens == completion
    assert result.summary.total_tokens == total
    assert result.cleanup_status == "verified"



def test_formal_jinja_template_engine_is_in_qualified_condition(fake, monkeypatch):
    import llm_evaluation.local_runtime_preflight as preflight

    prepare = preflight.prepare_generation
    observed = []

    def capture(binding, request, structured=None):
        observed.append(binding.condition.runtime_environment)
        return prepare(binding, request, structured)

    monkeypatch.setattr(preflight, "prepare_generation", capture)
    result = preflight.run_structured_diagnostic(**fake, reasoning_mode="off")
    assert result.status == "passed"
    assert observed
    assert all(env.get("jinja") is True for env in observed)

@pytest.mark.parametrize(
    "missing_field",
    (
        "executable",
        "artifact",
        "structured_domain_sha256",
        "structured_effective_sha256",
    ),
)
def test_structured_success_readback_requires_provenance(fake, tmp_path, missing_field):
    """Start from genuine synthetic success and remove exactly one datum."""
    import json
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    from llm_evaluation.local_runtime_preflight_records import StructuredDiagnosticRecord

    record = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert record.status == "passed"
    assert StructuredDiagnosticRecord.model_validate_json(fake["output"].read_bytes()) == record
    corrupted = record.model_dump(mode="json")
    assert corrupted[missing_field] is not None
    corrupted[missing_field] = None
    path = tmp_path / f"missing-{missing_field}.json"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValidationError, match="passed structured diagnostic"):
        StructuredDiagnosticRecord.model_validate_json(path.read_bytes())


def test_structured_success_readback_rejects_reasoning_presence_conflict(fake, tmp_path):
    """auto reasoning mode cannot excuse contradictory observed-byte metadata."""
    import json
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    from llm_evaluation.local_runtime_preflight_records import StructuredDiagnosticRecord

    record = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert record.status == "passed"
    assert record.summary is not None
    assert record.summary.reasoning_present is False
    assert record.summary.reasoning_bytes == 0
    corrupted = record.model_dump(mode="json")
    corrupted["summary"]["reasoning_bytes"] = 7
    path = tmp_path / "reasoning-contradiction.json"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValidationError, match="reasoning"):
        StructuredDiagnosticRecord.model_validate_json(path.read_bytes())


def test_structured_status_boundaries_preserve_real_success_failure_and_incomplete(
    fake, monkeypatch, tmp_path
):
    """Failed/incomplete observations need not invent absent identity evidence."""
    from llm_evaluation.local_runtime_preflight import run_structured_diagnostic
    from llm_evaluation.local_runtime_preflight_records import StructuredDiagnosticRecord
    from llm_evaluation.runtime.llamacpp.qualification import ExternalConflictStatus

    passed = run_structured_diagnostic(**fake, reasoning_mode="auto")
    assert passed.status == "passed"
    assert StructuredDiagnosticRecord.model_validate_json(fake["output"].read_bytes()) == passed

    failed_output = tmp_path / "failed.json"
    failed = run_structured_diagnostic(
        **(fake | {
            "executable": str(tmp_path / "missing-executable"),
            "output": failed_output,
        }),
        reasoning_mode="auto",
    )
    assert failed.status == "failed"
    assert failed.executable is None and failed.artifact is None
    assert failed.summary is None
    assert StructuredDiagnosticRecord.model_validate_json(failed_output.read_bytes()) == failed

    monkeypatch.setattr(
        "llm_evaluation.local_runtime_preflight.probe_known_external_conflicts",
        lambda: ExternalConflictStatus.CONFLICT_DETECTED,
    )
    incomplete_output = tmp_path / "incomplete.json"
    incomplete = run_structured_diagnostic(
        **(fake | {"output": incomplete_output}), reasoning_mode="auto",
    )
    assert incomplete.status == "incomplete"
    assert incomplete.summary is None
    assert StructuredDiagnosticRecord.model_validate_json(incomplete_output.read_bytes()) == incomplete

    # The v1 schema permits absent schema observations on failed/incomplete runs.
    import json
    for unfinished in (failed, incomplete):
        historical = unfinished.model_dump(mode="json")
        historical["structured_domain_sha256"] = None
        historical["structured_effective_sha256"] = None
        restored = StructuredDiagnosticRecord.model_validate_json(json.dumps(historical))
        assert restored.status == unfinished.status
        assert restored.summary is None

    # An observed but empty reasoning field is not contradictory.
    empty_reasoning = passed.model_dump(mode="json")
    empty_reasoning["summary"]["reasoning_present"] = True
    empty_reasoning["summary"]["reasoning_bytes"] = 0
    assert StructuredDiagnosticRecord.model_validate_json(
        json.dumps(empty_reasoning)
    ).status == "passed"
