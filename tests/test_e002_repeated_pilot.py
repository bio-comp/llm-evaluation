"""E-002 dev-only objective-oracle and replay negative controls.

All trajectories in THIS test module are explicitly synthetic software tests,
never observations from a model and never scientific pilot results.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from llm_evaluation.e002_repeated_pilot import (
    Condition,
    Grading,
    Task,
    TaskPacket,
    TrajectoryEvidence,
    Trial,
    TrialStatus,
    evaluate,
    grade,
    packet_hash,
    preflight_blocked,
)

_SOURCE = (
    Path(__file__).resolve().parents[1] / "fixtures/e002_objective_packet_dev_v1.json"
)
_BYTES = _SOURCE.read_bytes()
_PACKET = TaskPacket.model_validate_json(_BYTES)
_SHA = "a" * 64


def _condition() -> Condition:
    return Condition(
        registry_id="synthetic-ONLY-test-model",
        checkpoint_sha256=_SHA,
        checkpoint_size_bytes=123,
        runtime="llamacpp",
        runtime_sha256="b" * 64,
        runtime_build="test-only-not-a-real-model",
        chat_template_sha256="c" * 64,
        context_limit=2048,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        max_output_tokens=32,
        seed_supported=True,
        hardware="TEST-ONLY",
    )


def _record(
    task: Task,
    index: int,
    global_order: int,
    *,
    raw: str | None,
    status: TrialStatus = TrialStatus.COMPLETED,
) -> Trial:
    return Trial(
        task_id=task.id,
        ordinal=index,
        global_order=global_order,
        session_id="TEST-ONLY",
        seed=1000 + index,
        prompt_sha256=packet_hash(task.prompt.encode("utf-8")),
        status=status,
        raw_response=raw,
        raw_sha256=packet_hash(raw.encode("utf-8")) if raw is not None else None,
        failure_reason=None if status is TrialStatus.COMPLETED else "TEST ONLY",
        input_tokens=10,
        output_tokens=5 if raw is not None else 0,
        elapsed_seconds=0.25,
    )


def _evidence(*, missing: bool = False, failed: bool = False) -> TrajectoryEvidence:
    records = []
    for i in range(1, 17):
        for task in _PACKET.tasks:
            if missing and i == 16 and task == _PACKET.tasks[0]:
                continue
            is_failed = failed and i == 9 and task == _PACKET.tasks[0]
            # A test-only alternating trajectory; does not represent inference.
            raw = (
                None
                if is_failed
                else (
                    f"  +0{task.expected}"
                    if i % 2 == 0 and task.expected >= 0
                    else str(task.expected)
                    if i % 2 == 0
                    else str(task.expected + 1)
                )
            )
            records.append(
                _record(
                    task,
                    i,
                    1 + len(records),
                    raw=raw,
                    status=TrialStatus.FAILED if is_failed else TrialStatus.COMPLETED,
                )
            )
    return TrajectoryEvidence(
        schema_version=1,
        manifest_sha256=packet_hash(_BYTES),
        condition=_condition(),
        trials=tuple(records),
    )


def test_packet_six_public_safe_oracles_recompute_independently() -> None:
    assert len(_PACKET.tasks) == 6
    assert _PACKET.fixed_k == (1, 2, 4, 8, 16)
    assert _PACKET.adaptive_looks == (2, 4, 8, 16)
    for task in _PACKET.tasks:
        functions = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
        }
        assert functions[task.operation](task.lhs, task.rhs) == task.expected
        assert str(task.expected) not in task.prompt.split("Return exactly")[
            0
        ] or task.expected in (0, 42, 56, 132, -3, -13)
        assert grade(task, str(task.expected)) is Grading.CORRECT
        assert grade(task, str(task.expected + 1)) is Grading.INCORRECT


@pytest.mark.parametrize("text", ["56", " 56 ", "+056", "0056", "\t+56\r\n"])
def test_valid_alternate_answers_never_false_rejected(text: str) -> None:
    assert grade(_PACKET.tasks[0], text) is Grading.CORRECT


@pytest.mark.parametrize(
    "text", ["57", "-56", "fifty-six", "56.", "56 because", "56\n57", "５６", ""]
)
def test_wrong_and_malformed_controls(text: str) -> None:
    expected = Grading.INCORRECT if text in ("57", "-56") else Grading.MALFORMED
    assert grade(_PACKET.tasks[0], text) is expected


def test_replay_complete_synthetic_trajectory_accounting() -> None:
    evidence = _evidence()
    result = evaluate(_PACKET, _BYTES, evidence)
    assert result.status == "evaluated"
    assert result.task_count == 6 and result.attempted_count == 96
    assert (
        result.completed_count,
        result.failed_count,
        result.censored_count,
        result.missing_slots,
    ) == (96, 0, 0, 0)
    assert result.task_weighted_marginal == pytest.approx(0.5)
    assert result.task_marginal_range == (0.5, 0.5)
    assert result.input_tokens == 960 and result.output_tokens == 480
    assert result.elapsed_seconds == pytest.approx(24.0)
    assert (
        result.decision_error is None and result.population_uncertainty_interval is None
    )
    for task in result.task_scores:
        assert task.correct == 8 and task.malformed == 0
        assert task.marginal == 0.5
        assert tuple(r.k for r in task.metrics) == (1, 2, 4, 8, 16)
        for row in task.metrics:
            subsets = itertools.combinations([False, True] * 8, row.k)
            total = all_true = any_true = 0
            for s in subsets:
                total += 1
                any_true += any(s)
                all_true += all(s)
            assert row.best_of_k == pytest.approx(any_true / total)
            assert row.all_k == pytest.approx(all_true / total)
        for k, arm in zip((1, 2, 4, 8, 16), task.arms[:5], strict=True):
            assert arm.policy == f"fixed-{k}"
            assert arm.consumed == k
            assert arm.input_tokens == 10 * k
            assert arm.output_tokens == 5 * k
            assert arm.elapsed_seconds == pytest.approx(k / 4)
        assert task.arms[-2].policy == "group-10pct"
        assert task.arms[-1].policy == "sprt-p03-p07"
        assert all(
            a.decision in ("accept", "reject", "inconclusive") for a in task.arms
        )


@pytest.mark.parametrize("missing,failed", [(True, False), (False, True)])
def test_incomplete_and_failed_trajectory_never_published_as_comparison(
    missing: bool, failed: bool
) -> None:
    result = evaluate(_PACKET, _BYTES, _evidence(missing=missing, failed=failed))
    assert result.status == "incomplete"
    assert result.task_scores == ()
    assert result.task_weighted_marginal is None
    assert result.population_uncertainty_interval is None
    assert result.decision_error is None
    assert result.missing_slots == int(missing)
    assert result.failed_count == int(failed)
    assert result.completed_count == 96 - int(missing) - int(failed)
    assert result.elapsed_seconds > 0


def test_censor_is_not_counted_as_completed_or_graded_incorrect() -> None:
    original = _evidence()
    first = original.trials[0]
    censored = first.model_copy(
        update={
            "status": TrialStatus.CENSORED,
            "failure_reason": "TEST ONLY: output cap",
        }
    )
    result = evaluate(
        _PACKET,
        _BYTES,
        original.model_copy(
            update={
                "trials": (censored, *original.trials[1:]),
            }
        ),
    )
    assert result.status == "incomplete"
    assert result.censored_count == 1 and result.failed_count == 0
    assert result.completed_count == 95
    assert result.task_scores == ()


def test_source_hash_and_prompt_bytes_are_binding() -> None:
    original = _evidence()
    with pytest.raises(ValueError, match="packet source digest"):
        evaluate(_PACKET, _BYTES + b"\n", original)
    bad = original.trials[0].model_copy(update={"prompt_sha256": "b" * 64})
    altered = original.model_copy(update={"trials": (bad, *original.trials[1:])})
    with pytest.raises(ValueError, match="prompt/source"):
        evaluate(_PACKET, _BYTES, altered)


def test_duplicate_or_unsorted_chronology_rejected() -> None:
    original = _evidence()
    with pytest.raises(ValidationError):
        TrajectoryEvidence.model_validate(
            {
                **original.model_dump(mode="json"),
                "trials": [
                    original.trials[0].model_dump(mode="json"),
                    original.trials[0].model_dump(mode="json"),
                ],
            }
        )
    reordered = original.model_copy(
        update={
            "trials": (original.trials[1], original.trials[0], *original.trials[2:])
        }
    )
    with pytest.raises(ValueError, match="chronological"):
        evaluate(_PACKET, _BYTES, reordered)


def test_raw_tampering_rejected_with_strict_pydantic_boundaries() -> None:
    record = _evidence().trials[0].model_dump()
    record["raw_response"] = "mutated"
    with pytest.raises(ValidationError, match="digest"):
        Trial.model_validate(record)
    record = _evidence().trials[0].model_dump()
    record["input_tokens"] = "10"
    with pytest.raises(ValidationError):
        Trial.model_validate(record)
    record = _evidence().trials[0].model_dump()
    record["secret"] = "disallowed"
    with pytest.raises(ValidationError):
        Trial.model_validate(record)
    with pytest.raises(ValidationError):
        TaskPacket.model_validate_json(b'{"schema_version":1}')


def test_reference_answer_must_match_objective_executable_oracle() -> None:
    case = _PACKET.tasks[0].model_dump()
    case["expected"] = 1000
    with pytest.raises(ValidationError, match="contradicts executable arithmetic"):
        Task.model_validate(case)


def test_raw_finished_event_has_no_failure_reason() -> None:
    case = _evidence().trials[0].model_dump(mode="json")
    case["failure_reason"] = "fake"
    with pytest.raises(ValidationError):
        Trial.model_validate(case)


def test_blocked_preflight_no_model_call_and_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "model-registry.json"
    registry.write_text('{"schema_version":1,"models":[]}')
    monkeypatch.setattr(
        "llm_evaluation.e002_repeated_pilot.inspect_llama_executable",
        lambda _: SimpleNamespace(
            sha256="d" * 64,
            version="test-only",
            build=7,
            commit="aa",
        ),
    )
    result = preflight_blocked(_BYTES, tmp_path / "fake-bin", registry)
    assert result.status == "BLOCKED_NOT_EXECUTED"
    assert result.sampled_tasks == result.sampled_trials == 0
    assert result.approved_models == 0
    assert result.registry_sha256 == hashlib.sha256(registry.read_bytes()).hexdigest()
    assert result.packet_sha256 == hashlib.sha256(_BYTES).hexdigest()
    assert result.runtime_sha256 == "d" * 64
    assert result.model_validate_json(result.model_dump_json()) == result


def test_nonempty_registry_fails_closed_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text('{"schema_version":1,"models":[]}')
    monkeypatch.setattr(
        "llm_evaluation.e002_repeated_pilot.load_model_registry",
        lambda _: SimpleNamespace(models=(1,)),
    )

    def must_not_inspect(_: Path) -> None:
        raise AssertionError("must not inspect runtime with unqualified model")

    monkeypatch.setattr(
        "llm_evaluation.e002_repeated_pilot.inspect_llama_executable",
        must_not_inspect,
    )
    with pytest.raises(ValueError, match="require runtime qualification"):
        preflight_blocked(_BYTES, tmp_path / "fake-bin", registry)


@given(
    st.integers(min_value=-99999, max_value=99999),
    st.integers(min_value=1, max_value=99999),
)
@settings(max_examples=75, deadline=None)
def test_integer_grader_accepts_alternate_signs_and_zeros(a: int, b: int) -> None:
    case = Task(
        id="property",
        lhs=a,
        operation="+",
        rhs=b,
        expected=a + b,
        prompt=f"Compute {a} + {b}. Return exactly one decimal integer.",
    )
    answer = f"+0{a + b}" if a + b >= 0 else f"-00{abs(a + b)}"
    assert grade(case, answer) is Grading.CORRECT
    assert grade(case, str(a + b + 1)) is Grading.INCORRECT


def test_external_trajectory_cannot_claim_population_truth_or_error() -> None:
    r = evaluate(_PACKET, _BYTES, _evidence())
    assert all(math.isfinite(row.marginal) for row in r.task_scores)
    assert r.decision_error is None
    assert r.population_uncertainty_interval is None
    with pytest.raises(ValidationError):
        r.model_validate({**r.model_dump(), "decision_error": 0.0})


@pytest.mark.parametrize(
    "all_correct,expected",
    [
        (True, "accept"),
        (False, "reject"),
    ],
)
def test_adaptive_stopping_boundary_on_pure_test_trajectories(
    all_correct: bool, expected: str
) -> None:
    """An independent extreme path reveals early stopping and matched costs."""
    evidence = _evidence()
    rewritten = []
    for trial in evidence.trials:
        task = next(t for t in _PACKET.tasks if t.id == trial.task_id)
        raw = str(task.expected if all_correct else task.expected + 1)
        rewritten.append(
            Trial.model_validate(
                {
                    **trial.model_dump(),
                    "raw_response": raw,
                    "raw_sha256": packet_hash(raw.encode("utf-8")),
                }
            )
        )
    result = evaluate(
        _PACKET, _BYTES, evidence.model_copy(update={"trials": tuple(rewritten)})
    )
    assert result.status == "evaluated"
    for row in result.task_scores:
        assert row.arms[0].consumed == 1
        assert row.arms[0].decision == expected
        # Exact one-sided Binomial(8,.5) tail: 1/256 < .1/(2*4);
        # Binomial(4,.5) tail: 1/16 > .1/(2*4).
        group, sprt = row.arms[-2:]
        assert group.decision == expected and group.consumed == 8
        assert sprt.decision == expected and sprt.consumed == 4
        assert group.input_tokens == 80 and sprt.input_tokens == 40
        assert group.elapsed_seconds == 2.0 and sprt.elapsed_seconds == 1.0


def test_pilot_metrics_are_not_plugin_powers() -> None:
    """The exact finite-sample estimator differs from n-dependent plug-in."""
    score = evaluate(_PACKET, _BYTES, _evidence()).task_scores[0]
    k2 = next(m for m in score.metrics if m.k == 2)
    empirical = score.marginal
    assert empirical == 0.5
    assert k2.best_of_k != pytest.approx(1 - (1 - empirical) ** 2)
    assert k2.all_k != pytest.approx(empirical**2)


def test_no_trajectory_is_not_a_successful_zero_cost_evaluation() -> None:
    empty = TrajectoryEvidence(
        schema_version=1,
        manifest_sha256=packet_hash(_BYTES),
        condition=_condition(),
        trials=(),
    )
    result = evaluate(_PACKET, _BYTES, empty)
    assert result.status == "incomplete"
    assert result.task_count == 6
    assert result.attempted_count == 0
    assert result.missing_slots == 96
    assert result.task_scores == ()
    assert result.task_weighted_marginal is None
    assert result.decision_error is None
