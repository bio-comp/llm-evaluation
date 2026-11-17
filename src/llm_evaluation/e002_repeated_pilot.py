"""E-002 objective-task repeated-sampling DEVELOPMENT packet and strict replay.

The repeated-sampling experiment cannot run without a verified approved model.
This module NEVER calls a provider or generates synthetic model evidence.
For complete independently collected 16-trial trajectories, it computes
descriptive fixed/adaptive decisions and existing finite-sample estimators.
No task-level true success probability, decision-error, calibration guarantee,
or dependence-adjusted population interval is inferred from 16 observations.
"""

from __future__ import annotations

import hashlib
import math
import platform
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .metrics import all_k, best_of_k, marginal
from .model_registry import DEFAULT_MODEL_REGISTRY, load_model_registry
from .runtime.llamacpp.qualification import inspect_llama_executable
from .stopping import Policy, decision

_SHA256 = r"^[a-f0-9]{64}$"
_FIXED = (1, 2, 4, 8, 16)
_LOOKS = (2, 4, 8, 16)
_INTEGER_ANSWER = re.compile(r"[+-]?[0-9]{1,20}", flags=re.ASCII)


class Closed(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class Task(Closed):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    lhs: int
    operation: Literal["+", "-", "*"]
    rhs: int
    expected: int
    prompt: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def independently_check_oracle(self) -> Self:
        calculated = {
            "+": lambda: self.lhs + self.rhs,
            "-": lambda: self.lhs - self.rhs,
            "*": lambda: self.lhs * self.rhs,
        }[self.operation]()
        if calculated != self.expected:
            raise ValueError("reference answer contradicts executable arithmetic")
        # v1 supports exactly this executable question grammar (and two
        # equivalent output-format suffixes), not arbitrary arithmetic prose.
        question = (
            f"Compute {self.lhs} {self.operation} {self.rhs}. "
            "Return exactly one decimal integer"
        )
        if self.prompt not in (question + ".", question + " with no explanation."):
            raise ValueError("prompt contradicts supported v1 arithmetic grammar")
        if len(str(abs(self.expected))) > 20:
            raise ValueError("reference answer exceeds the 20-digit grader domain")
        return self


class TaskPacket(Closed):
    schema_version: Literal[1]
    design_id: Literal["e002-objective-packet-dev-v1"]
    fixed_k: tuple[Literal[1, 2, 4, 8, 16], ...]
    adaptive_looks: tuple[Literal[2, 4, 8, 16], ...]
    tasks: tuple[Task, ...]

    @model_validator(mode="after")
    def check_grid(self) -> Self:
        if self.fixed_k != _FIXED or self.adaptive_looks != _LOOKS:
            raise ValueError("only prespecified fixed/adaptive grid is supported")
        if not self.tasks or len({t.id for t in self.tasks}) != len(self.tasks):
            raise ValueError("nonempty tasks with unique identifiers required")
        return self


class Grading(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    MALFORMED = "malformed"


def grade(task: Task, raw: str) -> Grading:
    """Exact objective integer grader, accepting sign/leading-zero alternatives."""
    value = raw.strip(" \t\r\n")
    if _INTEGER_ANSWER.fullmatch(value) is None:
        return Grading.MALFORMED
    return Grading.CORRECT if int(value) == task.expected else Grading.INCORRECT


class TrialStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CENSORED = "censored"


class Condition(Closed):
    """Captured model/run identity. Not a substitute for local registry approval."""

    registry_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=_SHA256)
    checkpoint_size_bytes: int = Field(gt=0)
    runtime: Literal["llamacpp"]
    runtime_sha256: str = Field(pattern=_SHA256)
    runtime_build: str = Field(min_length=1)
    chat_template_sha256: str = Field(pattern=_SHA256)
    context_limit: int = Field(ge=128)
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1, le=512)
    seed_supported: bool
    hardware: str = Field(min_length=1)


class Trial(Closed):
    """Raw UTF-8 response plus request, seed, ordering and cost provenance."""

    task_id: str
    ordinal: int = Field(ge=1)
    global_order: int = Field(ge=1)
    session_id: str = Field(min_length=1)
    seed: int
    prompt_sha256: str = Field(pattern=_SHA256)
    status: TrialStatus
    raw_response: str | None
    raw_sha256: str | None = Field(pattern=_SHA256)
    failure_reason: str | None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        if self.status is TrialStatus.COMPLETED and self.raw_response is None:
            raise ValueError("completed generation requires raw response")
        if self.status is TrialStatus.COMPLETED and self.failure_reason is not None:
            raise ValueError("completed generation cannot be a failure")
        if self.status is not TrialStatus.COMPLETED and not self.failure_reason:
            raise ValueError("failed/censored attempt requires a reason")
        if self.raw_response is None:
            if self.raw_sha256 is not None:
                raise ValueError("absent raw bytes cannot have a digest")
        elif (
            hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()
            != self.raw_sha256
        ):
            raise ValueError("raw response digest mismatch")
        return self


def _check_task_chronology(trials: tuple[Trial, ...]) -> None:
    """Each task's ordinals must increase with genuine global observation order."""
    last_ordinal: dict[str, int] = {}
    for trial in sorted(trials, key=lambda t: t.global_order):
        if trial.ordinal <= last_ordinal.get(trial.task_id, 0):
            raise ValueError("task ordinal contradicts global chronology")
        last_ordinal[trial.task_id] = trial.ordinal


class TrajectoryEvidence(Closed):
    schema_version: Literal[1]
    manifest_sha256: str = Field(pattern=_SHA256)
    condition: Condition
    trials: tuple[Trial, ...]

    @model_validator(mode="after")
    def unique_trials(self) -> Self:
        keys = [(t.task_id, t.ordinal) for t in self.trials]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task/ordinal trial")
        positions = [t.global_order for t in self.trials]
        if len(set(positions)) != len(positions):
            raise ValueError("duplicate global chronology position")
        _check_task_chronology(self.trials)
        return self


class ArmScore(Closed):
    policy: str
    decision: Literal["accept", "reject", "inconclusive"]
    consumed: int
    correct: int
    malformed: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


class KScore(Closed):
    k: int
    best_of_k: float
    all_k: float


class TaskScore(Closed):
    task_id: str
    marginal: float
    correct: int
    malformed: int
    metrics: tuple[KScore, ...]
    arms: tuple[ArmScore, ...]
    switches_from_first: int


class Evaluation(Closed):
    schema_version: Literal[1]
    status: Literal["evaluated", "incomplete"]
    task_count: int
    attempted_count: int
    completed_count: int
    failed_count: int
    censored_count: int
    missing_slots: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    task_weighted_marginal: float | None
    task_marginal_range: tuple[float, float] | None
    task_scores: tuple[TaskScore, ...]
    decision_error: None = None
    population_uncertainty_interval: None = None


def _arms(
    correctness: tuple[bool, ...],
    grades: tuple[Grading, ...],
    trials: tuple[Trial, ...],
) -> tuple[ArmScore, ...]:
    policies: tuple[tuple[str, Policy], ...] = tuple(
        (f"fixed-{k}", Policy(kind="fixed", looks=(k,))) for k in _FIXED
    ) + (
        ("group-10pct", Policy(kind="group", looks=_LOOKS)),
        ("sprt-p03-p07", Policy(kind="sprt", looks=_LOOKS)),
    )
    results: list[ArmScore] = []
    for name, policy in policies:
        consumed = 0
        verdict = "inconclusive"
        for n in range(1, policy.cap + 1):
            consumed = n
            if n in policy.looks:
                verdict = decision(policy, n, sum(correctness[:n]))
                if verdict != "inconclusive" or n == policy.cap:
                    break
        prefix = trials[:consumed]
        typed_verdict: Literal["accept", "reject", "inconclusive"]
        if verdict == "accept":
            typed_verdict = "accept"
        elif verdict == "reject":
            typed_verdict = "reject"
        elif verdict == "inconclusive":
            typed_verdict = "inconclusive"
        else:
            raise RuntimeError("unknown stopping kernel verdict")
        results.append(
            ArmScore(
                policy=name,
                decision=typed_verdict,
                consumed=consumed,
                correct=sum(correctness[:consumed]),
                malformed=sum(g is Grading.MALFORMED for g in grades[:consumed]),
                input_tokens=sum(t.input_tokens for t in prefix),
                output_tokens=sum(t.output_tokens for t in prefix),
                elapsed_seconds=math.fsum(t.elapsed_seconds for t in prefix),
            )
        )
    return tuple(results)


def packet_hash(payload: bytes) -> str:
    """Source-byte digest; whitespace changes intentionally change identity."""
    return hashlib.sha256(payload).hexdigest()


def evaluate(
    packet: TaskPacket, packet_bytes: bytes, evidence: TrajectoryEvidence
) -> Evaluation:
    """Replay complete, contiguous, chronologically recorded trajectories only."""
    if TaskPacket.model_validate_json(packet_bytes) != packet:
        raise ValueError("packet/source semantic mismatch")
    if packet_hash(packet_bytes) != evidence.manifest_sha256:
        raise ValueError("packet source digest mismatch")
    tasks = {task.id: task for task in packet.tasks}
    slots: dict[str, dict[int, Trial]] = {task.id: {} for task in packet.tasks}
    for trial in evidence.trials:
        task = tasks.get(trial.task_id)
        if task is None or trial.ordinal > 16:
            raise ValueError("unknown task or ordinal beyond frozen 16-cap")
        if (
            hashlib.sha256(task.prompt.encode("utf-8")).hexdigest()
            != trial.prompt_sha256
        ):
            raise ValueError("prompt/source digest mismatch")
        slots[task.id][trial.ordinal] = trial
    if tuple(t.global_order for t in evidence.trials) != tuple(
        sorted(t.global_order for t in evidence.trials)
    ):
        raise ValueError("trial records must preserve chronological global order")
    # Defensive replay guard: Pydantic's model_copy(update=...) skips validation.
    _check_task_chronology(evidence.trials)
    attempted = len(evidence.trials)
    completed = sum(t.status is TrialStatus.COMPLETED for t in evidence.trials)
    failed = sum(t.status is TrialStatus.FAILED for t in evidence.trials)
    censored = sum(t.status is TrialStatus.CENSORED for t in evidence.trials)
    missing = sum(16 - len(values) for values in slots.values())
    complete = missing == 0 and completed == 16 * len(tasks)
    scores: list[TaskScore] = []
    if complete:
        for task in packet.tasks:
            by_n = slots[task.id]
            ordered = tuple(by_n[n] for n in range(1, 17))
            grades = tuple(grade(task, t.raw_response or "") for t in ordered)
            correctness = tuple(g is Grading.CORRECT for g in grades)
            c = sum(correctness)
            # Malformed responses count as incorrect, but remain separately labeled.
            arms = _arms(correctness, grades, ordered)
            scores.append(
                TaskScore(
                    task_id=task.id,
                    marginal=marginal(16, c),
                    correct=c,
                    malformed=sum(g is Grading.MALFORMED for g in grades),
                    metrics=tuple(
                        KScore(
                            k=k, best_of_k=best_of_k(16, c, k), all_k=all_k(16, c, k)
                        )
                        for k in _FIXED
                    ),
                    arms=arms,
                    switches_from_first=sum(
                        arm.decision != arms[0].decision for arm in arms[1:]
                    ),
                )
            )
    proportions = [score.marginal for score in scores]
    return Evaluation(
        schema_version=1,
        status="evaluated" if complete else "incomplete",
        task_count=len(tasks),
        attempted_count=attempted,
        completed_count=completed,
        failed_count=failed,
        censored_count=censored,
        missing_slots=missing,
        input_tokens=sum(t.input_tokens for t in evidence.trials),
        output_tokens=sum(t.output_tokens for t in evidence.trials),
        elapsed_seconds=math.fsum(t.elapsed_seconds for t in evidence.trials),
        task_weighted_marginal=math.fsum(proportions) / len(proportions)
        if complete
        else None,
        task_marginal_range=(min(proportions), max(proportions)) if complete else None,
        task_scores=tuple(scores),
    )


class Preflight(Closed):
    schema_version: Literal[1] = 1
    status: Literal["BLOCKED_NOT_EXECUTED"] = "BLOCKED_NOT_EXECUTED"
    reason: Literal["NO_APPROVED_MODEL"] = "NO_APPROVED_MODEL"
    registry_schema: Literal[1] = 1
    registry_sha256: str = Field(pattern=_SHA256)
    packet_sha256: str = Field(pattern=_SHA256)
    runtime_sha256: str = Field(pattern=_SHA256)
    runtime_version: str
    runtime_build: int
    runtime_commit: str
    hardware_os: str
    hardware_arch: str
    approved_models: Literal[0] = 0
    sampled_tasks: Literal[0] = 0
    sampled_trials: Literal[0] = 0
    policy_results: None = None


def preflight_blocked(
    packet_bytes: bytes,
    executable_path: Path,
    registry_path: Path = DEFAULT_MODEL_REGISTRY,
) -> Preflight:
    """Emit a truthful blocked record only after source and runtime inspection.

    An approved model requires separate checkpoint+runtime qualification and
    this function fails, never treating a registry entry as evidence of sampling.
    """
    TaskPacket.model_validate_json(packet_bytes)
    registry = load_model_registry(registry_path)
    if registry.models:
        raise ValueError("approved model exists: require runtime qualification")
    executable = inspect_llama_executable(executable_path)
    return Preflight(
        registry_sha256=packet_hash(registry_path.read_bytes()),
        packet_sha256=packet_hash(packet_bytes),
        runtime_sha256=executable.sha256,
        runtime_version=executable.version,
        runtime_build=executable.build,
        runtime_commit=executable.commit,
        hardware_os=platform.system(),
        hardware_arch=platform.machine(),
    )
