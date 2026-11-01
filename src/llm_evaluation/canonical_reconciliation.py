from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from canonical_sync.codec import (
    canonical_record_bytes,
    canonical_record_hash,
    parse_canonical_record,
)
from canonical_sync.git_publication import PublicationFile, PublicationTree
from canonical_sync.records import GitPublicationReceiptRecord

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ISO8601_UTC_SECOND_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_PUBLIC_REPOSITORY: Literal["bio-comp/llm-evaluation"] = "bio-comp/llm-evaluation"
_PUBLIC_MAIN_REF: Literal["refs/heads/main"] = "refs/heads/main"
_PUBLIC_VERIFY_CHECK: Literal["verify"] = "verify"
_PUBLIC_GIT_REMOTES = frozenset(
    {
        "https://github.com/bio-comp/llm-evaluation",
        "https://github.com/bio-comp/llm-evaluation.git",
        "git@github.com:bio-comp/llm-evaluation.git",
        "ssh://git@github.com/bio-comp/llm-evaluation.git",
    }
)


class _StrictFrozenProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def _reject_non_exact_int_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an exact integer")
        return value


class ProjectImpact(StrEnum):
    NO_PUBLIC_IMPACT = "no_public_impact"
    PUBLIC_CHANGE_REQUIRED = "public_change_required"
    UNKNOWN_OR_UNSAFE = "unknown_or_unsafe"


class ReconciliationDisposition(StrEnum):
    NONE = "NONE"
    REQUIRED = "REQUIRED"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"


class LastReconciledStateRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.last-reconciled-state"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_last_reconciled_state"] = (
        "llm_evaluation_last_reconciled_state"
    )
    schema_version: Literal[1] = 1
    snapshot_state_sha256: str

    @field_validator("snapshot_state_sha256")
    @classmethod
    def validate_snapshot_hash(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("snapshot identity must be a lowercase SHA-256")
        return value


class GitHubPullRequestMergeRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.github-pr-merge"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_github_pr_merge"] = "llm_evaluation_github_pr_merge"
    schema_version: Literal[2] = 2
    repository: Literal["bio-comp/llm-evaluation"] = _PUBLIC_REPOSITORY
    pull_request_number: int
    head_ref: str
    head_commit_sha: str
    base_ref: Literal["refs/heads/main"] = _PUBLIC_MAIN_REF
    merged_commit_sha: str
    merged_at: str

    @field_validator("pull_request_number")
    @classmethod
    def validate_pull_request_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("pull-request number must be positive")
        return value

    @field_validator("head_ref")
    @classmethod
    def validate_head_ref(cls, value: str) -> str:
        if not value.startswith("refs/heads/") or value == _PUBLIC_MAIN_REF:
            raise ValueError("reviewed pull request must identify a non-main branch ref (refs/heads/...)")
        return value

    @field_validator("head_commit_sha", "merged_commit_sha")
    @classmethod
    def validate_commit_sha(cls, value: str) -> str:
        if _GIT_OID_RE.fullmatch(value) is None:
            raise ValueError("project Git commits must be lowercase Git object IDs")
        return value

    @field_validator("merged_at")
    @classmethod
    def validate_merged_at(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("merged_at must be non-empty and trimmed")
        if not _ISO8601_UTC_SECOND_RE.fullmatch(value):
            raise ValueError(
                "merged_at must be an exact ISO 8601 UTC timestamp in format 'YYYY-MM-DDTHH:MM:SSZ'"
            )
        try:
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except Exception as exc:
            raise ValueError(f"merged_at must be a valid calendar date and time: {exc}") from exc
        return value


class GitHubCIVerificationRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.github-ci-verification"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_github_ci"] = "llm_evaluation_github_ci"
    schema_version: Literal[3] = 3
    repository: Literal["bio-comp/llm-evaluation"] = _PUBLIC_REPOSITORY
    run_id: int
    workflow_name: Literal["Public CI"] = "Public CI"
    workflow_path: Literal[".github/workflows/ci.yml"] = ".github/workflows/ci.yml"
    event: Literal["push"] = "push"
    head_branch: Literal["refs/heads/main"] = _PUBLIC_MAIN_REF
    commit_sha: str
    run_conclusion: Literal["success"] = "success"
    job_id: int
    job_name: Literal["verify"] = _PUBLIC_VERIFY_CHECK
    job_conclusion: Literal["success"] = "success"

    @field_validator("run_id", "job_id")
    @classmethod
    def validate_positive_ids(cls, value: int) -> int:
        if value < 1:
            raise ValueError("CI run ID and job ID must be positive")
        return value

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, value: str) -> str:
        if _GIT_OID_RE.fullmatch(value) is None:
            raise ValueError("CI commit must be a lowercase Git object ID")
        return value


class VerifiedPublicationReceiptRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.verified-publication-receipt"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_verified_publication"] = (
        "llm_evaluation_verified_publication"
    )
    schema_version: Literal[4] = 4
    snapshot_state_sha256: str
    public_contract_sha256: str
    publication_tree_sha256: str
    git: GitPublicationReceiptRecord
    pull_request: GitHubPullRequestMergeRecord
    ci: GitHubCIVerificationRecord

    @field_validator(
        "snapshot_state_sha256",
        "public_contract_sha256",
        "publication_tree_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("publication binding identities must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_project_transaction(self) -> Self:
        if self.git.remote not in _PUBLIC_GIT_REMOTES:
            raise ValueError("completion Git evidence must bind the public llm-evaluation repository")
        if self.git.remote_ref != _PUBLIC_MAIN_REF or self.git.status != "unchanged":
            raise ValueError("completion Git evidence must be post-merge main-branch readback")
        merged_commit = self.git.resulting_commit
        if (
            self.pull_request.merged_commit_sha != merged_commit
            or self.ci.commit_sha != merged_commit
        ):
            raise ValueError("PR merge, Git readback, and CI must bind the same merged commit")
        return self


class ReconciliationDispositionRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.reconciliation-disposition"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_reconciliation_disposition"] = (
        "llm_evaluation_reconciliation_disposition"
    )
    schema_version: Literal[3] = 3
    current_snapshot_sha256: str
    previous_snapshot_sha256: str | None = None
    impact: ProjectImpact
    disposition: ReconciliationDisposition
    public_contract_sha256: str | None = None
    publication_tree_sha256: str | None = None
    verified_publication: VerifiedPublicationReceiptRecord | None = None

    @field_validator(
        "current_snapshot_sha256",
        "previous_snapshot_sha256",
        "public_contract_sha256",
        "publication_tree_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError("reconciliation identities must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        changed = self.previous_snapshot_sha256 != self.current_snapshot_sha256
        if self.disposition is ReconciliationDisposition.NONE:
            if self.impact is not ProjectImpact.NO_PUBLIC_IMPACT:
                raise ValueError("NONE disposition requires NO_PUBLIC_IMPACT")
            if (
                self.public_contract_sha256 is not None
                or self.publication_tree_sha256 is not None
            ):
                raise ValueError("NONE disposition must not declare publication hashes")
            if self.verified_publication is not None:
                raise ValueError("NONE cannot contain verified publication evidence")
            return self

        if not changed:
            raise ValueError(f"{self.disposition.value} disposition requires a changed snapshot")

        if self.disposition is ReconciliationDisposition.BLOCKED:
            if self.impact is not ProjectImpact.UNKNOWN_OR_UNSAFE:
                raise ValueError("BLOCKED disposition requires UNKNOWN_OR_UNSAFE")
            if (
                self.public_contract_sha256 is not None
                or self.publication_tree_sha256 is not None
            ):
                raise ValueError("BLOCKED disposition must not declare publication hashes")
            if self.verified_publication is not None:
                raise ValueError("BLOCKED needs unknown/unsafe impact and no completion receipt")
            return self

        if self.disposition is ReconciliationDisposition.REQUIRED:
            if self.impact is not ProjectImpact.PUBLIC_CHANGE_REQUIRED:
                raise ValueError("REQUIRED needs public-change-required impact")
            if (
                self.public_contract_sha256 is None
                or self.publication_tree_sha256 is None
            ):
                raise ValueError("REQUIRED needs an approved contract/tree and no completion receipt")
            if self.verified_publication is not None:
                raise ValueError("REQUIRED needs an approved contract/tree and no completion receipt")
            return self

        if self.disposition is ReconciliationDisposition.COMPLETE:
            receipt = self.verified_publication
            if (
                self.impact is not ProjectImpact.PUBLIC_CHANGE_REQUIRED
                or self.public_contract_sha256 is None
                or self.publication_tree_sha256 is None
                or receipt is None
                or receipt.snapshot_state_sha256 != self.current_snapshot_sha256
                or receipt.public_contract_sha256 != self.public_contract_sha256
                or receipt.publication_tree_sha256 != self.publication_tree_sha256
                or receipt.git.publication_tree_sha256 != self.publication_tree_sha256
                or not receipt.git.readback_verified
                or not receipt.git.fast_forward_only
                or receipt.pull_request.merged_commit_sha != receipt.git.resulting_commit
                or receipt.ci.commit_sha != receipt.git.resulting_commit
            ):
                raise ValueError("COMPLETE requires exact verified Git/readback/CI evidence")
            return self

        raise ValueError(f"unsupported disposition: {self.disposition}")


class ReconciliationStateError(RuntimeError):
    """Raised when durable reconciliation state persistence, readback, or verification fails."""


class DurableReconciliationStateRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.durable-reconciliation-state"
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_durable_reconciliation_state"] = (
        "llm_evaluation_durable_reconciliation_state"
    )
    schema_version: Literal[1] = 1
    state: LastReconciledStateRecord
    disposition: ReconciliationDispositionRecord

    @model_validator(mode="after")
    def validate_durable_consistency(self) -> Self:
        if self.disposition.disposition is not ReconciliationDisposition.COMPLETE:
            raise ValueError(
                "durable reconciliation state requires disposition COMPLETE, "
                f"got {self.disposition.disposition.value}"
            )
        if self.disposition.verified_publication is None:
            raise ValueError(
                "durable reconciliation state requires verified_publication evidence"
            )
        if self.state.snapshot_state_sha256 != self.disposition.current_snapshot_sha256:
            raise ValueError(
                "durable state snapshot hash must equal disposition current_snapshot_sha256: "
                f"{self.state.snapshot_state_sha256} != {self.disposition.current_snapshot_sha256}"
            )
        return self


class ReconciliationPolicyContract(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.reconciliation-policy"
    canonical_identity_version: ClassVar[int] = 1
    projection_type: Literal["llm_evaluation_reconciliation_policy"] = (
        "llm_evaluation_reconciliation_policy"
    )
    schema_version: Literal[2] = 2
    dispositions: tuple[Literal["NONE", "REQUIRED", "BLOCKED", "COMPLETE"], ...]
    complete_requires_snapshot_match: bool
    complete_requires_contract_match: bool
    complete_requires_readback: bool
    complete_requires_authoritative_github_evidence: bool
    complete_requires_merged_pull_request: bool
    complete_requires_post_merge_main_push_ci: bool
    complete_requires_exact_workflow_identity: bool
    complete_requires_exact_required_job_identity: bool


class ReviewedCanonicalImpact(_StrictFrozenProjectRecord):
    impact: ProjectImpact
    public_contract: ReconciliationPolicyContract | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.impact is ProjectImpact.PUBLIC_CHANGE_REQUIRED:
            if self.public_contract is None:
                raise ValueError("public-change-required review must include a public contract")
        elif self.public_contract is not None:
            raise ValueError("only a public-change-required review may include a public contract")
        return self


@dataclass(frozen=True, slots=True)
class ProjectReconciliationResult:
    disposition: ReconciliationDispositionRecord
    public_projection: ReconciliationPolicyContract | None
    publication_tree: PublicationTree | None


def current_executable_policy_contract() -> ReconciliationPolicyContract:
    return ReconciliationPolicyContract(
        dispositions=("NONE", "REQUIRED", "BLOCKED", "COMPLETE"),
        complete_requires_snapshot_match=True,
        complete_requires_contract_match=True,
        complete_requires_readback=True,
        complete_requires_authoritative_github_evidence=True,
        complete_requires_merged_pull_request=True,
        complete_requires_post_merge_main_push_ci=True,
        complete_requires_exact_workflow_identity=True,
        complete_requires_exact_required_job_identity=True,
    )


def approved_policy_tree(contract: ReconciliationPolicyContract) -> PublicationTree:
    if not isinstance(contract, ReconciliationPolicyContract):
        raise TypeError("approved policy publication requires a ReconciliationPolicyContract")
    return PublicationTree(
        files=(
            PublicationFile(
                path="fixtures/reconciliation_policy_contract.json",
                data=canonical_record_bytes(contract),
            ),
        )
    )


def verify_executable_policy_contract(authoritative: ReconciliationPolicyContract) -> None:
    if authoritative != current_executable_policy_contract():
        raise ValueError("executable reconciliation policy is stale relative to reviewed authority")


def classify_reconciliation(
    *,
    current_snapshot_sha256: str,
    previous: LastReconciledStateRecord | None,
    impact: ProjectImpact,
    public_contract_sha256: str | None = None,
    publication_tree_sha256: str | None = None,
) -> ReconciliationDispositionRecord:
    previous_hash = previous.snapshot_state_sha256 if previous is not None else None
    if previous_hash == current_snapshot_sha256:
        return ReconciliationDispositionRecord(
            current_snapshot_sha256=current_snapshot_sha256,
            previous_snapshot_sha256=previous_hash,
            impact=ProjectImpact.NO_PUBLIC_IMPACT,
            disposition=ReconciliationDisposition.NONE,
        )
    if impact is ProjectImpact.NO_PUBLIC_IMPACT:
        disposition = ReconciliationDisposition.NONE
    elif impact is ProjectImpact.PUBLIC_CHANGE_REQUIRED:
        if public_contract_sha256 is None or publication_tree_sha256 is None:
            raise ValueError("public change requires approved contract and publication-tree identities")
        disposition = ReconciliationDisposition.REQUIRED
    else:
        disposition = ReconciliationDisposition.BLOCKED
    return ReconciliationDispositionRecord(
        current_snapshot_sha256=current_snapshot_sha256,
        previous_snapshot_sha256=previous_hash,
        impact=impact,
        disposition=disposition,
        public_contract_sha256=public_contract_sha256,
        publication_tree_sha256=publication_tree_sha256,
    )


def complete_reconciliation(
    required: ReconciliationDispositionRecord,
    receipt: VerifiedPublicationReceiptRecord,
) -> ReconciliationDispositionRecord:
    if required.disposition is not ReconciliationDisposition.REQUIRED:
        raise ValueError("only a REQUIRED reconciliation can become COMPLETE")
    try:
        return ReconciliationDispositionRecord(
            current_snapshot_sha256=required.current_snapshot_sha256,
            previous_snapshot_sha256=required.previous_snapshot_sha256,
            impact=required.impact,
            disposition=ReconciliationDisposition.COMPLETE,
            public_contract_sha256=required.public_contract_sha256,
            publication_tree_sha256=required.publication_tree_sha256,
            verified_publication=receipt,
        )
    except ValueError as exc:
        raise ValueError("reconciliation cannot complete without exact verified evidence") from exc


def reconcile_reviewed_snapshot(
    *,
    current_snapshot_sha256: str,
    previous: LastReconciledStateRecord | None,
    reviewed: ReviewedCanonicalImpact,
) -> ProjectReconciliationResult:
    contract = reviewed.public_contract
    tree = approved_policy_tree(contract) if contract is not None else None
    contract_hash = canonical_record_hash(contract) if contract is not None else None
    tree_hash = tree.sha256 if tree is not None else None
    disposition = classify_reconciliation(
        current_snapshot_sha256=current_snapshot_sha256,
        previous=previous,
        impact=reviewed.impact,
        public_contract_sha256=contract_hash,
        publication_tree_sha256=tree_hash,
    )
    if disposition.disposition is not ReconciliationDisposition.REQUIRED:
        contract = None
        tree = None
    return ProjectReconciliationResult(
        disposition=disposition,
        public_projection=contract,
        publication_tree=tree,
    )


DEFAULT_RECONCILIATION_STATE_PATH: Final = (
    Path.home() / ".config" / "llm-evaluation" / "last-reconciled-state.json"
)
RECONCILIATION_STATE_ENV_VAR: Final = "LLM_EVALUATION_RECONCILIATION_STATE_PATH"


def resolve_reconciliation_state_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_path = os.environ.get(RECONCILIATION_STATE_ENV_VAR)
    if env_path is not None and env_path.strip():
        return Path(env_path.strip()).expanduser()
    return DEFAULT_RECONCILIATION_STATE_PATH.expanduser()


def load_durable_reconciliation_state(
    path: Path | str | None = None,
) -> DurableReconciliationStateRecord | None:
    target = resolve_reconciliation_state_path(path)
    if target.is_symlink():
        raise ReconciliationStateError(
            f"durable reconciliation state path must not be a symlink: {target}"
        )
    if not target.exists():
        return None
    if not target.is_file():
        raise ReconciliationStateError(
            f"durable reconciliation state path must be a regular file: {target}"
        )
    try:
        raw_bytes = target.read_bytes()
    except OSError as exc:
        raise ReconciliationStateError(
            f"failed to read durable reconciliation state file '{target}': {exc}"
        ) from exc
    if not raw_bytes:
        raise ReconciliationStateError(
            f"durable reconciliation state file '{target}' is empty"
        )
    try:
        record = parse_canonical_record(raw_bytes, DurableReconciliationStateRecord)
    except Exception as exc:
        raise ReconciliationStateError(
            f"failed to parse durable reconciliation state from '{target}': {exc}"
        ) from exc
    return record


def load_last_reconciled_state(
    path: Path | str | None = None,
) -> LastReconciledStateRecord | None:
    durable = load_durable_reconciliation_state(path)
    return durable.state if durable is not None else None


def save_durable_reconciliation_state(
    record: DurableReconciliationStateRecord,
    path: Path | str | None = None,
) -> Path:
    if not isinstance(record, DurableReconciliationStateRecord):
        raise TypeError(
            f"expected DurableReconciliationStateRecord, got {type(record).__name__}"
        )
    target = resolve_reconciliation_state_path(path)
    if target.is_symlink():
        raise ReconciliationStateError(
            f"durable reconciliation state target must not be a symlink: {target}"
        )
    parent = target.parent
    if parent.is_symlink():
        raise ReconciliationStateError(
            f"durable reconciliation state parent directory must not be a symlink: {parent}"
        )
    try:
        parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(parent, 0o700)
        except OSError:
            pass
    except OSError as exc:
        raise ReconciliationStateError(
            f"failed to create state parent directory '{parent}': {exc}"
        ) from exc

    payload = canonical_record_bytes(record)
    temp_path = parent / f".last-reconciled-state.tmp.{os.getpid()}.{time.time_ns()}.json"
    try:
        fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(fd, "wb", closefd=True) as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(temp_path), str(target))
        try:
            dir_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise ReconciliationStateError(
            f"failed to atomically persist durable reconciliation state: {exc}"
        ) from exc

    # Readback verification
    try:
        readback_bytes = target.read_bytes()
    except OSError as exc:
        raise ReconciliationStateError(
            f"readback verification failed to read '{target}': {exc}"
        ) from exc
    if readback_bytes != payload:
        raise ReconciliationStateError(
            f"readback verification failed: written bytes do not match expected canonical bytes for '{target}'"
        )
    try:
        readback_record = parse_canonical_record(readback_bytes, DurableReconciliationStateRecord)
    except Exception as exc:
        raise ReconciliationStateError(
            f"readback verification failed to parse record from '{target}': {exc}"
        ) from exc
    if canonical_record_hash(readback_record) != canonical_record_hash(record):
        raise ReconciliationStateError(
            f"readback verification failed: hash mismatch ({canonical_record_hash(readback_record)} != {canonical_record_hash(record)})"
        )
    return target


def persist_completed_reconciliation(
    completed: ReconciliationDispositionRecord,
    path: Path | str | None = None,
) -> DurableReconciliationStateRecord:
    if completed.disposition is not ReconciliationDisposition.COMPLETE:
        raise ValueError(
            f"only a COMPLETE disposition can be durably persisted, got {completed.disposition.value}"
        )
    if completed.verified_publication is None:
        raise ValueError(
            "persisted completion requires a verified_publication receipt"
        )
    state = LastReconciledStateRecord(
        snapshot_state_sha256=completed.current_snapshot_sha256
    )
    durable = DurableReconciliationStateRecord(
        state=state,
        disposition=completed,
    )
    save_durable_reconciliation_state(durable, path=path)
    return durable
