from __future__ import annotations

import hashlib
import json
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
_ISO8601_UTC_SECOND_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
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
_POLICY_FIXTURE_PATH: Final = "fixtures/reconciliation_policy_contract.json"


def _normalize_obligation_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        raise ValueError("public reconciliation requires at least one obligation ID")
    normalized = tuple(sorted(value))
    for item in normalized:
        if not item or item != item.strip() or any(ord(char) < 32 for char in item):
            raise ValueError(
                "obligation IDs must be non-empty, trimmed, and contain no control characters"
            )
    if len(normalized) != len(set(normalized)):
        raise ValueError("obligation IDs must be unique")
    return normalized


def _validate_contract_in_publication_tree(
    contract: "ReconciliationPolicyContract", tree: PublicationTree
) -> None:
    contract_bytes = tree.as_dict().get(_POLICY_FIXTURE_PATH)
    if contract_bytes is None:
        raise ValueError(
            f"approved publication tree must contain '{_POLICY_FIXTURE_PATH}'"
        )
    if contract_bytes != canonical_record_bytes(contract):
        raise ValueError(
            "approved publication tree policy fixture must exactly match the reviewed contract"
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
    record_type: Literal["llm_evaluation_github_pr_merge"] = (
        "llm_evaluation_github_pr_merge"
    )
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
            raise ValueError(
                "reviewed pull request must identify a non-main branch ref (refs/heads/...)"
            )
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
            raise ValueError(
                f"merged_at must be a valid calendar date and time: {exc}"
            ) from exc
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
    canonical_identity_domain: ClassVar[str] = (
        "llm-evaluation.verified-publication-receipt"
    )
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_verified_publication"] = (
        "llm_evaluation_verified_publication"
    )
    schema_version: Literal[5] = 5
    snapshot_state_sha256: str
    public_contract_sha256: str
    publication_tree_sha256: str
    obligation_ids: tuple[str, ...]
    git: GitPublicationReceiptRecord
    pull_request: GitHubPullRequestMergeRecord
    ci: GitHubCIVerificationRecord

    @field_validator("obligation_ids")
    @classmethod
    def validate_obligation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_obligation_ids(value)

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
            raise ValueError(
                "completion Git evidence must bind the public llm-evaluation repository"
            )
        if self.git.remote_ref != _PUBLIC_MAIN_REF or self.git.status != "unchanged":
            raise ValueError(
                "completion Git evidence must be post-merge main-branch readback"
            )
        merged_commit = self.git.resulting_commit
        if (
            self.pull_request.merged_commit_sha != merged_commit
            or self.ci.commit_sha != merged_commit
        ):
            raise ValueError(
                "PR merge, Git readback, and CI must bind the same merged commit"
            )
        return self


class ReconciliationDispositionRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = (
        "llm-evaluation.reconciliation-disposition"
    )
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_reconciliation_disposition"] = (
        "llm_evaluation_reconciliation_disposition"
    )
    schema_version: Literal[4] = 4
    current_snapshot_sha256: str
    previous_snapshot_sha256: str | None = None
    impact: ProjectImpact
    disposition: ReconciliationDisposition
    public_contract_sha256: str | None = None
    publication_tree_sha256: str | None = None
    obligation_ids: tuple[str, ...] = ()
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

    @field_validator("obligation_ids")
    @classmethod
    def validate_obligation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            return value
        return _normalize_obligation_ids(value)

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
            if self.obligation_ids:
                raise ValueError(
                    "NONE must not declare public reconciliation obligations"
                )
            if self.verified_publication is not None:
                raise ValueError("NONE cannot contain verified publication evidence")
            return self

        if not changed:
            raise ValueError(
                f"{self.disposition.value} disposition requires a changed snapshot"
            )

        if self.disposition is ReconciliationDisposition.BLOCKED:
            if self.impact is not ProjectImpact.UNKNOWN_OR_UNSAFE:
                raise ValueError("BLOCKED disposition requires UNKNOWN_OR_UNSAFE")
            if (
                self.public_contract_sha256 is not None
                or self.publication_tree_sha256 is not None
            ):
                raise ValueError(
                    "BLOCKED disposition must not declare publication hashes"
                )
            if self.obligation_ids:
                raise ValueError(
                    "BLOCKED must not declare public reconciliation obligations"
                )
            if self.verified_publication is not None:
                raise ValueError(
                    "BLOCKED needs unknown/unsafe impact and no completion receipt"
                )
            return self

        if self.disposition is ReconciliationDisposition.REQUIRED:
            if self.impact is not ProjectImpact.PUBLIC_CHANGE_REQUIRED:
                raise ValueError("REQUIRED needs public-change-required impact")
            if (
                self.public_contract_sha256 is None
                or self.publication_tree_sha256 is None
                or not self.obligation_ids
            ):
                raise ValueError(
                    "REQUIRED needs an approved contract/tree, non-empty obligation set, and no completion receipt"
                )
            if self.verified_publication is not None:
                raise ValueError(
                    "REQUIRED needs an approved contract/tree, non-empty obligation set, and no completion receipt"
                )
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
                or not self.obligation_ids
                or receipt.obligation_ids != self.obligation_ids
                or receipt.git.publication_tree_sha256 != self.publication_tree_sha256
                or not receipt.git.readback_verified
                or not receipt.git.fast_forward_only
                or receipt.pull_request.merged_commit_sha
                != receipt.git.resulting_commit
                or receipt.ci.commit_sha != receipt.git.resulting_commit
            ):
                raise ValueError(
                    "COMPLETE requires exact verified Git/readback/CI evidence"
                )
            return self

        raise ValueError(f"unsupported disposition: {self.disposition}")


class ReconciliationStateError(RuntimeError):
    """Raised when durable reconciliation state persistence, readback, or verification fails."""


class DurableReconciliationStateRecord(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = (
        "llm-evaluation.durable-reconciliation-state"
    )
    canonical_identity_version: ClassVar[int] = 1
    record_type: Literal["llm_evaluation_durable_reconciliation_state"] = (
        "llm_evaluation_durable_reconciliation_state"
    )
    schema_version: Literal[2] = 2
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
    schema_version: Literal[4] = 4
    dispositions: tuple[Literal["NONE", "REQUIRED", "BLOCKED", "COMPLETE"], ...]
    complete_requires_snapshot_match: bool
    complete_requires_contract_match: bool
    complete_requires_readback: bool
    complete_requires_authoritative_github_evidence: bool
    complete_requires_merged_pull_request: bool
    complete_requires_post_merge_main_push_ci: bool
    complete_requires_exact_workflow_identity: bool
    complete_requires_exact_required_job_identity: bool
    complete_requires_exact_publication_tree: bool
    complete_requires_obligation_set_match: bool
    durable_advance_requires_complete_obligation_set: bool


class ReviewedCanonicalImpact(_StrictFrozenProjectRecord):
    impact: ProjectImpact
    public_contract: ReconciliationPolicyContract | None = None
    publication_tree: PublicationTree | None = None
    obligation_ids: tuple[str, ...] = ()

    @field_validator("obligation_ids")
    @classmethod
    def validate_obligation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            return value
        return _normalize_obligation_ids(value)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.impact is ProjectImpact.PUBLIC_CHANGE_REQUIRED:
            if self.public_contract is None:
                raise ValueError(
                    "public-change-required review must include a public contract"
                )
            if self.publication_tree is None:
                raise ValueError(
                    "public-change-required review must include an exact approved publication tree"
                )
            if not self.obligation_ids:
                raise ValueError(
                    "public-change-required review must include a non-empty obligation set"
                )
            _validate_contract_in_publication_tree(
                self.public_contract, self.publication_tree
            )
        elif (
            self.public_contract is not None
            or self.publication_tree is not None
            or self.obligation_ids
        ):
            raise ValueError(
                "only a public-change-required review may include a public contract, publication tree, or obligations"
            )
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
        complete_requires_exact_publication_tree=True,
        complete_requires_obligation_set_match=True,
        durable_advance_requires_complete_obligation_set=True,
    )


def approved_policy_tree(contract: ReconciliationPolicyContract) -> PublicationTree:
    if not isinstance(contract, ReconciliationPolicyContract):
        raise TypeError(
            "approved policy publication requires a ReconciliationPolicyContract"
        )
    return PublicationTree(
        files=(
            PublicationFile(
                path="fixtures/reconciliation_policy_contract.json",
                data=canonical_record_bytes(contract),
            ),
        )
    )


def verify_executable_policy_contract(
    authoritative: ReconciliationPolicyContract,
) -> None:
    if authoritative != current_executable_policy_contract():
        raise ValueError(
            "executable reconciliation policy is stale relative to reviewed authority"
        )


def classify_reconciliation(
    *,
    current_snapshot_sha256: str,
    previous: LastReconciledStateRecord | None,
    impact: ProjectImpact,
    public_contract_sha256: str | None = None,
    publication_tree_sha256: str | None = None,
    obligation_ids: tuple[str, ...] = (),
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
        if (
            public_contract_sha256 is None
            or publication_tree_sha256 is None
            or not obligation_ids
        ):
            raise ValueError(
                "public change requires approved contract/publication-tree identities and obligations"
            )
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
        obligation_ids=obligation_ids
        if disposition is ReconciliationDisposition.REQUIRED
        else (),
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
            obligation_ids=required.obligation_ids,
            verified_publication=receipt,
        )
    except ValueError as exc:
        raise ValueError(
            "reconciliation cannot complete without exact verified evidence"
        ) from exc


def reconcile_reviewed_snapshot(
    *,
    current_snapshot_sha256: str,
    previous: LastReconciledStateRecord | None,
    reviewed: ReviewedCanonicalImpact,
) -> ProjectReconciliationResult:
    contract = reviewed.public_contract
    tree = reviewed.publication_tree
    contract_hash = canonical_record_hash(contract) if contract is not None else None
    tree_hash = tree.sha256 if tree is not None else None
    disposition = classify_reconciliation(
        current_snapshot_sha256=current_snapshot_sha256,
        previous=previous,
        impact=reviewed.impact,
        public_contract_sha256=contract_hash,
        publication_tree_sha256=tree_hash,
        obligation_ids=reviewed.obligation_ids,
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
        untrusted = json.loads(raw_bytes)
    except Exception as exc:
        raise ReconciliationStateError(
            f"failed to parse durable reconciliation state from '{target}': {exc}"
        ) from exc
    if (
        isinstance(untrusted, dict)
        and untrusted.get("record_type")
        == "llm_evaluation_durable_reconciliation_state"
        and untrusted.get("schema_version") == 1
    ):
        raise ReconciliationStateError(
            "legacy durable reconciliation state v1 requires explicit supersession recovery; "
            "refusing to treat it as current state"
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


def supersede_legacy_reconciliation_state(
    *,
    expected_raw_sha256: str,
    path: Path | str | None = None,
) -> Path:
    if _SHA256_RE.fullmatch(expected_raw_sha256) is None:
        raise ValueError("expected_raw_sha256 must be a lowercase SHA-256")
    target = resolve_reconciliation_state_path(path)
    if target.is_symlink():
        raise ReconciliationStateError(
            f"legacy reconciliation state path must not be a symlink: {target}"
        )
    if not target.exists() or not target.is_file():
        raise ReconciliationStateError(
            "legacy reconciliation state file is unavailable for supersession"
        )
    parent = target.parent
    if parent.is_symlink():
        raise ReconciliationStateError(
            f"legacy reconciliation state parent directory must not be a symlink: {parent}"
        )
    try:
        raw_bytes = target.read_bytes()
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to read legacy reconciliation state for supersession"
        ) from exc
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha256 != expected_raw_sha256:
        raise ReconciliationStateError(
            f"legacy reconciliation state hash mismatch ({raw_sha256} != {expected_raw_sha256})"
        )
    try:
        untrusted = json.loads(raw_bytes)
    except Exception as exc:
        raise ReconciliationStateError(
            "legacy reconciliation state is not valid JSON"
        ) from exc
    if not (
        isinstance(untrusted, dict)
        and untrusted.get("record_type")
        == "llm_evaluation_durable_reconciliation_state"
        and untrusted.get("schema_version") == 1
    ):
        raise ReconciliationStateError(
            "supersession is restricted to legacy durable reconciliation state schema v1"
        )

    history_dir = parent / "reconciliation-history"
    if history_dir.is_symlink():
        raise ReconciliationStateError(
            "reconciliation history directory must not be a symlink"
        )
    try:
        history_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        try:
            os.chmod(history_dir, 0o700)
        except OSError:
            pass
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to create reconciliation history directory"
        ) from exc
    archive = history_dir / f"superseded-v1-{raw_sha256}.json"
    if archive.is_symlink():
        raise ReconciliationStateError(
            "legacy reconciliation history target must not be a symlink"
        )
    if archive.exists():
        try:
            if archive.read_bytes() != raw_bytes:
                raise ReconciliationStateError(
                    "existing reconciliation history file does not match the legacy state bytes"
                )
        except OSError as exc:
            raise ReconciliationStateError(
                "failed to verify existing reconciliation history file"
            ) from exc
    else:
        try:
            fd = os.open(str(archive), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with open(fd, "wb", closefd=True) as f:
                f.write(raw_bytes)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise ReconciliationStateError(
                "failed to archive legacy reconciliation state"
            ) from exc
    try:
        archived_bytes = archive.read_bytes()
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to read back archived legacy reconciliation state"
        ) from exc
    if (
        archived_bytes != raw_bytes
        or hashlib.sha256(archived_bytes).hexdigest() != raw_sha256
    ):
        raise ReconciliationStateError(
            "legacy reconciliation history readback verification failed"
        )
    try:
        history_fd = os.open(str(history_dir), os.O_RDONLY)
        try:
            os.fsync(history_fd)
        finally:
            os.close(history_fd)
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to durably synchronize reconciliation history before supersession"
        ) from exc

    try:
        current_bytes = target.read_bytes()
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to re-read active legacy reconciliation state"
        ) from exc
    if current_bytes != raw_bytes:
        raise ReconciliationStateError(
            "active legacy reconciliation state changed during supersession; refusing to remove it"
        )
    try:
        target.unlink()
        dir_fd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to retire active legacy reconciliation state"
        ) from exc
    return archive



def supersede_invalid_durable_reconciliation_state(
    *,
    replacement: DurableReconciliationStateRecord,
    expected_active_raw_sha256: str,
    expected_active_canonical_sha256: str,
    expected_replacement_raw_sha256: str,
    expected_replacement_canonical_sha256: str,
    path: Path | str | None = None,
) -> DurableReconciliationStateRecord:
    """Archive a known-invalid current v2 checkpoint and atomically restore verified v2 state."""
    for label, value in (
        ("active raw", expected_active_raw_sha256),
        ("active canonical", expected_active_canonical_sha256),
        ("replacement raw", expected_replacement_raw_sha256),
        ("replacement canonical", expected_replacement_canonical_sha256),
    ):
        if _SHA256_RE.fullmatch(value) is None:
            raise ReconciliationStateError(f"expected {label} identity must be lowercase SHA-256")
    if not isinstance(replacement, DurableReconciliationStateRecord):
        raise TypeError(
            f"replacement must be DurableReconciliationStateRecord, got {type(replacement).__name__}"
        )

    target = resolve_reconciliation_state_path(path)
    if target.is_symlink():
        raise ReconciliationStateError(
            f"durable reconciliation state target must not be a symlink: {target}"
        )
    if not target.exists() or not target.is_file():
        raise ReconciliationStateError(
            "active durable reconciliation state is unavailable for invalid-state supersession"
        )
    parent = target.parent
    if parent.is_symlink():
        raise ReconciliationStateError(
            f"durable reconciliation state parent directory must not be a symlink: {parent}"
        )

    try:
        active_bytes = target.read_bytes()
        active_record = parse_canonical_record(
            active_bytes, DurableReconciliationStateRecord
        )
    except Exception as exc:
        raise ReconciliationStateError(
            "active durable reconciliation state cannot be verified for supersession"
        ) from exc
    active_raw_sha256 = hashlib.sha256(active_bytes).hexdigest()
    active_canonical_sha256 = canonical_record_hash(active_record)
    if (
        active_raw_sha256 != expected_active_raw_sha256
        or active_canonical_sha256 != expected_active_canonical_sha256
    ):
        raise ReconciliationStateError(
            "active durable reconciliation state hash/identity mismatch"
        )

    replacement_bytes = canonical_record_bytes(replacement)
    replacement_raw_sha256 = hashlib.sha256(replacement_bytes).hexdigest()
    replacement_canonical_sha256 = canonical_record_hash(replacement)
    if (
        replacement_raw_sha256 != expected_replacement_raw_sha256
        or replacement_canonical_sha256 != expected_replacement_canonical_sha256
    ):
        raise ReconciliationStateError(
            "replacement durable reconciliation state hash/identity mismatch"
        )

    history_dir = parent / "reconciliation-history"
    if history_dir.is_symlink():
        raise ReconciliationStateError(
            "reconciliation history directory must not be a symlink"
        )
    try:
        history_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        try:
            os.chmod(history_dir, 0o700)
        except OSError:
            pass
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to create reconciliation history directory"
        ) from exc

    archive = history_dir / f"superseded-invalid-v2-{active_raw_sha256}.json"
    if archive.is_symlink():
        raise ReconciliationStateError(
            "invalid reconciliation history target must not be a symlink"
        )
    if archive.exists():
        try:
            existing = archive.read_bytes()
        except OSError as exc:
            raise ReconciliationStateError(
                "failed to verify existing invalid-state history record"
            ) from exc
        if existing != active_bytes:
            raise ReconciliationStateError(
                "existing invalid-state history record does not match active state bytes"
            )
    else:
        try:
            fd = os.open(str(archive), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with open(fd, "wb", closefd=True) as f:
                f.write(active_bytes)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise ReconciliationStateError(
                "failed to archive invalid durable reconciliation state"
            ) from exc
    try:
        archived_bytes = archive.read_bytes()
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to read back archived invalid durable reconciliation state"
        ) from exc
    if archived_bytes != active_bytes or hashlib.sha256(archived_bytes).hexdigest() != active_raw_sha256:
        raise ReconciliationStateError(
            "invalid durable reconciliation history readback verification failed"
        )
    try:
        history_fd = os.open(str(history_dir), os.O_RDONLY)
        try:
            os.fsync(history_fd)
        finally:
            os.close(history_fd)
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to durably synchronize reconciliation history before recovery"
        ) from exc

    temp_path = parent / f".last-reconciled-state.recovery.{os.getpid()}.{time.time_ns()}.json"
    try:
        fd = os.open(str(temp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(fd, "wb", closefd=True) as f:
            f.write(replacement_bytes)
            f.flush()
            os.fsync(f.fileno())
        # Re-read immediately before replacement so a moved active checkpoint fails closed.
        if target.read_bytes() != active_bytes:
            raise ReconciliationStateError(
                "active durable reconciliation state changed during recovery"
            )
        os.replace(str(temp_path), str(target))
        dir_fd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        if isinstance(exc, ReconciliationStateError):
            raise
        raise ReconciliationStateError(
            "failed to atomically restore verified durable reconciliation state"
        ) from exc

    try:
        readback_bytes = target.read_bytes()
        readback = parse_canonical_record(
            readback_bytes, DurableReconciliationStateRecord
        )
    except Exception as exc:
        raise ReconciliationStateError(
            "recovered durable reconciliation state readback failed"
        ) from exc
    if (
        readback_bytes != replacement_bytes
        or hashlib.sha256(readback_bytes).hexdigest() != expected_replacement_raw_sha256
        or canonical_record_hash(readback) != expected_replacement_canonical_sha256
    ):
        raise ReconciliationStateError(
            "recovered durable reconciliation state readback identity mismatch"
        )
    return readback

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
    if target.exists():
        if not target.is_file():
            raise ReconciliationStateError(
                f"durable reconciliation state target must be a regular file: {target}"
            )
        try:
            existing_bytes = target.read_bytes()
            existing_untrusted = json.loads(existing_bytes)
        except (OSError, ValueError, TypeError) as exc:
            raise ReconciliationStateError(
                "existing durable reconciliation state cannot be safely classified before replacement"
            ) from exc
        if (
            isinstance(existing_untrusted, dict)
            and existing_untrusted.get("record_type")
            == "llm_evaluation_durable_reconciliation_state"
            and existing_untrusted.get("schema_version") == 1
        ):
            raise ReconciliationStateError(
                "legacy durable reconciliation state v1 requires explicit supersession before v2 persistence"
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
    temp_path = (
        parent / f".last-reconciled-state.tmp.{os.getpid()}.{time.time_ns()}.json"
    )
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
        readback_record = parse_canonical_record(
            readback_bytes, DurableReconciliationStateRecord
        )
    except Exception as exc:
        raise ReconciliationStateError(
            f"readback verification failed to parse record from '{target}': {exc}"
        ) from exc
    if canonical_record_hash(readback_record) != canonical_record_hash(record):
        raise ReconciliationStateError(
            f"readback verification failed: hash mismatch ({canonical_record_hash(readback_record)} != {canonical_record_hash(record)})"
        )
    return target


def persist_partial_reconciliation_receipt(
    completed: ReconciliationDispositionRecord,
    *,
    reconciliation_state_path: Path | str | None = None,
) -> Path:
    """Archive verified subset completion without advancing the global snapshot baseline."""
    if completed.disposition is not ReconciliationDisposition.COMPLETE:
        raise ValueError(
            f"only a COMPLETE disposition can be archived as a partial reconciliation receipt, got {completed.disposition.value}"
        )
    if completed.verified_publication is None:
        raise ValueError("partial completion requires a verified_publication receipt")

    state_target = resolve_reconciliation_state_path(reconciliation_state_path)
    parent = state_target.parent
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
            "failed to create reconciliation state parent for partial receipt"
        ) from exc

    history_dir = parent / "reconciliation-history"
    if history_dir.is_symlink():
        raise ReconciliationStateError(
            "reconciliation history directory must not be a symlink"
        )
    try:
        history_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        try:
            os.chmod(history_dir, 0o700)
        except OSError:
            pass
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to create reconciliation history directory"
        ) from exc

    payload = canonical_record_bytes(completed)
    receipt_hash = canonical_record_hash(completed)
    receipt_path = history_dir / (
        f"partial-complete-{completed.current_snapshot_sha256}-{receipt_hash}.json"
    )
    if receipt_path.is_symlink():
        raise ReconciliationStateError(
            "partial reconciliation history target must not be a symlink"
        )
    if receipt_path.exists():
        try:
            existing = receipt_path.read_bytes()
        except OSError as exc:
            raise ReconciliationStateError(
                "failed to verify existing partial reconciliation receipt"
            ) from exc
        if existing != payload:
            raise ReconciliationStateError(
                "existing partial reconciliation receipt does not match expected canonical bytes"
            )
        return receipt_path

    try:
        fd = os.open(str(receipt_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(fd, "wb", closefd=True) as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to persist partial reconciliation receipt"
        ) from exc

    try:
        readback = receipt_path.read_bytes()
        parsed = parse_canonical_record(readback, ReconciliationDispositionRecord)
    except Exception as exc:
        raise ReconciliationStateError(
            "partial reconciliation receipt readback verification failed"
        ) from exc
    if readback != payload or canonical_record_hash(parsed) != receipt_hash:
        raise ReconciliationStateError(
            "partial reconciliation receipt readback verification failed"
        )
    try:
        history_fd = os.open(str(history_dir), os.O_RDONLY)
        try:
            os.fsync(history_fd)
        finally:
            os.close(history_fd)
    except OSError as exc:
        raise ReconciliationStateError(
            "failed to durably synchronize partial reconciliation history"
        ) from exc
    return receipt_path


def persist_completed_reconciliation(
    completed: ReconciliationDispositionRecord,
    path: Path | str | None = None,
    *,
    expected_complete_obligation_ids: tuple[str, ...],
) -> DurableReconciliationStateRecord:
    if completed.disposition is not ReconciliationDisposition.COMPLETE:
        raise ValueError(
            f"only a COMPLETE disposition can be durably persisted, got {completed.disposition.value}"
        )
    if completed.verified_publication is None:
        raise ValueError("persisted completion requires a verified_publication receipt")
    expected = _normalize_obligation_ids(expected_complete_obligation_ids)
    if completed.obligation_ids != expected:
        raise ValueError(
            "global durable advance requires completion to cover the complete outstanding obligation set"
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
