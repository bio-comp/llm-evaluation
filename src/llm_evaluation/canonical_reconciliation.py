from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from canonical_sync.codec import canonical_record_bytes, canonical_record_hash
from canonical_sync.git_publication import PublicationFile, PublicationTree
from canonical_sync.records import GitPublicationReceiptRecord

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _StrictFrozenProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


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


class VerifiedPublicationReceiptRecord(_StrictFrozenProjectRecord):
    record_type: Literal["llm_evaluation_verified_publication"] = (
        "llm_evaluation_verified_publication"
    )
    schema_version: Literal[1] = 1
    snapshot_state_sha256: str
    public_contract_sha256: str
    publication_tree_sha256: str
    git: GitPublicationReceiptRecord
    ci_verified: bool

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


class ReconciliationDispositionRecord(_StrictFrozenProjectRecord):
    record_type: Literal["llm_evaluation_reconciliation_disposition"] = (
        "llm_evaluation_reconciliation_disposition"
    )
    schema_version: Literal[1] = 1
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
            if self.verified_publication is not None:
                raise ValueError("NONE cannot contain verified publication evidence")
            if changed and self.impact is not ProjectImpact.NO_PUBLIC_IMPACT:
                raise ValueError("changed state can be NONE only for explicit no-public-impact")
        elif self.disposition is ReconciliationDisposition.REQUIRED:
            if self.impact is not ProjectImpact.PUBLIC_CHANGE_REQUIRED:
                raise ValueError("REQUIRED needs public-change-required impact")
            if (
                self.public_contract_sha256 is None
                or self.publication_tree_sha256 is None
                or self.verified_publication is not None
            ):
                raise ValueError("REQUIRED needs an approved contract/tree and no completion receipt")
        elif self.disposition is ReconciliationDisposition.BLOCKED:
            if self.impact is not ProjectImpact.UNKNOWN_OR_UNSAFE or self.verified_publication is not None:
                raise ValueError("BLOCKED needs unknown/unsafe impact and no completion receipt")
        elif self.disposition is ReconciliationDisposition.COMPLETE:
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
                or not receipt.ci_verified
            ):
                raise ValueError("COMPLETE requires exact verified Git/readback/CI evidence")
        return self


class ReconciliationPolicyContract(_StrictFrozenProjectRecord):
    canonical_identity_domain: ClassVar[str] = "llm-evaluation.reconciliation-policy"
    canonical_identity_version: ClassVar[int] = 1
    projection_type: Literal["llm_evaluation_reconciliation_policy"] = (
        "llm_evaluation_reconciliation_policy"
    )
    schema_version: Literal[1] = 1
    dispositions: tuple[Literal["NONE", "REQUIRED", "BLOCKED", "COMPLETE"], ...]
    complete_requires_snapshot_match: bool
    complete_requires_contract_match: bool
    complete_requires_readback: bool
    complete_requires_ci: bool


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
        complete_requires_ci=True,
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
