from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from canonical_sync.codec import canonical_record_hash
from canonical_sync.publication import PublicProjectionRecord
from canonical_sync.reconciliation import classify_reconciliation
from canonical_sync.records import (
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReconciliationDispositionRecord,
)


class ReconciliationPolicyContract(PublicProjectionRecord):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    projection_type: Literal["llm_evaluation_reconciliation_policy"] = (
        "llm_evaluation_reconciliation_policy"
    )
    schema_version: Literal[1] = 1
    dispositions: tuple[Literal["NONE", "REQUIRED", "BLOCKED", "COMPLETE"], ...]
    complete_requires_snapshot_match: bool
    complete_requires_contract_match: bool
    complete_requires_readback: bool
    complete_requires_ci: bool


class ReviewedCanonicalImpact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
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


def current_executable_policy_contract() -> ReconciliationPolicyContract:
    return ReconciliationPolicyContract(
        dispositions=("NONE", "REQUIRED", "BLOCKED", "COMPLETE"),
        complete_requires_snapshot_match=True,
        complete_requires_contract_match=True,
        complete_requires_readback=True,
        complete_requires_ci=True,
    )


def verify_executable_policy_contract(authoritative: ReconciliationPolicyContract) -> None:
    if authoritative != current_executable_policy_contract():
        raise ValueError("executable reconciliation policy is stale relative to reviewed authority")


def reconcile_reviewed_snapshot(
    *,
    current_snapshot_sha256: str,
    previous: LastReconciledStateRecord | None,
    reviewed: ReviewedCanonicalImpact,
) -> ProjectReconciliationResult:
    contract_hash = (
        canonical_record_hash(reviewed.public_contract)
        if reviewed.public_contract is not None
        else None
    )
    disposition = classify_reconciliation(
        current_snapshot_sha256=current_snapshot_sha256,
        previous=previous,
        impact=reviewed.impact,
        public_contract_sha256=contract_hash,
    )
    projection = (
        reviewed.public_contract
        if disposition.disposition is ReconciliationDisposition.REQUIRED
        else None
    )
    return ProjectReconciliationResult(disposition=disposition, public_projection=projection)
