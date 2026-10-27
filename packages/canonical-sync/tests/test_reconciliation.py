from __future__ import annotations

import pytest
from pydantic import ValidationError

from canonical_sync.reconciliation import classify_reconciliation, complete_reconciliation
from canonical_sync.records import (
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReconciliationDispositionRecord,
    VerifiedPublicationReceiptRecord,
)

H1, H2, H3, H4 = (c * 64 for c in "1234")


def previous(value: str = H1) -> LastReconciledStateRecord:
    return LastReconciledStateRecord(snapshot_state_sha256=value)


def receipt(*, snapshot: str = H2, contract: str = H3, readback: bool = True, ci: bool = True):
    return VerifiedPublicationReceiptRecord(
        snapshot_state_sha256=snapshot,
        public_contract_sha256=contract,
        git_commit_sha="a" * 40,
        artifact_path="contracts/reconciliation-policy.json",
        artifact_sha256=H4,
        readback_verified=readback,
        ci_verified=ci,
    )


def test_all_four_dispositions_are_mechanically_distinct() -> None:
    unchanged = classify_reconciliation(
        current_snapshot_sha256=H1,
        previous=previous(H1),
        impact=ProjectImpact.UNKNOWN_OR_UNSAFE,
    )
    no_public = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.NO_PUBLIC_IMPACT,
    )
    required = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
        public_contract_sha256=H3,
    )
    blocked = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.UNKNOWN_OR_UNSAFE,
    )
    complete = complete_reconciliation(required, receipt())

    assert unchanged.disposition is ReconciliationDisposition.NONE
    assert no_public.disposition is ReconciliationDisposition.NONE
    assert required.disposition is ReconciliationDisposition.REQUIRED
    assert blocked.disposition is ReconciliationDisposition.BLOCKED
    assert complete.disposition is ReconciliationDisposition.COMPLETE
    assert len({item.disposition for item in (unchanged, required, blocked, complete)}) == 4


def test_changed_snapshot_is_not_synonymous_with_required() -> None:
    none = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.NO_PUBLIC_IMPACT,
    )
    blocked = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.UNKNOWN_OR_UNSAFE,
    )
    assert none.disposition is ReconciliationDisposition.NONE
    assert blocked.disposition is ReconciliationDisposition.BLOCKED


def test_complete_requires_exact_snapshot_contract_readback_and_ci_binding() -> None:
    required = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
        public_contract_sha256=H3,
    )
    for bad in (
        receipt(snapshot=H1),
        receipt(contract=H4),
        receipt(readback=False),
        receipt(ci=False),
    ):
        with pytest.raises(ValueError, match="complete"):
            complete_reconciliation(required, bad)

    completed = complete_reconciliation(required, receipt())
    assert completed.verified_publication == receipt()


def test_branch_edit_alone_cannot_complete() -> None:
    required = classify_reconciliation(
        current_snapshot_sha256=H2,
        previous=previous(H1),
        impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
        public_contract_sha256=H3,
    )
    branch_only = receipt(readback=False, ci=False)
    with pytest.raises(ValueError, match="complete"):
        complete_reconciliation(required, branch_only)


def test_persisted_state_rejects_invalid_direct_transition() -> None:
    with pytest.raises(ValidationError):
        ReconciliationDispositionRecord(
            current_snapshot_sha256=H2,
            previous_snapshot_sha256=H1,
            impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
            disposition=ReconciliationDisposition.COMPLETE,
            public_contract_sha256=H3,
            verified_publication=None,
        )
