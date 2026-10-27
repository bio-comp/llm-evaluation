from __future__ import annotations

import canonical_sync.records as generic_records

from llm_evaluation.canonical_reconciliation import (
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReviewedCanonicalImpact,
    reconcile_reviewed_snapshot,
)

H1 = "1" * 64
H2 = "2" * 64


def test_project_governance_vocabulary_is_not_owned_by_generic_package() -> None:
    for name in (
        "ProjectImpact",
        "ReconciliationDisposition",
        "LastReconciledStateRecord",
        "ReconciliationDispositionRecord",
        "VerifiedPublicationReceiptRecord",
    ):
        assert not hasattr(generic_records, name)


def test_llm_evaluation_still_owns_four_way_reconciliation() -> None:
    unchanged = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H1,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(impact=ProjectImpact.UNKNOWN_OR_UNSAFE),
    )
    private_only = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(impact=ProjectImpact.NO_PUBLIC_IMPACT),
    )
    blocked = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(impact=ProjectImpact.UNKNOWN_OR_UNSAFE),
    )

    assert unchanged.disposition.disposition is ReconciliationDisposition.NONE
    assert private_only.disposition.disposition is ReconciliationDisposition.NONE
    assert blocked.disposition.disposition is ReconciliationDisposition.BLOCKED
