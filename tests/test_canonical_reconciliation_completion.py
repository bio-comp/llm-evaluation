from __future__ import annotations

import pytest

from canonical_sync.records import GitPublicationReceiptRecord
from llm_evaluation.canonical_reconciliation import (
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReviewedCanonicalImpact,
    VerifiedPublicationReceiptRecord,
    complete_reconciliation,
    current_executable_policy_contract,
    reconcile_reviewed_snapshot,
)

H1 = "1" * 64
H2 = "2" * 64


def _git_receipt(tree_sha256: str) -> GitPublicationReceiptRecord:
    return GitPublicationReceiptRecord(
        status="published",
        remote_ref="refs/heads/test",
        expected_base_commit=None,
        resulting_commit="a" * 40,
        git_tree_oid="b" * 40,
        publication_tree_sha256=tree_sha256,
        verified_remote_commit="a" * 40,
        readback_verified=True,
        fast_forward_only=True,
    )


def _required():
    result = reconcile_reviewed_snapshot(
        current_snapshot_sha256=H2,
        previous=LastReconciledStateRecord(snapshot_state_sha256=H1),
        reviewed=ReviewedCanonicalImpact(
            impact=ProjectImpact.PUBLIC_CHANGE_REQUIRED,
            public_contract=current_executable_policy_contract(),
        ),
    )
    assert result.publication_tree is not None
    return result


def test_complete_binds_project_state_to_verified_generic_git_receipt() -> None:
    required = _required()
    disposition = required.disposition
    assert disposition.public_contract_sha256 is not None
    assert disposition.publication_tree_sha256 is not None
    receipt = VerifiedPublicationReceiptRecord(
        snapshot_state_sha256=H2,
        public_contract_sha256=disposition.public_contract_sha256,
        publication_tree_sha256=disposition.publication_tree_sha256,
        git=_git_receipt(disposition.publication_tree_sha256),
        ci_verified=True,
    )

    completed = complete_reconciliation(disposition, receipt)
    assert completed.disposition is ReconciliationDisposition.COMPLETE
    assert completed.verified_publication == receipt


def test_complete_rejects_project_binding_mismatch_or_missing_ci() -> None:
    required = _required().disposition
    assert required.public_contract_sha256 is not None
    assert required.publication_tree_sha256 is not None
    good = dict(
        snapshot_state_sha256=H2,
        public_contract_sha256=required.public_contract_sha256,
        publication_tree_sha256=required.publication_tree_sha256,
        git=_git_receipt(required.publication_tree_sha256),
        ci_verified=True,
    )
    bad_receipts = (
        VerifiedPublicationReceiptRecord(**(good | {"snapshot_state_sha256": H1})),
        VerifiedPublicationReceiptRecord(**(good | {"public_contract_sha256": "3" * 64})),
        VerifiedPublicationReceiptRecord(**(good | {"ci_verified": False})),
    )
    for receipt in bad_receipts:
        with pytest.raises(ValueError, match="complete"):
            complete_reconciliation(required, receipt)
