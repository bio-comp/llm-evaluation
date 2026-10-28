from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from canonical_sync.records import GitPublicationReceiptRecord
from llm_evaluation.canonical_reconciliation import (
    GitHubCIVerificationRecord,
    GitHubPullRequestMergeRecord,
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
MERGED = "a" * 40
REVIEWED_HEAD = "c" * 40
PUBLIC_REMOTE = "https://github.com/bio-comp/llm-evaluation.git"


def _git_receipt(
    tree_sha256: str,
    *,
    remote: str = PUBLIC_REMOTE,
    ref: str = "refs/heads/main",
    status: Literal["published", "unchanged"] = "unchanged",
) -> GitPublicationReceiptRecord:
    expected_base = MERGED if status == "unchanged" else None
    return GitPublicationReceiptRecord(
        status=status,
        remote=remote,
        remote_ref=ref,
        expected_base_commit=expected_base,
        resulting_commit=MERGED,
        git_tree_oid="b" * 40,
        publication_tree_sha256=tree_sha256,
        verified_remote_commit=MERGED,
        readback_verified=True,
        fast_forward_only=True,
    )


def _pull_request() -> GitHubPullRequestMergeRecord:
    return GitHubPullRequestMergeRecord(
        pull_request_number=42,
        head_ref="refs/heads/feat/task2-drive-reconciliation",
        head_commit_sha=REVIEWED_HEAD,
        merged_commit_sha=MERGED,
    )


def _ci(*, commit_sha: str = MERGED) -> GitHubCIVerificationRecord:
    return GitHubCIVerificationRecord(run_id=12345, commit_sha=commit_sha, workflow_name="Public CI")


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


def _verified_receipt(
    *,
    snapshot_state_sha256: str = H2,
    public_contract_sha256: str,
    publication_tree_sha256: str,
    git: GitPublicationReceiptRecord | None = None,
    ci: GitHubCIVerificationRecord | None = None,
) -> VerifiedPublicationReceiptRecord:
    return VerifiedPublicationReceiptRecord(
        snapshot_state_sha256=snapshot_state_sha256,
        public_contract_sha256=public_contract_sha256,
        publication_tree_sha256=publication_tree_sha256,
        git=git or _git_receipt(publication_tree_sha256),
        pull_request=_pull_request(),
        ci=ci or _ci(),
    )


def test_complete_binds_project_state_to_merged_git_readback_pr_and_ci() -> None:
    required = _required().disposition
    assert required.public_contract_sha256 is not None
    assert required.publication_tree_sha256 is not None
    receipt = _verified_receipt(
        public_contract_sha256=required.public_contract_sha256,
        publication_tree_sha256=required.publication_tree_sha256,
    )

    completed = complete_reconciliation(required, receipt)
    assert completed.disposition is ReconciliationDisposition.COMPLETE
    assert completed.verified_publication == receipt
    assert receipt.git.resulting_commit == receipt.pull_request.merged_commit_sha
    assert receipt.git.resulting_commit == receipt.ci.commit_sha


def test_complete_rejects_project_binding_mismatch() -> None:
    required = _required().disposition
    assert required.public_contract_sha256 is not None
    assert required.publication_tree_sha256 is not None
    bad_receipts = (
        _verified_receipt(
            snapshot_state_sha256=H1,
            public_contract_sha256=required.public_contract_sha256,
            publication_tree_sha256=required.publication_tree_sha256,
        ),
        _verified_receipt(
            public_contract_sha256="3" * 64,
            publication_tree_sha256=required.publication_tree_sha256,
        ),
    )
    for receipt in bad_receipts:
        with pytest.raises(ValueError, match="complete"):
            complete_reconciliation(required, receipt)


@pytest.mark.parametrize(
    "git",
    [
        pytest.param(
            _git_receipt("2" * 64, ref="refs/heads/arbitrary"),
            id="wrong-ref",
        ),
        pytest.param(
            _git_receipt("2" * 64, remote="https://github.com/example/wrong.git"),
            id="wrong-repository",
        ),
        pytest.param(
            _git_receipt("2" * 64, status="published"),
            id="not-post-merge-readback",
        ),
    ],
)
def test_verified_completion_evidence_rejects_wrong_repository_ref_or_transaction(
    git: GitPublicationReceiptRecord,
) -> None:
    required = _required().disposition
    assert required.public_contract_sha256 is not None
    assert required.publication_tree_sha256 is not None
    with pytest.raises(ValidationError):
        _verified_receipt(
            public_contract_sha256=required.public_contract_sha256,
            publication_tree_sha256=required.publication_tree_sha256,
            git=git,
        )


def test_verified_completion_evidence_rejects_ci_for_different_commit() -> None:
    required = _required().disposition
    assert required.public_contract_sha256 is not None
    assert required.publication_tree_sha256 is not None
    with pytest.raises(ValidationError, match="same merged commit"):
        _verified_receipt(
            public_contract_sha256=required.public_contract_sha256,
            publication_tree_sha256=required.publication_tree_sha256,
            ci=_ci(commit_sha="d" * 40),
        )


def test_free_standing_ci_verified_boolean_is_not_completion_evidence() -> None:
    required = _required().disposition
    assert required.public_contract_sha256 is not None
    assert required.publication_tree_sha256 is not None
    payload = {
        "snapshot_state_sha256": H2,
        "public_contract_sha256": required.public_contract_sha256,
        "publication_tree_sha256": required.publication_tree_sha256,
        "git": _git_receipt(required.publication_tree_sha256),
        "pull_request": _pull_request(),
        "ci_verified": True,
    }
    with pytest.raises(ValidationError):
        VerifiedPublicationReceiptRecord.model_validate(payload, strict=True)


def test_ci_requires_explicit_workflow_identity():
    with pytest.raises(ValidationError):
        GitHubCIVerificationRecord(run_id=1, commit_sha=MERGED)


@pytest.mark.parametrize('fields', [
    {'workflow_name': 'Unrelated'}, {'conclusion': 'failure'},
    {'conclusion': 'cancelled'}, {'conclusion': 'skipped'},
    {'repository': 'example/wrong'}, {'check_name': 'unrelated'},
])
def test_ci_rejects_unrelated_or_unsuccessful_evidence(fields):
    with pytest.raises(ValidationError):
        GitHubCIVerificationRecord.model_validate(
            dict(run_id=1, commit_sha=MERGED, workflow_name='Public CI') | fields)


def test_spoofed_host_cannot_complete():
    required = _required().disposition
    with pytest.raises(ValidationError):
        _verified_receipt(public_contract_sha256=required.public_contract_sha256,
                          publication_tree_sha256=required.publication_tree_sha256,
                          git=_git_receipt(required.publication_tree_sha256,
                              remote='https://github.com.evil.invalid/bio-comp/llm-evaluation.git'))


def test_wrong_base_and_merge_are_rejected():
    with pytest.raises(ValidationError):
        GitHubPullRequestMergeRecord.model_validate(_pull_request().model_dump() |
                                                    {'base_ref': 'refs/heads/other'})
    required = _required().disposition
    receipt = _verified_receipt(public_contract_sha256=required.public_contract_sha256,
                               publication_tree_sha256=required.publication_tree_sha256)
    changed = receipt.model_dump()
    changed['pull_request']['merged_commit_sha'] = 'f' * 40
    with pytest.raises(ValidationError):
        VerifiedPublicationReceiptRecord.model_validate(changed)


def test_durable_parent_versions_reject_historical_nested_contracts():
    from canonical_sync.codec import canonical_record_bytes, parse_canonical_record
    from llm_evaluation.canonical_reconciliation import ReconciliationDispositionRecord
    required = _required().disposition
    receipt = _verified_receipt(public_contract_sha256=required.public_contract_sha256,
                               publication_tree_sha256=required.publication_tree_sha256)
    for record, old_versions in [(receipt.git, (1, 2)), (receipt, (1, 2)),
                                 (receipt.ci, (1,)), (required, (1,))]:
        for old in old_versions:
            raw = canonical_record_bytes(record.model_copy(update={'schema_version': old}))
            with pytest.raises(ValidationError):
                parse_canonical_record(raw, type(record))
    assert ReconciliationDispositionRecord.model_fields['schema_version'].default == 2


def test_complete_disposition_rejects_unchanged_snapshot() -> None:
    from llm_evaluation.canonical_reconciliation import ReconciliationDispositionRecord
    required = _required().disposition
    receipt = _verified_receipt(
        public_contract_sha256=required.public_contract_sha256,
        publication_tree_sha256=required.publication_tree_sha256,
    )
    with pytest.raises(ValidationError, match="requires a changed snapshot"):
        ReconciliationDispositionRecord(
            current_snapshot_sha256=required.current_snapshot_sha256,
            previous_snapshot_sha256=required.current_snapshot_sha256,
            impact=required.impact,
            disposition=ReconciliationDisposition.COMPLETE,
            public_contract_sha256=required.public_contract_sha256,
            publication_tree_sha256=required.publication_tree_sha256,
            verified_publication=receipt,
        )

