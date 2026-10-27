from __future__ import annotations

from .records import (
    LastReconciledStateRecord,
    ProjectImpact,
    ReconciliationDisposition,
    ReconciliationDispositionRecord,
    VerifiedPublicationReceiptRecord,
)


def classify_reconciliation(
    *,
    current_snapshot_sha256: str,
    previous: LastReconciledStateRecord | None,
    impact: ProjectImpact,
    public_contract_sha256: str | None = None,
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
        if public_contract_sha256 is None:
            raise ValueError("public change requires a public contract identity")
        disposition = ReconciliationDisposition.REQUIRED
    else:
        disposition = ReconciliationDisposition.BLOCKED
    return ReconciliationDispositionRecord(
        current_snapshot_sha256=current_snapshot_sha256,
        previous_snapshot_sha256=previous_hash,
        impact=impact,
        disposition=disposition,
        public_contract_sha256=public_contract_sha256,
    )


def complete_reconciliation(
    required: ReconciliationDispositionRecord,
    receipt: VerifiedPublicationReceiptRecord,
) -> ReconciliationDispositionRecord:
    if required.disposition is not ReconciliationDisposition.REQUIRED:
        raise ValueError("only a REQUIRED reconciliation can complete")
    try:
        return ReconciliationDispositionRecord(
            current_snapshot_sha256=required.current_snapshot_sha256,
            previous_snapshot_sha256=required.previous_snapshot_sha256,
            impact=required.impact,
            disposition=ReconciliationDisposition.COMPLETE,
            public_contract_sha256=required.public_contract_sha256,
            verified_publication=receipt,
        )
    except ValueError as exc:
        raise ValueError("reconciliation cannot complete without exact verified evidence") from exc
