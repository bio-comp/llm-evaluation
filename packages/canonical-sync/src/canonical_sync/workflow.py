from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .acquisition import AcquisitionCandidate, acquire_stable_candidate
from .domain import AuthorityProfile
from .errors import CanonicalSyncError, SourceMovementError
from .records import AcquisitionReceiptRecord
from .store import PublicationResult, SnapshotStore
from .transport import SourceTransport


@dataclass(frozen=True, slots=True)
class AcquisitionRunResult:
    candidate: AcquisitionCandidate
    publication: PublicationResult
    receipt: AcquisitionReceiptRecord
    receipt_sha256: str


def acquire_and_publish(
    transport: SourceTransport,
    profile: AuthorityProfile,
    store: SnapshotStore,
    *,
    max_movement_attempts: int = 3,
    clock: Callable[[], datetime] | None = None,
) -> AcquisitionRunResult:
    actual_clock = clock or (lambda: datetime.now(UTC))
    started_at = actual_clock()
    candidate: AcquisitionCandidate | None = None
    try:
        candidate = acquire_stable_candidate(
            transport,
            profile,
            max_movement_attempts=max_movement_attempts,
            clock=actual_clock,
        )
        publication = store.publish(store.stage(candidate))
    except CanonicalSyncError as exc:
        ended_at = actual_clock()
        if candidate is not None:
            attempts = candidate.attempt_count
        elif isinstance(exc, SourceMovementError):
            attempts = max_movement_attempts
        else:
            attempts = 1
        receipt = AcquisitionReceiptRecord(
            started_at=started_at,
            ended_at=ended_at,
            authority_profile_sha256=profile.spec_sha256,
            transaction_attempt_count=attempts,
            status="failed",
            error_code=type(exc).__name__,
        )
        store.persist_receipt(receipt)
        raise

    receipt = AcquisitionReceiptRecord(
        started_at=started_at,
        ended_at=actual_clock(),
        authority_profile_sha256=profile.spec_sha256,
        transaction_attempt_count=candidate.attempt_count,
        status="success",
        snapshot_state_sha256=publication.state_sha256,
        snapshot_observation_sha256=publication.observation_sha256,
    )
    receipt_hash = store.persist_receipt(receipt)
    return AcquisitionRunResult(
        candidate=candidate,
        publication=publication,
        receipt=receipt,
        receipt_sha256=receipt_hash,
    )
