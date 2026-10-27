from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from canonical_sync.domain import SourceContent, SourceObservation, profile_from_record
from canonical_sync.errors import AuthorityMismatchError
from canonical_sync.records import (
    AcquisitionReceiptRecord,
    AuthorityObjectSpecRecord,
    AuthorityProfileRecord,
)
from canonical_sync.store import SnapshotStore
from canonical_sync.workflow import acquire_and_publish
from fakes import FakeTransport

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
MIME = "application/vnd.google-apps.document"


def _profile():
    return profile_from_record(
        AuthorityProfileRecord(
            profile_name="workflow-test",
            objects=(
                AuthorityObjectSpecRecord(
                    logical_path="README.md",
                    provider="google_drive",
                    object_id="object-001",
                    expected_name="README.md",
                    expected_mime_type=MIME,
                    expected_parent_id="parent-001",
                    representation_schema="test-representation-v1",
                ),
            ),
        )
    )


def _observation(*, name: str = "README.md") -> SourceObservation:
    return SourceObservation(
        object_id="object-001",
        name=name,
        mime_type=MIME,
        version=1,
        modified_time=T0,
        parent_ids=("parent-001",),
        trashed=False,
        can_download=True,
    )


def _transport(*, first_name: str = "README.md") -> FakeTransport:
    first = _observation(name=first_name)
    stable = _observation()
    return FakeTransport(
        observations={"object-001": [first, stable]},
        contents={
            "object-001": SourceContent(
                representation_schema="test-representation-v1",
                data=b"README\n",
            )
        },
    )


def _clock(*values: datetime):
    iterator = iter(values)
    return lambda: next(iterator)


def test_successful_workflow_persists_and_verifies_acquisition_receipt(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "private-store")
    result = acquire_and_publish(
        _transport(),
        _profile(),
        store,
        clock=_clock(T0, T0 + timedelta(seconds=1), T0 + timedelta(seconds=2)),
    )

    assert result.receipt.status == "success"
    assert result.receipt.transaction_attempt_count == 1
    assert result.receipt.snapshot_state_sha256 == result.publication.state_sha256
    assert result.receipt.snapshot_observation_sha256 == result.publication.observation_sha256
    assert store.verify_receipt(result.receipt_sha256) == result.receipt
    assert store.list_receipt_hashes() == (result.receipt_sha256,)


def test_failed_workflow_persists_sanitized_failure_receipt_then_reraises(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "private-store")
    with pytest.raises(AuthorityMismatchError):
        acquire_and_publish(
            _transport(first_name="WRONG.md"),
            _profile(),
            store,
            clock=_clock(T0, T0 + timedelta(seconds=1)),
        )

    hashes = store.list_receipt_hashes()
    assert len(hashes) == 1
    receipt = store.verify_receipt(hashes[0])
    assert receipt.status == "failed"
    assert receipt.transaction_attempt_count == 1
    assert receipt.error_code == "AuthorityMismatchError"
    assert receipt.snapshot_state_sha256 is None
    assert receipt.snapshot_observation_sha256 is None


def test_acquisition_receipt_status_invariants_are_strict() -> None:
    with pytest.raises(ValidationError):
        AcquisitionReceiptRecord(
            started_at=T0,
            ended_at=T0,
            authority_profile_sha256="1" * 64,
            transaction_attempt_count=1,
            status="success",
            error_code=None,
            snapshot_state_sha256=None,
            snapshot_observation_sha256=None,
        )

    with pytest.raises(ValidationError):
        AcquisitionReceiptRecord(
            started_at=T0,
            ended_at=T0 - timedelta(seconds=1),
            authority_profile_sha256="1" * 64,
            transaction_attempt_count=0,
            status="failed",
            error_code="AuthorityMismatchError",
            snapshot_state_sha256=None,
            snapshot_observation_sha256=None,
        )
