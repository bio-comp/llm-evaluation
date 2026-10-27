from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from canonical_sync.acquisition import acquire_stable_candidate
from canonical_sync.codec import canonical_record_hash
from canonical_sync.domain import SourceContent, SourceObservation, profile_from_record
from canonical_sync.errors import SnapshotIntegrityError
from canonical_sync.records import AuthorityObjectSpecRecord, AuthorityProfileRecord
from canonical_sync.store import SnapshotStore
from fakes import FakeTransport

T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
MIME = "application/vnd.google-apps.document"


def _candidate():
    specs = (
        AuthorityObjectSpecRecord(
            logical_path="README.md", provider="google_drive", object_id="object-001",
            expected_name="README.md", expected_mime_type=MIME, expected_parent_id="parent-001",
            representation_schema="test-representation-v1",
        ),
        AuthorityObjectSpecRecord(
            logical_path="Research/DECISIONS.md", provider="google_drive", object_id="object-002",
            expected_name="DECISIONS.md", expected_mime_type=MIME, expected_parent_id="parent-001",
            representation_schema="test-representation-v1",
        ),
    )
    profile = profile_from_record(AuthorityProfileRecord(profile_name="store-test", objects=specs))
    def obs(object_id: str, name: str) -> SourceObservation:
        return SourceObservation(
            object_id=object_id, name=name, mime_type=MIME, version=1, modified_time=T0,
            parent_ids=("parent-001",), trashed=False, can_download=True,
        )
    transport = FakeTransport(
        observations={
            "object-001": [obs("object-001", "README.md"), obs("object-001", "README.md")],
            "object-002": [obs("object-002", "DECISIONS.md"), obs("object-002", "DECISIONS.md")],
        },
        contents={
            "object-001": SourceContent("test-representation-v1", b"README\n"),
            "object-002": SourceContent("test-representation-v1", b"DECISIONS\n"),
        },
    )
    return acquire_stable_candidate(transport, profile, clock=lambda: T0)


def test_publish_verify_round_trip_and_existing_target_is_verified_noop(tmp_path: Path) -> None:
    candidate = _candidate()
    store = SnapshotStore(tmp_path / "private-store")
    first = store.publish(store.stage(candidate))
    second = store.publish(store.stage(candidate))
    verified = store.verify_completed(first.observation_sha256)

    assert first.published_state is True and first.published_observation is True
    assert second.published_state is False and second.published_observation is False
    assert verified.state == candidate.state
    assert verified.observation == candidate.observation
    assert {item.logical_path: item.data for item in verified.representations} == {
        "README.md": b"README\n",
        "Research/DECISIONS.md": b"DECISIONS\n",
    }


def test_symlinked_store_root_fails_closed(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SnapshotIntegrityError, match="symlink"):
        SnapshotStore(link)


def test_symlinked_operational_directory_fails_before_publication(tmp_path: Path) -> None:
    store_root = tmp_path / "private-store"
    store = SnapshotStore(store_root)
    snapshots = store_root / "snapshots"
    snapshots.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    snapshots.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SnapshotIntegrityError, match="symlink"):
        store.publish(store.stage(_candidate()))
    assert list(outside.iterdir()) == []


def test_tampered_staged_content_fails_before_publication(tmp_path: Path) -> None:
    candidate = _candidate()
    store = SnapshotStore(tmp_path / "private-store")
    staged = store.stage(candidate)
    (staged.path / "snapshot" / "files" / "README.md").write_bytes(b"tampered")
    with pytest.raises(SnapshotIntegrityError, match="hash"):
        store.publish(staged)
    state_hash = canonical_record_hash(candidate.state)
    assert not (store.root / "snapshots" / state_hash).exists()


def test_corrupt_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    candidate = _candidate()
    store = SnapshotStore(tmp_path / "private-store")
    staged = store.stage(candidate)
    target = store.root / "snapshots" / canonical_record_hash(candidate.state)
    target.mkdir()
    (target / "state.json").write_bytes(b"corrupt")
    before = (target / "state.json").read_bytes()
    with pytest.raises(SnapshotIntegrityError):
        store.publish(staged)
    assert (target / "state.json").read_bytes() == before


def test_orphan_semantic_snapshot_is_not_a_completed_acquisition(tmp_path: Path) -> None:
    candidate = _candidate()
    store = SnapshotStore(tmp_path / "private-store")
    result = store.publish(store.stage(candidate))
    observation_dir = store.root / "observations" / result.observation_sha256
    for child in observation_dir.iterdir():
        child.unlink()
    observation_dir.rmdir()
    assert (store.root / "snapshots" / result.state_sha256).is_dir()
    with pytest.raises(SnapshotIntegrityError, match="observation"):
        store.verify_completed(result.observation_sha256)


def test_unexpected_or_symlinked_snapshot_inventory_fails_verification(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "private-store")
    result = store.publish(store.stage(_candidate()))
    snapshot = store.root / "snapshots" / result.state_sha256
    (snapshot / "unexpected.txt").write_text("x")
    with pytest.raises(SnapshotIntegrityError, match="inventory"):
        store.verify_completed(result.observation_sha256)
    (snapshot / "unexpected.txt").unlink()
    target = snapshot / "files" / "README.md"
    target.unlink()
    target.symlink_to(snapshot / "state.json")
    with pytest.raises(SnapshotIntegrityError, match="regular"):
        store.verify_completed(result.observation_sha256)


def test_identical_concurrent_publishers_serialize_to_publish_and_noop(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "private-store")
    candidate = _candidate()
    staged = (store.stage(candidate), store.stage(candidate))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(store.publish, staged))
    assert sum(result.published_state for result in results) == 1
    assert sum(result.published_observation for result in results) == 1
    store.verify_completed(results[0].observation_sha256)
