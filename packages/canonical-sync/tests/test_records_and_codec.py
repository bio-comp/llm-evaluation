from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from canonical_sync.codec import canonical_record_bytes, canonical_record_hash, parse_canonical_record
from canonical_sync.domain import profile_from_record
from canonical_sync.records import (
    AuthorityObjectSpecRecord,
    AuthorityProfileRecord,
    SemanticEntryRecord,
    SnapshotObservationRecord,
    SnapshotStateRecord,
    SourceObservationRecord,
)

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def object_spec(*, path: str = "README.md", object_id: str = "opaque-object-001") -> AuthorityObjectSpecRecord:
    return AuthorityObjectSpecRecord(
        logical_path=path,
        provider="google_drive",
        object_id=object_id,
        expected_name=path.rsplit("/", 1)[-1],
        expected_mime_type="application/vnd.google-apps.document",
        expected_parent_id="opaque-parent-001",
        representation_schema="google-docs-visible-text-v1",
    )


def profile() -> AuthorityProfileRecord:
    return AuthorityProfileRecord(
        profile_name="test-profile",
        objects=(object_spec(), object_spec(path="DECISIONS.md", object_id="opaque-object-002")),
    )


def state(*, authority_hash: str = H1, first_content_hash: str = H2) -> SnapshotStateRecord:
    return SnapshotStateRecord(
        authority_profile_sha256=authority_hash,
        entries=(
            SemanticEntryRecord(
                logical_path="README.md",
                representation_schema="google-docs-visible-text-v1",
                content_sha256=first_content_hash,
            ),
            SemanticEntryRecord(
                logical_path="DECISIONS.md",
                representation_schema="google-docs-visible-text-v1",
                content_sha256=H3,
            ),
        ),
    )


def observation(*, snapshot_hash: str, version: int = 7, acquired_at: datetime = T0) -> SnapshotObservationRecord:
    return SnapshotObservationRecord(
        snapshot_state_sha256=snapshot_hash,
        authority_profile_sha256=H1,
        acquired_at=acquired_at,
        attempt_count=1,
        sources=(
            SourceObservationRecord(
                logical_path="README.md",
                object_id="opaque-object-001",
                name="README.md",
                mime_type="application/vnd.google-apps.document",
                version=version,
                modified_time=T0,
                parent_ids=("opaque-parent-001",),
                trashed=False,
                can_download=True,
                representation_schema="google-docs-visible-text-v1",
                content_sha256=H2,
            ),
        ),
    )


def test_strict_boundary_rejects_unknown_fields_and_type_coercion() -> None:
    with pytest.raises(ValidationError):
        AuthorityObjectSpecRecord.model_validate(object_spec().model_dump() | {"unexpected": True})
    with pytest.raises(ValidationError):
        SourceObservationRecord(
            logical_path="README.md",
            object_id="opaque-object-001",
            name="README.md",
            mime_type="application/vnd.google-apps.document",
            version="7",
            modified_time=T0,
            parent_ids=("opaque-parent-001",),
            trashed=False,
            can_download=True,
            representation_schema="google-docs-visible-text-v1",
            content_sha256=H2,
        )


def test_profile_rejects_unsafe_duplicate_and_portably_colliding_paths() -> None:
    for bad in ("", "/absolute.md", "../escape.md", "Research/../escape.md", ".git/config"):
        with pytest.raises(ValidationError):
            object_spec(path=bad)

    with pytest.raises(ValidationError):
        AuthorityProfileRecord(profile_name="dup", objects=(object_spec(), object_spec()))

    with pytest.raises(ValidationError):
        AuthorityProfileRecord(
            profile_name="casefold",
            objects=(
                object_spec(path="Research/Foo.md", object_id="opaque-object-001"),
                object_spec(path="Research/foo.md", object_id="opaque-object-002"),
            ),
        )


def test_profile_converts_to_frozen_slotted_trusted_domain() -> None:
    trusted = profile_from_record(profile())
    assert trusted.profile_name == "test-profile"
    assert trusted.objects[0].logical_path == "DECISIONS.md"
    with pytest.raises((AttributeError, TypeError)):
        trusted.profile_name = "changed"  # type: ignore[misc]


def test_canonical_round_trip_is_byte_identical_and_noncanonical_json_is_rejected() -> None:
    value = state()
    encoded = canonical_record_bytes(value)
    assert parse_canonical_record(encoded, SnapshotStateRecord) == value
    assert canonical_record_bytes(parse_canonical_record(encoded, SnapshotStateRecord)) == encoded

    noncanonical = json.dumps(value.model_dump(mode="json"), indent=2, sort_keys=False).encode()
    assert noncanonical != encoded
    with pytest.raises(ValueError, match="canonical"):
        parse_canonical_record(noncanonical, SnapshotStateRecord)


def test_semantic_state_identity_is_order_independent_but_content_and_authority_sensitive() -> None:
    first = state()
    reordered = SnapshotStateRecord(
        authority_profile_sha256=H1,
        entries=tuple(reversed(first.entries)),
    )
    content_changed = state(first_content_hash="4" * 64)
    authority_changed = state(authority_hash="5" * 64)

    assert first.entries == reordered.entries
    assert canonical_record_hash(first) == canonical_record_hash(reordered)
    assert canonical_record_hash(first) != canonical_record_hash(content_changed)
    assert canonical_record_hash(first) != canonical_record_hash(authority_changed)


def test_observation_identity_changes_for_metadata_only_churn_without_changing_semantic_state() -> None:
    semantic = state()
    semantic_hash = canonical_record_hash(semantic)
    first = observation(snapshot_hash=semantic_hash, version=7, acquired_at=T0)
    second = observation(
        snapshot_hash=semantic_hash,
        version=8,
        acquired_at=T0 + timedelta(minutes=5),
    )

    assert first.snapshot_state_sha256 == second.snapshot_state_sha256 == semantic_hash
    assert canonical_record_hash(first) != canonical_record_hash(second)


def test_hash_fields_are_lowercase_sha256_only() -> None:
    with pytest.raises(ValidationError):
        SemanticEntryRecord(
            logical_path="README.md",
            representation_schema="google-docs-visible-text-v1",
            content_sha256="not-a-hash",
        )
    with pytest.raises(ValidationError):
        state(authority_hash="A" * 64)
