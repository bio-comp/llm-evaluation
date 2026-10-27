from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .codec import canonical_record_hash
from .records import AuthorityProfileRecord, AuthoritySpecificationRecord


@dataclass(frozen=True, slots=True)
class AuthorityObjectSpec:
    logical_path: str
    provider: str
    object_id: str
    expected_name: str
    expected_mime_type: str
    expected_parent_id: str | None
    representation_schema: str


@dataclass(frozen=True, slots=True)
class ExactMembershipSpec:
    parent_id: str
    expected_child_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuthorityProfile:
    profile_name: str
    spec_sha256: str
    objects: tuple[AuthorityObjectSpec, ...]
    memberships: tuple[ExactMembershipSpec, ...]


@dataclass(frozen=True, slots=True)
class SourceObservation:
    object_id: str
    name: str
    mime_type: str
    version: int
    modified_time: datetime
    parent_ids: tuple[str, ...]
    trashed: bool
    can_download: bool

    @property
    def movement_fingerprint(self) -> tuple[object, ...]:
        return (
            self.object_id,
            self.name,
            self.mime_type,
            self.version,
            self.modified_time,
            self.parent_ids,
            self.trashed,
            self.can_download,
        )


@dataclass(frozen=True, slots=True)
class SourceContent:
    representation_schema: str
    data: bytes


@dataclass(frozen=True, slots=True)
class TransportProvenance:
    transport_name: str
    transport_version: str
    auth_provenance: str


def profile_from_record(record: AuthorityProfileRecord) -> AuthorityProfile:
    specification = AuthoritySpecificationRecord(
        objects=record.objects,
        memberships=record.memberships,
    )
    return AuthorityProfile(
        profile_name=record.profile_name,
        spec_sha256=canonical_record_hash(specification),
        objects=tuple(
            AuthorityObjectSpec(
                logical_path=item.logical_path,
                provider=item.provider,
                object_id=item.object_id,
                expected_name=item.expected_name,
                expected_mime_type=item.expected_mime_type,
                expected_parent_id=item.expected_parent_id,
                representation_schema=item.representation_schema,
            )
            for item in record.objects
        ),
        memberships=tuple(
            ExactMembershipSpec(
                parent_id=item.parent_id,
                expected_child_ids=item.expected_child_ids,
            )
            for item in record.memberships
        ),
    )
