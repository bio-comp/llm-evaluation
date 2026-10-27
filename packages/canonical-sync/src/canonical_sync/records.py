from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StrictFrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonempty_trimmed(value: str, *, label: str) -> str:
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{label} must be non-empty, trimmed, and contain no control characters")
    return value


def _safe_logical_path(value: str) -> str:
    _nonempty_trimmed(value, label="logical path")
    if "\\" in value:
        raise ValueError("logical path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("logical path must be a canonical relative POSIX path")
    if any(part in {"", ".", ".."} or part.casefold() == ".git" for part in path.parts):
        raise ValueError("logical path contains an unsafe component")
    return value


def _sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError("value must be a lowercase SHA-256 hex digest")
    return value


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


class AuthorityObjectSpecRecord(StrictFrozenRecord):
    logical_path: str
    provider: str
    object_id: str
    expected_name: str
    expected_mime_type: str
    expected_parent_id: str | None = None
    representation_schema: str

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        return _safe_logical_path(value)

    @field_validator(
        "provider",
        "object_id",
        "expected_name",
        "expected_mime_type",
        "representation_schema",
    )
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        return _nonempty_trimmed(value, label="authority field")

    @field_validator("expected_parent_id")
    @classmethod
    def validate_parent(cls, value: str | None) -> str | None:
        if value is not None:
            return _nonempty_trimmed(value, label="parent ID")
        return value


class ExactMembershipSpecRecord(StrictFrozenRecord):
    parent_id: str
    expected_child_ids: tuple[str, ...]

    @field_validator("parent_id")
    @classmethod
    def validate_parent(cls, value: str) -> str:
        return _nonempty_trimmed(value, label="membership parent ID")

    @field_validator("expected_child_ids")
    @classmethod
    def validate_children(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("exact membership requires at least one child")
        normalized = tuple(sorted(_nonempty_trimmed(item, label="child ID") for item in value))
        if len(normalized) != len(set(normalized)):
            raise ValueError("exact membership child IDs must be unique")
        return normalized


class AuthorityProfileRecord(StrictFrozenRecord):
    record_type: Literal["authority_profile"] = "authority_profile"
    schema_version: Literal[1] = 1
    profile_name: str
    objects: tuple[AuthorityObjectSpecRecord, ...]
    memberships: tuple[ExactMembershipSpecRecord, ...] = ()

    @field_validator("profile_name")
    @classmethod
    def validate_profile_name(cls, value: str) -> str:
        return _nonempty_trimmed(value, label="profile name")

    @field_validator("objects")
    @classmethod
    def normalize_objects(
        cls, value: tuple[AuthorityObjectSpecRecord, ...]
    ) -> tuple[AuthorityObjectSpecRecord, ...]:
        if not value:
            raise ValueError("authority profile requires at least one object")
        ordered = tuple(sorted(value, key=lambda item: item.logical_path))
        folded_paths = tuple(item.logical_path.casefold() for item in ordered)
        if len(folded_paths) != len(set(folded_paths)):
            raise ValueError("authority logical paths collide portably")
        object_ids = tuple(item.object_id for item in ordered)
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("authority object IDs must be unique")
        return ordered

    @field_validator("memberships")
    @classmethod
    def normalize_memberships(
        cls, value: tuple[ExactMembershipSpecRecord, ...]
    ) -> tuple[ExactMembershipSpecRecord, ...]:
        ordered = tuple(sorted(value, key=lambda item: item.parent_id))
        parents = tuple(item.parent_id for item in ordered)
        if len(parents) != len(set(parents)):
            raise ValueError("membership parents must be unique")
        return ordered


class SemanticEntryRecord(StrictFrozenRecord):
    logical_path: str
    representation_schema: str
    content_sha256: str

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        return _safe_logical_path(value)

    @field_validator("representation_schema")
    @classmethod
    def validate_representation(cls, value: str) -> str:
        return _nonempty_trimmed(value, label="representation schema")

    @field_validator("content_sha256")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _sha256(value)


class SnapshotStateRecord(StrictFrozenRecord):
    record_type: Literal["snapshot_state"] = "snapshot_state"
    schema_version: Literal[1] = 1
    authority_profile_sha256: str
    entries: tuple[SemanticEntryRecord, ...]

    @field_validator("authority_profile_sha256")
    @classmethod
    def validate_profile_hash(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("entries")
    @classmethod
    def normalize_entries(
        cls, value: tuple[SemanticEntryRecord, ...]
    ) -> tuple[SemanticEntryRecord, ...]:
        if not value:
            raise ValueError("snapshot state requires at least one semantic entry")
        ordered = tuple(sorted(value, key=lambda item: item.logical_path))
        folded = tuple(item.logical_path.casefold() for item in ordered)
        if len(folded) != len(set(folded)):
            raise ValueError("snapshot logical paths collide portably")
        return ordered


class SourceObservationRecord(StrictFrozenRecord):
    logical_path: str
    object_id: str
    name: str
    mime_type: str
    version: int
    modified_time: datetime
    parent_ids: tuple[str, ...]
    trashed: bool
    can_download: bool
    representation_schema: str
    content_sha256: str

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        return _safe_logical_path(value)

    @field_validator("object_id", "name", "mime_type", "representation_schema")
    @classmethod
    def validate_strings(cls, value: str) -> str:
        return _nonempty_trimmed(value, label="source observation field")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if value < 0:
            raise ValueError("source version must be nonnegative")
        return value

    @field_validator("modified_time")
    @classmethod
    def validate_modified_time(cls, value: datetime) -> datetime:
        return _aware(value)

    @field_validator("parent_ids")
    @classmethod
    def validate_parents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(_nonempty_trimmed(item, label="parent ID") for item in value))
        if len(normalized) != len(set(normalized)):
            raise ValueError("parent IDs must be unique")
        return normalized

    @field_validator("content_sha256")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _sha256(value)


class SnapshotObservationRecord(StrictFrozenRecord):
    record_type: Literal["snapshot_observation"] = "snapshot_observation"
    schema_version: Literal[1] = 1
    snapshot_state_sha256: str
    authority_profile_sha256: str
    acquired_at: datetime
    attempt_count: int
    sources: tuple[SourceObservationRecord, ...]

    @field_validator("snapshot_state_sha256", "authority_profile_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("acquired_at")
    @classmethod
    def validate_acquired_at(cls, value: datetime) -> datetime:
        return _aware(value)

    @field_validator("attempt_count")
    @classmethod
    def validate_attempt_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("attempt count must be positive")
        return value

    @field_validator("sources")
    @classmethod
    def normalize_sources(
        cls, value: tuple[SourceObservationRecord, ...]
    ) -> tuple[SourceObservationRecord, ...]:
        if not value:
            raise ValueError("snapshot observation requires at least one source")
        ordered = tuple(sorted(value, key=lambda item: item.logical_path))
        paths = tuple(item.logical_path.casefold() for item in ordered)
        if len(paths) != len(set(paths)):
            raise ValueError("source observations must have unique logical paths")
        return ordered

    @model_validator(mode="after")
    def validate_source_hashes(self) -> Self:
        for source in self.sources:
            _sha256(source.content_sha256)
        return self
