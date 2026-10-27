from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ILLEGAL_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_STEM_RE = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE
)


class StrictFrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _nonempty_trimmed(value: str, *, label: str) -> str:
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{label} must be non-empty, trimmed, and contain no control characters")
    return value


def _portable_path_key(value: str) -> str:
    _nonempty_trimmed(value, label="logical path")
    if "\\" in value:
        raise ValueError("logical path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("logical path must be a canonical relative POSIX path")
    normalized: list[str] = []
    for part in path.parts:
        if part in {"", ".", ".."} or part.casefold() == ".git":
            raise ValueError("logical path contains an unsafe component")
        if any(character in _WINDOWS_ILLEGAL_CHARS for character in part):
            raise ValueError("logical path contains a portable-illegal character")
        stem = part.split(".", 1)[0]
        if _WINDOWS_RESERVED_STEM_RE.fullmatch(stem):
            raise ValueError("logical path uses a reserved portable filename")
        if part.endswith((" ", ".")):
            raise ValueError("logical path component has an unsafe trailing character")
        if any(ord(character) <= 0x1F or ord(character) == 0x7F for character in part):
            raise ValueError("logical path contains an ASCII control character")
        normalized.append(unicodedata.normalize("NFC", part).casefold())
    return "/".join(normalized)


def _safe_logical_path(value: str) -> str:
    _portable_path_key(value)
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
        folded_paths = tuple(_portable_path_key(item.logical_path) for item in ordered)
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

    @model_validator(mode="after")
    def validate_membership_coherence(self) -> Self:
        by_id = {item.object_id: item for item in self.objects}
        for membership in self.memberships:
            for child_id in membership.expected_child_ids:
                child = by_id.get(child_id)
                if child is None:
                    raise ValueError("exact membership references an undeclared authority object")
                if child.expected_parent_id != membership.parent_id:
                    raise ValueError("exact membership parent conflicts with object authority")
        return self


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
        folded = tuple(_portable_path_key(item.logical_path) for item in ordered)
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
        paths = tuple(_portable_path_key(item.logical_path) for item in ordered)
        if len(paths) != len(set(paths)):
            raise ValueError("source observations must have unique logical paths")
        return ordered

    @model_validator(mode="after")
    def validate_source_hashes(self) -> Self:
        for source in self.sources:
            _sha256(source.content_sha256)
        return self


class AcquisitionReceiptRecord(StrictFrozenRecord):
    record_type: Literal["acquisition_receipt"] = "acquisition_receipt"
    schema_version: Literal[1] = 1
    started_at: datetime
    ended_at: datetime
    authority_profile_sha256: str
    transaction_attempt_count: int
    status: Literal["success", "failed"]
    error_code: str | None = None
    snapshot_state_sha256: str | None = None
    snapshot_observation_sha256: str | None = None

    @field_validator("started_at", "ended_at")
    @classmethod
    def validate_times(cls, value: datetime) -> datetime:
        return _aware(value)

    @field_validator("authority_profile_sha256")
    @classmethod
    def validate_authority_hash(cls, value: str) -> str:
        return _sha256(value)

    @field_validator("snapshot_state_sha256", "snapshot_observation_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        if value is not None:
            return _sha256(value)
        return value

    @field_validator("transaction_attempt_count")
    @classmethod
    def validate_attempt_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("transaction attempt count must be nonnegative")
        return value

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None) -> str | None:
        if value is not None:
            return _nonempty_trimmed(value, label="error code")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("acquisition receipt ended_at precedes started_at")
        if self.status == "success":
            if self.transaction_attempt_count < 1:
                raise ValueError("successful acquisition requires at least one transaction attempt")
            if self.error_code is not None:
                raise ValueError("successful acquisition cannot contain an error code")
            if self.snapshot_state_sha256 is None or self.snapshot_observation_sha256 is None:
                raise ValueError("successful acquisition requires state and observation identities")
        else:
            if self.error_code is None:
                raise ValueError("failed acquisition requires a structured error code")
            if self.snapshot_state_sha256 is not None or self.snapshot_observation_sha256 is not None:
                raise ValueError("failed acquisition cannot claim successful snapshot identities")
        return self


class GitPublicationReceiptRecord(StrictFrozenRecord):
    record_type: Literal["git_publication_receipt"] = "git_publication_receipt"
    schema_version: Literal[1] = 1
    status: Literal["published", "unchanged"]
    remote_ref: str
    expected_base_commit: str | None = None
    resulting_commit: str
    git_tree_oid: str
    publication_tree_sha256: str
    verified_remote_commit: str
    readback_verified: bool
    fast_forward_only: bool

    @field_validator("remote_ref")
    @classmethod
    def validate_remote_ref(cls, value: str) -> str:
        value = _nonempty_trimmed(value, label="remote ref")
        if not value.startswith("refs/heads/"):
            raise ValueError("publication remote ref must be a branch ref")
        return value

    @field_validator(
        "expected_base_commit",
        "resulting_commit",
        "git_tree_oid",
        "verified_remote_commit",
    )
    @classmethod
    def validate_git_oid(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
            raise ValueError("Git object IDs must be lowercase 40- or 64-character hex")
        return value

    @field_validator("publication_tree_sha256")
    @classmethod
    def validate_tree_hash(cls, value: str) -> str:
        return _sha256(value)

    @model_validator(mode="after")
    def validate_verified_result(self) -> Self:
        if self.resulting_commit != self.verified_remote_commit:
            raise ValueError("publication receipt remote verification does not match result")
        if not self.readback_verified or not self.fast_forward_only:
            raise ValueError("publication receipt requires verified readback and fast-forward-only semantics")
        if self.status == "unchanged" and self.expected_base_commit != self.resulting_commit:
            raise ValueError("unchanged publication must preserve the expected remote base")
        return self
