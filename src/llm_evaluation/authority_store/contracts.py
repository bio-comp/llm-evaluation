from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class AuthorityStoreError(RuntimeError):
    """Base exception for fail-closed authority-store operations."""


class AuthorityMismatchError(AuthorityStoreError):
    """The acquired object is not the exact privately registered authority."""


class SourceMovementError(AuthorityStoreError):
    """The source moved during each bounded acquisition attempt."""


class ClosedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AcquisitionCapability(StrEnum):
    READ_ONLY = "READ_ONLY"
    MUTATION_CAPABLE = "MUTATION_CAPABLE"


class ReconciliationStatus(StrEnum):
    NONE = "NONE"
    REQUIRED = "REQUIRED"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"


class PublicImpact(StrEnum):
    NONE = "NONE"
    REQUIRED = "REQUIRED"
    UNKNOWN = "UNKNOWN"


class AuthorityObjectIdentity(ClosedRecord):
    logical_role: str
    file_id: str
    expected_name: str
    expected_mime_type: str
    expected_parent_id: str
    capability: AcquisitionCapability

    @field_validator(
        "logical_role",
        "file_id",
        "expected_name",
        "expected_mime_type",
        "expected_parent_id",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("identity strings must be non-empty and trimmed")
        return value


class SourceObservation(ClosedRecord):
    file_id: str
    name: str
    mime_type: str
    version: int
    modified_time: str
    parent_ids: tuple[str, ...]
    capability: AcquisitionCapability

    @field_validator("version")
    @classmethod
    def nonnegative_version(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Drive version must be nonnegative")
        return value


class SemanticState(ClosedRecord):
    logical_role: str
    projection_schema: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("semantic_sha256 must be lowercase SHA-256")
        return value


class AcquisitionReceipt(ClosedRecord):
    source: SourceObservation
    attempt_count: int
    capability: AcquisitionCapability

    @field_validator("attempt_count")
    @classmethod
    def positive_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("attempt_count must be positive")
        return value


class LastReconciledState(ClosedRecord):
    semantic_sha256: str
    public_commit: str | None = None

    @field_validator("semantic_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("semantic_sha256 must be lowercase SHA-256")
        return value

    @field_validator("public_commit")
    @classmethod
    def valid_commit(cls, value: str | None) -> str | None:
        if value is not None and not _GIT_OID.fullmatch(value):
            raise ValueError("public_commit must be a lowercase Git object ID")
        return value


class ReconciliationObligation(ClosedRecord):
    status: ReconciliationStatus
    semantic_sha256: str
    previous_semantic_sha256: str | None
    public_impact: PublicImpact

    @field_validator("semantic_sha256", "previous_semantic_sha256")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("reconciliation hashes must be lowercase SHA-256")
        return value


class VerifiedPublicationReceipt(ClosedRecord):
    semantic_sha256: str
    expected_base: str
    resulting_commit: str
    readback_commit: str
    public_payload_sha256: str

    @field_validator("semantic_sha256", "public_payload_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("receipt hashes must be lowercase SHA-256")
        return value

    @field_validator("expected_base", "resulting_commit", "readback_commit")
    @classmethod
    def valid_oid(cls, value: str) -> str:
        if not _GIT_OID.fullmatch(value):
            raise ValueError("receipt commits must be lowercase Git object IDs")
        return value
