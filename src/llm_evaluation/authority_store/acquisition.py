from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from .contracts import (
    AcquisitionCapability,
    AcquisitionReceipt,
    AuthorityMismatchError,
    AuthorityObjectIdentity,
    SemanticState,
    SourceMovementError,
    SourceObservation,
)
from .projection import SemanticProjection, project_native_document


class Transport(Protocol):
    def get_metadata(self, file_id: str) -> dict[str, Any]: ...
    def get_document(self, file_id: str) -> dict[str, Any]: ...


class DriveMetadataWire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    name: str
    mimeType: str
    version: str
    modifiedTime: str
    parents: list[str]
    trashed: bool

    @field_validator("version")
    @classmethod
    def decimal_version(cls, value: str) -> str:
        if not value.isascii() or not value.isdecimal():
            raise ValueError("Drive version must be a decimal int64 string")
        return value


@dataclass(frozen=True)
class AcquisitionResult:
    semantic: SemanticState
    projection: SemanticProjection
    receipt: AcquisitionReceipt


def _observe(
    raw: dict[str, Any], identity: AuthorityObjectIdentity
) -> SourceObservation:
    wire = DriveMetadataWire.model_validate(raw)
    if wire.trashed:
        raise AuthorityMismatchError("registered authority is trashed")
    if wire.id != identity.file_id:
        raise AuthorityMismatchError(
            "Drive file ID does not match registered authority"
        )
    if wire.name != identity.expected_name:
        raise AuthorityMismatchError(
            "Drive file name does not match registered authority"
        )
    if wire.mimeType != identity.expected_mime_type:
        raise AuthorityMismatchError(
            "Drive MIME type does not match registered authority"
        )
    if tuple(wire.parents) != (identity.expected_parent_id,):
        raise AuthorityMismatchError("Drive parent does not match registered authority")
    return SourceObservation(
        file_id=wire.id,
        name=wire.name,
        mime_type=wire.mimeType,
        version=int(wire.version),
        modified_time=wire.modifiedTime,
        parent_ids=tuple(wire.parents),
        capability=identity.capability,
    )


def acquire_native_document(
    transport: Transport,
    identity: AuthorityObjectIdentity,
    *,
    max_attempts: int = 2,
) -> AcquisitionResult:
    if identity.capability is not AcquisitionCapability.READ_ONLY:
        raise ValueError("this acquisition path is read-only only")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(1, max_attempts + 1):
        pre = _observe(transport.get_metadata(identity.file_id), identity)
        projection = project_native_document(
            transport.get_document(identity.file_id), identity
        )
        post = _observe(transport.get_metadata(identity.file_id), identity)
        if pre != post:
            continue
        semantic = SemanticState(
            logical_role=identity.logical_role,
            projection_schema=projection.projection_schema,
            sha256=projection.sha256,
        )
        receipt = AcquisitionReceipt(
            source=post,
            attempt_count=attempt,
            capability=AcquisitionCapability.READ_ONLY,
        )
        return AcquisitionResult(
            semantic=semantic, projection=projection, receipt=receipt
        )
    raise SourceMovementError("source moved during every bounded acquisition attempt")
