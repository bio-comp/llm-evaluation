from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from pydantic import BaseModel, ConfigDict, field_validator

from .codec import canonical_record_bytes
from .errors import PublicationBoundaryError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_PATTERNS = (
    re.compile(r"ya29\.[A-Za-z0-9._-]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |)PRIVATE KEY-----"),
    re.compile(r"\"(?:client_secret|refresh_token|private_key)\"\s*:"),
    re.compile(r"Authorization:\s*Bearer\s+", re.IGNORECASE),
)


class PublicProjectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    projection_type: str
    schema_version: int

    @field_validator("projection_type")
    @classmethod
    def validate_projection_type(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("projection type must be non-empty and trimmed")
        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("projection schema version must be positive")
        return value


class PublicArtifactReceiptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source_snapshot_sha256: str
    projection_type: str
    artifact_path: str
    artifact_sha256: str

    @field_validator("source_snapshot_sha256", "artifact_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("value must be a lowercase SHA-256")
        return value


def _safe_relative_path(value: str) -> Path:
    if not value or "\\" in value:
        raise PublicationBoundaryError("public artifact path is unsafe")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value:
        raise PublicationBoundaryError("public artifact path must be canonical and relative")
    if any(part in {"", ".", ".."} or part.casefold() == ".git" for part in pure.parts):
        raise PublicationBoundaryError("public artifact path contains traversal/reserved components")
    return Path(*pure.parts)


def _reject_symlink(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise PublicationBoundaryError("public publication target is a symlink")
    if not stat.S_ISREG(mode):
        raise PublicationBoundaryError("public publication target is not a regular file")


def _ensure_safe_parent(root: Path, relative: Path) -> Path:
    if not root.exists() or root.is_symlink() or not root.is_dir():
        raise PublicationBoundaryError("public root must be an existing non-symlink directory")
    current = root
    for part in relative.parent.parts:
        current = current / part
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise PublicationBoundaryError("public artifact parent is unsafe or symlinked")
        else:
            current.mkdir()
    return root / relative


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)


def _guard_payload(
    payload: bytes,
    projection: PublicProjectionRecord,
    *,
    private_markers: Iterable[str],
    private_bodies: Iterable[bytes],
) -> None:
    for marker in private_markers:
        if marker and marker.encode("utf-8") in payload:
            raise PublicationBoundaryError("public projection contains a configured private marker")
    for body in private_bodies:
        if body and body in payload:
            raise PublicationBoundaryError("public projection contains acquired private canonical content")
    text = payload.decode("utf-8")
    raw_strings = tuple(_iter_strings(projection.model_dump(mode="python")))
    if any(pattern.search(text) for pattern in _CREDENTIAL_PATTERNS) or any(
        pattern.search(value) for value in raw_strings for pattern in _CREDENTIAL_PATTERNS
    ):
        raise PublicationBoundaryError("public projection contains credential-like material")


def publish_public_projection(
    projection: PublicProjectionRecord,
    *,
    public_root: Path,
    relative_path: str,
    source_snapshot_sha256: str,
    private_markers: Iterable[str] = (),
    private_bodies: Iterable[bytes] = (),
    expected_existing_sha256: str | None = None,
) -> PublicArtifactReceiptRecord:
    if not isinstance(projection, PublicProjectionRecord):
        raise TypeError("public projection must be a PublicProjectionRecord")
    if _SHA256_RE.fullmatch(source_snapshot_sha256) is None:
        raise PublicationBoundaryError("source snapshot identity is malformed")
    if expected_existing_sha256 is not None and _SHA256_RE.fullmatch(expected_existing_sha256) is None:
        raise PublicationBoundaryError("expected existing artifact hash is malformed")
    relative = _safe_relative_path(relative_path)
    target = _ensure_safe_parent(public_root, relative)
    _reject_symlink(target)
    payload = canonical_record_bytes(projection)
    _guard_payload(
        payload, projection, private_markers=private_markers, private_bodies=private_bodies
    )
    artifact_hash = hashlib.sha256(payload).hexdigest()

    if target.exists():
        existing = target.read_bytes()
        existing_hash = hashlib.sha256(existing).hexdigest()
        if existing == payload:
            return PublicArtifactReceiptRecord(
                source_snapshot_sha256=source_snapshot_sha256,
                projection_type=projection.projection_type,
                artifact_path=relative.as_posix(),
                artifact_sha256=artifact_hash,
            )
        if expected_existing_sha256 is None:
            raise PublicationBoundaryError("existing public artifact differs; explicit compare-and-swap hash required")
        if existing_hash != expected_existing_sha256:
            raise PublicationBoundaryError("existing public artifact does not match compare-and-swap hash")
    elif expected_existing_sha256 is not None:
        raise PublicationBoundaryError("expected existing public artifact is missing")

    fd, temp_name = tempfile.mkstemp(prefix=".canonical-sync-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    if target.read_bytes() != payload:
        raise PublicationBoundaryError("public artifact readback did not match projection bytes")
    return PublicArtifactReceiptRecord(
        source_snapshot_sha256=source_snapshot_sha256,
        projection_type=projection.projection_type,
        artifact_path=relative.as_posix(),
        artifact_sha256=artifact_hash,
    )
