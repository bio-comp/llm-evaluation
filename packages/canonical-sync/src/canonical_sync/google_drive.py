from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from .domain import AuthorityObjectSpec, SourceContent, SourceObservation
from .errors import (
    TransportAuthError,
    TransportNotFoundError,
    TransportPermissionError,
    TransportPermanentError,
    TransportTransientError,
    UnsupportedRepresentationError,
)

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_METADATA_FIELDS = (
    "id,name,mimeType,version,modifiedTime,parents,trashed,capabilities(canDownload)"
)
GOOGLE_DOC_MARKDOWN_REPRESENTATION = "google-drive-export-markdown-v1"
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_MARKDOWN_MIME = "text/markdown"
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class _Request(Protocol):
    def execute(self) -> object: ...


class _FilesResource(Protocol):
    def get(self, **kwargs: object) -> _Request: ...
    def export(self, **kwargs: object) -> _Request: ...
    def list(self, **kwargs: object) -> _Request: ...


class _DriveService(Protocol):
    def files(self) -> _FilesResource: ...


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _CapabilitiesWire(_WireModel):
    canDownload: bool


class _DriveMetadataWire(_WireModel):
    id: str
    name: str
    mimeType: str
    version: str
    modifiedTime: str
    parents: list[str]
    trashed: bool
    capabilities: _CapabilitiesWire


class _DriveListWire(_WireModel):
    files: list[_DriveMetadataWire]
    nextPageToken: str | None = None


def _parse_modified_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TransportPermanentError("Drive metadata contains an invalid modified time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TransportPermanentError("Drive metadata modified time must be timezone-aware")
    return parsed


def _metadata_to_observation(raw: object) -> SourceObservation:
    try:
        wire = _DriveMetadataWire.model_validate(raw, strict=True)
        version = int(wire.version)
    except (ValidationError, ValueError) as exc:
        raise TransportPermanentError("Drive metadata failed strict validation") from exc
    if version < 0 or str(version) != wire.version:
        raise TransportPermanentError("Drive metadata version is not canonical")
    if not wire.parents:
        raise TransportPermanentError("Drive metadata contains no parent identity")
    return SourceObservation(
        object_id=wire.id,
        name=wire.name,
        mime_type=wire.mimeType,
        version=version,
        modified_time=_parse_modified_time(wire.modifiedTime),
        parent_ids=tuple(sorted(wire.parents)),
        trashed=wire.trashed,
        can_download=wire.capabilities.canDownload,
    )


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "resp", None)
    value = getattr(response, "status", None)
    return value if type(value) is int else None


def _http_content(exc: BaseException) -> bytes:
    value = getattr(exc, "content", b"")
    return value if isinstance(value, bytes) else b""


def _rate_limited_403(exc: BaseException) -> bool:
    if _http_status(exc) != 403:
        return False
    body = _http_content(exc)
    return b'"reason":"rateLimitExceeded"' in body or b'"reason":"userRateLimitExceeded"' in body


class GoogleDriveTransport:
    def __init__(
        self,
        *,
        drive_service: _DriveService,
        http_error_types: tuple[type[BaseException], ...],
        request_attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(request_attempts) is not int or request_attempts < 1:
            raise ValueError("request_attempts must be a positive integer")
        self._drive = drive_service
        self._http_error_types = http_error_types
        self._request_attempts = request_attempts
        self._sleeper = sleeper

    @classmethod
    def from_application_default(cls, *, request_attempts: int = 3) -> GoogleDriveTransport:
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError
            from googleapiclient.discovery import build  # type: ignore[import-untyped]
            from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TransportAuthError("Google Drive support is not installed") from exc
        try:
            credentials, _ = google.auth.default(scopes=[DRIVE_READONLY_SCOPE])
            drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        except (DefaultCredentialsError, GoogleAuthError, OSError) as exc:
            raise TransportAuthError("Google application-default read credentials are unavailable") from exc
        return cls(
            drive_service=drive_service,
            http_error_types=(HttpError,),
            request_attempts=request_attempts,
        )

    def _execute(self, factory: Callable[[], _Request]) -> object:
        last_transient: BaseException | None = None
        for attempt in range(self._request_attempts):
            try:
                return factory().execute()
            except self._http_error_types as exc:
                status = _http_status(exc)
                transient = status in _TRANSIENT_STATUSES or _rate_limited_403(exc)
                if transient:
                    last_transient = exc
                    if attempt + 1 < self._request_attempts:
                        self._sleeper(float(2**attempt))
                        continue
                    break
                if status == 401:
                    raise TransportAuthError("Google source request was not authenticated") from None
                if status == 403:
                    raise TransportPermissionError("Google source request was not permitted") from None
                if status == 404:
                    raise TransportNotFoundError("Google source object was not found") from None
                raise TransportPermanentError("Google source request failed permanently") from None
        assert last_transient is not None
        raise TransportTransientError("Google source request exhausted transient retries") from None

    def observe(self, spec: AuthorityObjectSpec) -> SourceObservation:
        raw = self._execute(
            lambda: self._drive.files().get(
                fileId=spec.object_id,
                fields=DRIVE_METADATA_FIELDS,
                supportsAllDrives=True,
            )
        )
        return _metadata_to_observation(raw)

    def read(self, spec: AuthorityObjectSpec) -> SourceContent:
        if spec.representation_schema != GOOGLE_DOC_MARKDOWN_REPRESENTATION:
            raise UnsupportedRepresentationError("Google source representation schema is unsupported")
        if spec.expected_mime_type != _GOOGLE_DOC_MIME:
            raise UnsupportedRepresentationError("Markdown export requires a native Google Doc")
        raw = self._execute(
            lambda: self._drive.files().export(fileId=spec.object_id, mimeType=_MARKDOWN_MIME)
        )
        if type(raw) is not bytes:
            raise TransportPermanentError("Drive Markdown export did not return bytes")
        return SourceContent(representation_schema=GOOGLE_DOC_MARKDOWN_REPRESENTATION, data=raw)

    def list_members(self, parent_id: str) -> tuple[SourceObservation, ...]:
        observations: list[SourceObservation] = []
        page_token: str | None = None
        while True:
            kwargs: dict[str, object] = {
                "q": f"'{parent_id}' in parents and trashed = false",
                "fields": f"nextPageToken,files({DRIVE_METADATA_FIELDS})",
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
                "pageSize": 1000,
            }
            if page_token is not None:
                kwargs["pageToken"] = page_token

            def request() -> _Request:
                return self._drive.files().list(**kwargs)

            raw = self._execute(request)
            try:
                wire = _DriveListWire.model_validate(raw, strict=True)
            except ValidationError as exc:
                raise TransportPermanentError("Drive membership response failed strict validation") from exc
            observations.extend(
                _metadata_to_observation(item.model_dump(mode="python")) for item in wire.files
            )
            page_token = wire.nextPageToken
            if page_token is None:
                break
        return tuple(sorted(observations, key=lambda item: item.object_id))
