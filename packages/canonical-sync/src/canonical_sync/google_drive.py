from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

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
DOCS_FIELDS = "documentId,title,tabs(tabProperties,documentTab(body(content)),childTabs)"
_DOC_REPRESENTATION = "google-docs-visible-text-v1"
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_GOOGLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class _Request(Protocol):
    def execute(self) -> object: ...


class _FilesResource(Protocol):
    def get(self, **kwargs: object) -> _Request: ...
    def list(self, **kwargs: object) -> _Request: ...


class _DocumentsResource(Protocol):
    def get(self, **kwargs: object) -> _Request: ...


class _DriveService(Protocol):
    def files(self) -> _FilesResource: ...


class _DocsService(Protocol):
    def documents(self) -> _DocumentsResource: ...


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


class _DocumentWire(_WireModel):
    documentId: str
    title: str
    tabs: list[dict[str, object]]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _google_id(value: str, *, label: str) -> str:
    if _GOOGLE_ID_RE.fullmatch(value) is None:
        raise TransportPermanentError(f"{label} is not a supported Google Drive identifier")
    return value


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


def _tab_property(record: dict[str, object], name: str, expected: type[Any]) -> Any:
    value = record.get(name)
    if not isinstance(value, expected):
        raise UnsupportedRepresentationError(f"Docs tab {name} has an unsupported shape")
    return value


def _project_text_run(raw: object) -> str:
    if not isinstance(raw, dict):
        raise UnsupportedRepresentationError("Docs text run has an unsupported shape")
    allowed = {"content", "textStyle", "suggestedInsertionIds", "suggestedDeletionIds"}
    if set(raw) - allowed:
        raise UnsupportedRepresentationError("Docs text run contains an unsupported field")
    content = raw.get("content")
    if not isinstance(content, str):
        raise UnsupportedRepresentationError("Docs text run content is not text")
    for key in ("suggestedInsertionIds", "suggestedDeletionIds"):
        value = raw.get(key)
        if value not in (None, []):
            raise UnsupportedRepresentationError("Docs suggested text cannot enter base representation")
    style = raw.get("textStyle")
    if style is not None:
        if not isinstance(style, dict):
            raise UnsupportedRepresentationError("Docs text style has an unsupported shape")
        if "link" in style:
            raise UnsupportedRepresentationError("Docs linked text is not supported by visible-text-v1")
    return content


def _project_paragraph(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise UnsupportedRepresentationError("Docs paragraph has an unsupported shape")
    allowed = {"elements", "paragraphStyle"}
    if set(raw) - allowed:
        raise UnsupportedRepresentationError("Docs paragraph contains unsupported semantic structure")
    elements = raw.get("elements", [])
    if not isinstance(elements, list):
        raise UnsupportedRepresentationError("Docs paragraph elements have an unsupported shape")
    text_parts: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            raise UnsupportedRepresentationError("Docs paragraph element has an unsupported shape")
        content_keys = set(element) - {"startIndex", "endIndex"}
        if content_keys != {"textRun"}:
            raise UnsupportedRepresentationError("Docs paragraph element kind is unsupported")
        text_parts.append(_project_text_run(element["textRun"]))
    style = raw.get("paragraphStyle", {})
    if not isinstance(style, dict):
        raise UnsupportedRepresentationError("Docs paragraph style has an unsupported shape")
    unsupported_style = set(style) - {"namedStyleType", "headingId"}
    if unsupported_style:
        raise UnsupportedRepresentationError("Docs paragraph style contains unsupported structure")
    named_style = style.get("namedStyleType", "NORMAL_TEXT")
    if not isinstance(named_style, str):
        raise UnsupportedRepresentationError("Docs paragraph named style is invalid")
    return {"kind": "paragraph", "named_style": named_style, "text": "".join(text_parts)}


def _project_body(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, dict) or set(raw) != {"content"}:
        raise UnsupportedRepresentationError("Docs body has an unsupported shape")
    content = raw["content"]
    if not isinstance(content, list):
        raise UnsupportedRepresentationError("Docs body content is not a list")
    projected: list[dict[str, object]] = []
    for element in content:
        if not isinstance(element, dict):
            raise UnsupportedRepresentationError("Docs structural element has an unsupported shape")
        semantic = set(element) - {"startIndex", "endIndex"}
        if "table" in semantic or "tableOfContents" in semantic:
            raise UnsupportedRepresentationError("Docs table structures are unsupported by visible-text-v1")
        if semantic == {"sectionBreak"}:
            projected.append({"kind": "section_break"})
        elif semantic == {"paragraph"}:
            projected.append(_project_paragraph(element["paragraph"]))
        else:
            raise UnsupportedRepresentationError("Docs structural element kind is unsupported")
    return projected


def _projected_index(item: dict[str, object]) -> int:
    value = item.get("index")
    if type(value) is not int:
        raise UnsupportedRepresentationError("projected Docs tab index is invalid")
    return value


def _project_tab(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise UnsupportedRepresentationError("Docs tab has an unsupported shape")
    allowed = {"tabProperties", "documentTab", "childTabs"}
    if set(raw) - allowed or "tabProperties" not in raw or "documentTab" not in raw:
        raise UnsupportedRepresentationError("Docs tab fields do not match visible-text-v1")
    props = raw["tabProperties"]
    if not isinstance(props, dict):
        raise UnsupportedRepresentationError("Docs tab properties have an unsupported shape")
    allowed_props = {"tabId", "title", "parentTabId", "index", "nestingLevel", "iconEmoji"}
    if set(props) - allowed_props:
        raise UnsupportedRepresentationError("Docs tab properties contain unsupported fields")
    title = _tab_property(props, "title", str)
    index = _tab_property(props, "index", int)
    icon = props.get("iconEmoji")
    if icon is not None and not isinstance(icon, str):
        raise UnsupportedRepresentationError("Docs tab icon has an unsupported shape")
    document_tab = raw["documentTab"]
    if not isinstance(document_tab, dict) or set(document_tab) != {"body"}:
        raise UnsupportedRepresentationError("Docs document tab has an unsupported shape")
    children_raw = raw.get("childTabs", [])
    if not isinstance(children_raw, list):
        raise UnsupportedRepresentationError("Docs child tabs have an unsupported shape")
    children = sorted((_project_tab(item) for item in children_raw), key=_projected_index)
    return {
        "title": title,
        "index": index,
        "icon_emoji": icon,
        "body": _project_body(document_tab["body"]),
        "children": children,
    }


def _project_document(raw: object, spec: AuthorityObjectSpec) -> bytes:
    try:
        wire = _DocumentWire.model_validate(raw, strict=True)
    except ValidationError as exc:
        raise TransportPermanentError("Docs response failed strict validation") from exc
    if wire.documentId != spec.object_id or wire.title != spec.expected_name:
        raise TransportPermanentError("Docs document identity does not match declared source")
    tabs = sorted((_project_tab(item) for item in wire.tabs), key=_projected_index)
    return _canonical_json_bytes(
        {
            "representation_schema": _DOC_REPRESENTATION,
            "document_title": wire.title,
            "tabs": tabs,
        }
    )


class GoogleDriveDocsTransport:
    def __init__(
        self,
        *,
        drive_service: _DriveService,
        docs_service: _DocsService,
        http_error_types: tuple[type[BaseException], ...],
        request_attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(request_attempts) is not int or request_attempts < 1:
            raise ValueError("request_attempts must be a positive integer")
        self._drive = drive_service
        self._docs = docs_service
        self._http_error_types = http_error_types
        self._request_attempts = request_attempts
        self._sleeper = sleeper

    @classmethod
    def from_application_default(cls, *, request_attempts: int = 3) -> GoogleDriveDocsTransport:
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
            docs_service = build("docs", "v1", credentials=credentials, cache_discovery=False)
        except (DefaultCredentialsError, GoogleAuthError, OSError) as exc:
            raise TransportAuthError("Google application-default read credentials are unavailable") from exc
        return cls(
            drive_service=drive_service,
            docs_service=docs_service,
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
        object_id = _google_id(spec.object_id, label="object ID")
        raw = self._execute(
            lambda: self._drive.files().get(
                fileId=object_id,
                fields=DRIVE_METADATA_FIELDS,
                supportsAllDrives=True,
            )
        )
        return _metadata_to_observation(raw)

    def read(self, spec: AuthorityObjectSpec) -> SourceContent:
        if spec.representation_schema != _DOC_REPRESENTATION:
            raise UnsupportedRepresentationError("Google source representation schema is unsupported")
        if spec.expected_mime_type != "application/vnd.google-apps.document":
            raise UnsupportedRepresentationError("visible-text-v1 requires a native Google Doc")
        object_id = _google_id(spec.object_id, label="document ID")
        raw = self._execute(
            lambda: self._docs.documents().get(
                documentId=object_id,
                includeTabsContent=True,
                suggestionsViewMode="PREVIEW_WITHOUT_SUGGESTIONS",
                fields=DOCS_FIELDS,
            )
        )
        return SourceContent(
            representation_schema=_DOC_REPRESENTATION,
            data=_project_document(raw, spec),
        )

    def list_members(self, parent_id: str) -> tuple[SourceObservation, ...]:
        parent = _google_id(parent_id, label="parent ID")
        observations: list[SourceObservation] = []
        page_token: str | None = None
        while True:
            kwargs: dict[str, object] = {
                "q": f"'{parent}' in parents and trashed = false",
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
            observations.extend(_metadata_to_observation(item.model_dump()) for item in wire.files)
            page_token = wire.nextPageToken
            if page_token is None:
                break
        return tuple(sorted(observations, key=lambda item: item.object_id))
