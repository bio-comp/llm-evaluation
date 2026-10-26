from __future__ import annotations

from typing import Any


DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_FIELDS = "id,name,mimeType,version,modifiedTime,parents,trashed"
DOCS_FIELDS = (
    "documentId,title,suggestionsViewMode,"
    "tabs(tabProperties(tabId,title,parentTabId,index),childTabs(tabProperties(tabId,title,parentTabId,index)),"
    "documentTab(body(content(startIndex,endIndex,"
    "paragraph(elements(startIndex,endIndex,textRun(content,textStyle(link))),paragraphStyle(namedStyleType),bullet(listId,nestingLevel)),"
    "sectionBreak(sectionStyle(sectionType)),table(rows),tableOfContents(content(startIndex,endIndex)))),"
    "headers,footers,footnotes,inlineObjects,positionedObjects,namedRanges))"
)


class GoogleReadError(RuntimeError):
    """Sanitized Google read/auth failure; provider response text is deliberately omitted."""


class GoogleDriveDocsTransport:
    def __init__(self, drive_service: Any, docs_service: Any):
        self._drive = drive_service
        self._docs = docs_service

    @classmethod
    def from_adc(cls) -> "GoogleDriveDocsTransport":
        try:
            import google.auth  # type: ignore[import-untyped]
            from googleapiclient.discovery import build  # type: ignore[import-untyped]

            credentials, _ = google.auth.default(scopes=[DRIVE_READONLY_SCOPE])
            declared = getattr(credentials, "scopes", None)
            if declared is None or set(declared) != {DRIVE_READONLY_SCOPE}:
                raise GoogleReadError(
                    "local Google credentials do not prove the exact drive.readonly scope"
                )
            drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
            docs = build("docs", "v1", credentials=credentials, cache_discovery=False)
            return cls(drive, docs)
        except GoogleReadError:
            raise
        except Exception:
            raise GoogleReadError("read-only Google authentication failed") from None

    def get_metadata(self, file_id: str) -> dict[str, Any]:
        try:
            value = (
                self._drive.files()
                .get(fileId=file_id, fields=DRIVE_FIELDS, supportsAllDrives=True)
                .execute(num_retries=0)
            )
        except Exception:
            raise GoogleReadError("read-only Drive metadata request failed") from None
        if not isinstance(value, dict):
            raise GoogleReadError("Drive metadata response was not an object")
        return value

    def get_document(self, file_id: str) -> dict[str, Any]:
        try:
            value = (
                self._docs.documents()
                .get(
                    documentId=file_id,
                    fields=DOCS_FIELDS,
                    includeTabsContent=True,
                    suggestionsViewMode="PREVIEW_WITHOUT_SUGGESTIONS",
                )
                .execute(num_retries=0)
            )
        except Exception:
            raise GoogleReadError("read-only Docs request failed") from None
        if not isinstance(value, dict):
            raise GoogleReadError("Docs response was not an object")
        return value
