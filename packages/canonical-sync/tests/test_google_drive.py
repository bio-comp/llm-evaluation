from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pytest

from canonical_sync.domain import AuthorityObjectSpec
from canonical_sync.errors import (
    TransportAuthError,
    TransportNotFoundError,
    TransportPermissionError,
    TransportPermanentError,
    TransportTransientError,
    UnsupportedRepresentationError,
)
from canonical_sync.google_drive import (
    DOCS_FIELDS,
    DRIVE_METADATA_FIELDS,
    DRIVE_READONLY_SCOPE,
    GoogleDriveDocsTransport,
)


class FakeHttpError(Exception):
    def __init__(self, status: int, content: bytes = b"") -> None:
        super().__init__(f"provider error {status}: {content!r}")
        self.resp = type("Resp", (), {"status": status})()
        self.content = content


class Request:
    def __init__(self, value=None, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error

    def execute(self):
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.value)


class Resource:
    def __init__(self, values: list[object]) -> None:
        self.values = list(values)
        self.calls: list[dict[str, object]] = []

    def get(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            return Request(error=value)
        return Request(value=value)

    def list(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            return Request(error=value)
        return Request(value=value)


class DriveService:
    def __init__(self, values: list[object]) -> None:
        self.resource = Resource(values)

    def files(self):
        return self.resource


class DocsService:
    def __init__(self, values: list[object]) -> None:
        self.resource = Resource(values)

    def documents(self):
        return self.resource


def metadata(*, version: str = "7", object_id: str = "object-001", name: str = "README.md"):
    return {
        "id": object_id,
        "name": name,
        "mimeType": "application/vnd.google-apps.document",
        "version": version,
        "modifiedTime": "2026-10-26T12:00:00.000Z",
        "parents": ["parent-001"],
        "trashed": False,
        "capabilities": {"canDownload": True},
    }


def docs(*, table: bool = False):
    content = (
        [{"table": {"tableRows": []}}]
        if table
        else [
            {"sectionBreak": {"sectionStyle": {"columnSeparatorStyle": "NONE"}}},
            {
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    "elements": [
                        {
                            "startIndex": 1,
                            "endIndex": 9,
                            "textRun": {"content": "Heading\n", "textStyle": {"bold": True}},
                        }
                    ],
                }
            },
            {
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                    "elements": [{"textRun": {"content": "Body\n"}}],
                }
            },
        ]
    )
    return {
        "documentId": "object-001",
        "title": "README.md",
        "tabs": [
            {
                "tabProperties": {"tabId": "t.0", "title": "Tab 1", "index": 0},
                "documentTab": {"body": {"content": content}},
                "childTabs": [],
            }
        ],
    }


def spec() -> AuthorityObjectSpec:
    return AuthorityObjectSpec(
        logical_path="README.md",
        provider="google_drive",
        object_id="object-001",
        expected_name="README.md",
        expected_mime_type="application/vnd.google-apps.document",
        expected_parent_id="parent-001",
        representation_schema="google-docs-visible-text-v1",
    )


def transport(drive_values, docs_values=()):
    drive = DriveService(list(drive_values))
    docs_service = DocsService(list(docs_values))
    value = GoogleDriveDocsTransport(
        drive_service=drive,
        docs_service=docs_service,
        http_error_types=(FakeHttpError,),
        request_attempts=2,
        sleeper=lambda _: None,
    )
    return value, drive, docs_service


def test_exact_readonly_scope_and_request_fields_options() -> None:
    value, drive, docs_service = transport([metadata()], [docs()])
    observed = value.observe(spec())
    acquired = value.read(spec())

    assert DRIVE_READONLY_SCOPE == "https://www.googleapis.com/auth/drive.readonly"
    assert observed.version == 7
    assert drive.resource.calls == [{
        "fileId": "object-001",
        "fields": DRIVE_METADATA_FIELDS,
        "supportsAllDrives": True,
    }]
    assert docs_service.resource.calls == [{
        "documentId": "object-001",
        "includeTabsContent": True,
        "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
        "fields": DOCS_FIELDS,
    }]
    assert acquired.representation_schema == "google-docs-visible-text-v1"
    assert b"Heading" in acquired.data and b"Body" in acquired.data


def test_representation_is_deterministic_across_irrelevant_mapping_order() -> None:
    first = docs()
    second = copy.deepcopy(first)
    second["tabs"][0]["tabProperties"] = dict(
        reversed(list(second["tabs"][0]["tabProperties"].items()))
    )
    a, _, _ = transport([], [first])
    b, _, _ = transport([], [second])
    assert a.read(spec()).data == b.read(spec()).data


def test_unsupported_document_structure_fails_closed() -> None:
    value, _, _ = transport([], [docs(table=True)])
    with pytest.raises(UnsupportedRepresentationError, match="table"):
        value.read(spec())


def test_malformed_or_mismatched_provider_responses_fail_without_raw_body() -> None:
    value, _, _ = transport([metadata(version="not-an-int")])
    with pytest.raises(TransportPermanentError, match="metadata"):
        value.observe(spec())
    wrong = docs()
    wrong["documentId"] = "wrong"
    value, _, _ = transport([], [wrong])
    with pytest.raises(TransportPermanentError, match="document identity"):
        value.read(spec())


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, b"SECRET auth", TransportAuthError),
        (403, b"SECRET forbidden", TransportPermissionError),
        (404, b"SECRET absent", TransportNotFoundError),
        (400, b"SECRET bad request", TransportPermanentError),
    ],
)
def test_nonretryable_http_errors_are_typed_and_sanitized(
    status: int, body: bytes, expected: type[Exception]
) -> None:
    value, _, _ = transport([FakeHttpError(status, body)])
    with pytest.raises(expected) as caught:
        value.observe(spec())
    assert "SECRET" not in str(caught.value)


def test_transient_http_retries_boundedly_and_sanitizes_failure() -> None:
    value, drive, _ = transport([
        FakeHttpError(503, b"SECRET first"),
        FakeHttpError(503, b"SECRET second"),
    ])
    with pytest.raises(TransportTransientError) as caught:
        value.observe(spec())
    assert len(drive.resource.calls) == 2
    assert "SECRET" not in str(caught.value)


def test_rate_limit_403_is_transient_but_other_403_is_permission() -> None:
    rate = b'{"error":{"errors":[{"reason":"rateLimitExceeded"}]}}'
    value, drive, _ = transport([FakeHttpError(403, rate), metadata()])
    assert value.observe(spec()).version == 7
    assert len(drive.resource.calls) == 2


def test_package_google_clients_are_optional() -> None:
    project = tomllib.loads(Path("packages/canonical-sync/pyproject.toml").read_text())
    assert project["project"]["dependencies"] == ["pydantic==2.14.1"]
    assert project["project"]["optional-dependencies"]["drive"] == [
        "google-api-python-client==2.201.0",
        "google-auth==2.62.0",
    ]
