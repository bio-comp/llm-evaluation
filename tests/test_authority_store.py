from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from llm_evaluation.authority_store.acquisition import (
    AuthorityMismatchError,
    SourceMovementError,
    acquire_native_document,
)
from llm_evaluation.authority_store.contracts import (
    AcquisitionCapability,
    AuthorityObjectIdentity,
    LastReconciledState,
    PublicImpact,
    ReconciliationStatus,
    SourceObservation,
    VerifiedPublicationReceipt,
)
from llm_evaluation.authority_store.google_api import (
    DOCS_FIELDS,
    DRIVE_FIELDS,
    DRIVE_READONLY_SCOPE,
    GoogleDriveDocsTransport,
)
from llm_evaluation.authority_store.projection import (
    UnsupportedDocumentStructure,
    project_native_document,
)
from llm_evaluation.authority_store.reconciliation import (
    PublicationBoundaryError,
    RemoteRaceError,
    atomic_replace_public,
    classify_reconciliation,
    complete_reconciliation,
    filter_public_payload,
    verify_remote_base,
    read_last_reconciled_state,
    write_last_reconciled_state,
)

FILE_ID = "fake-file-id-1234567890"
PARENT_ID = "fake-parent-id-1234567890"
MIME = "application/vnd.google-apps.document"


def identity() -> AuthorityObjectIdentity:
    return AuthorityObjectIdentity(
        logical_role="README",
        file_id=FILE_ID,
        expected_name="README.md",
        expected_mime_type=MIME,
        expected_parent_id=PARENT_ID,
        capability=AcquisitionCapability.READ_ONLY,
    )


def metadata(*, version: int = 7, name: str = "README.md", parent: str = PARENT_ID):
    return {
        "id": FILE_ID,
        "name": name,
        "mimeType": MIME,
        "version": str(version),
        "modifiedTime": "2026-10-26T01:02:03.004Z",
        "parents": [parent],
        "trashed": False,
    }


def document(text: str = "# Public-looking syntax\n\nBody\n"):
    return {
        "documentId": FILE_ID,
        "title": "README.md",
        "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
        "tabs": [
            {
                "tabProperties": {"tabId": "t.0", "title": "Tab 1", "index": 0},
                "childTabs": [],
                "documentTab": {
                    "body": {
                        "content": [
                            {
                                "endIndex": 1,
                                "sectionBreak": {
                                    "sectionStyle": {"sectionType": "CONTINUOUS"}
                                },
                            },
                            {
                                "startIndex": 1,
                                "endIndex": 1 + len(text),
                                "paragraph": {
                                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                                    "elements": [
                                        {
                                            "startIndex": 1,
                                            "endIndex": 1 + len(text),
                                            "textRun": {
                                                "content": text,
                                                "textStyle": {},
                                            },
                                        }
                                    ],
                                },
                            },
                        ]
                    },
                    "headers": {},
                    "footers": {},
                    "footnotes": {},
                    "inlineObjects": {},
                    "positionedObjects": {},
                    "namedRanges": {},
                },
            }
        ],
    }


class FakeTransport:
    def __init__(self, metadata_sequence, doc=None):
        self.metadata_sequence = list(metadata_sequence)
        self.doc = document() if doc is None else doc
        self.metadata_calls = 0
        self.document_calls = 0

    def get_metadata(self, file_id: str):
        self.metadata_calls += 1
        return self.metadata_sequence.pop(0)

    def get_document(self, file_id: str):
        self.document_calls += 1
        return self.doc


class ContractTests(unittest.TestCase):
    def test_strict_read_only_contract_rejects_revision_and_coercion(self):
        with self.assertRaises(ValidationError):
            SourceObservation(
                file_id=FILE_ID,
                name="README.md",
                mime_type=MIME,
                version="7",
                modified_time="2026-10-26T01:02:03.004Z",
                parent_ids=(PARENT_ID,),
                capability=AcquisitionCapability.READ_ONLY,
            )
        good = SourceObservation(
            file_id=FILE_ID,
            name="README.md",
            mime_type=MIME,
            version=7,
            modified_time="2026-10-26T01:02:03.004Z",
            parent_ids=(PARENT_ID,),
            capability=AcquisitionCapability.READ_ONLY,
        )
        with self.assertRaises(ValidationError):
            SourceObservation.model_validate(
                good.model_dump() | {"revision_id": "opaque"}
            )
        with self.assertRaises(ValidationError):
            SourceObservation.model_validate(
                good.model_dump() | {"headRevisionId": "binary-only"}
            )

    def test_status_and_capability_vocabularies_are_distinct(self):
        self.assertEqual(
            {x.value for x in ReconciliationStatus},
            {"NONE", "REQUIRED", "BLOCKED", "COMPLETE"},
        )
        self.assertEqual(
            {x.value for x in AcquisitionCapability}, {"READ_ONLY", "MUTATION_CAPABLE"}
        )


class ProjectionTests(unittest.TestCase):
    def test_projection_is_deterministic_and_text_sensitive(self):
        a = project_native_document(document("same\n"), identity())
        reordered = json.loads(json.dumps(document("same\n"), sort_keys=True))
        b = project_native_document(reordered, identity())
        c = project_native_document(document("different\n"), identity())
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(a.text, "same\n")
        self.assertNotEqual(a.sha256, c.sha256)

    def test_projection_rejects_wrong_doc_identity_or_suggestion_view(self):
        wrong = document()
        wrong["documentId"] = "other"
        with self.assertRaises(AuthorityMismatchError):
            project_native_document(wrong, identity())
        suggestions = document()
        suggestions["suggestionsViewMode"] = "SUGGESTIONS_INLINE"
        with self.assertRaises(UnsupportedDocumentStructure):
            project_native_document(suggestions, identity())

    def test_projection_fails_closed_on_semantic_structures(self):
        mutations = []
        x = document()
        x["tabs"].append(x["tabs"][0].copy())
        mutations.append(x)
        x = document()
        x["tabs"][0]["childTabs"] = [
            {"tabProperties": {"tabId": "c", "title": "c", "index": 0}}
        ]
        mutations.append(x)
        x = document()
        x["tabs"][0]["documentTab"]["body"]["content"][1]["paragraph"]["bullet"] = {
            "listId": "l",
            "nestingLevel": 0,
        }
        mutations.append(x)
        x = document()
        x["tabs"][0]["documentTab"]["body"]["content"][1]["paragraph"][
            "paragraphStyle"
        ]["namedStyleType"] = "HEADING_1"
        mutations.append(x)
        x = document()
        x["tabs"][0]["documentTab"]["body"]["content"][1]["paragraph"]["elements"][0][
            "textRun"
        ]["textStyle"] = {"link": {"url": "https://example.invalid"}}
        mutations.append(x)
        x = document()
        x["tabs"][0]["documentTab"]["headers"] = {"h": {}}
        mutations.append(x)
        x = document()
        x["tabs"][0]["documentTab"]["body"]["content"][1] = {
            "startIndex": 1,
            "endIndex": 2,
            "table": {"rows": []},
        }
        mutations.append(x)
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaises(UnsupportedDocumentStructure):
                    project_native_document(value, identity())


class AcquisitionTests(unittest.TestCase):
    def test_stable_acquisition_separates_semantic_and_source_identity(self):
        first = acquire_native_document(
            FakeTransport([metadata(), metadata()]), identity()
        )
        second = acquire_native_document(
            FakeTransport([metadata(version=99), metadata(version=99)]), identity()
        )
        self.assertEqual(first.semantic.sha256, second.semantic.sha256)
        self.assertNotEqual(first.receipt.source.version, second.receipt.source.version)
        self.assertEqual(first.receipt.attempt_count, 1)
        self.assertEqual(first.receipt.capability, AcquisitionCapability.READ_ONLY)

    def test_wrong_registered_identity_fails_closed(self):
        for bad in (
            metadata(name="OTHER.md"),
            metadata(parent="wrong"),
            metadata() | {"mimeType": "text/plain"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(AuthorityMismatchError):
                    acquire_native_document(FakeTransport([bad, bad]), identity())

    def test_movement_retries_once_but_repeated_movement_blocks(self):
        moved_once = FakeTransport(
            [
                metadata(version=1),
                metadata(version=2),
                metadata(version=2),
                metadata(version=2),
            ]
        )
        result = acquire_native_document(moved_once, identity())
        self.assertEqual(result.receipt.attempt_count, 2)
        self.assertEqual(moved_once.document_calls, 2)
        repeated = FakeTransport(
            [
                metadata(version=1),
                metadata(version=2),
                metadata(version=2),
                metadata(version=3),
            ]
        )
        with self.assertRaises(SourceMovementError):
            acquire_native_document(repeated, identity())


class ReconciliationTests(unittest.TestCase):
    def test_four_way_classification_does_not_collapse(self):
        current = "a" * 64
        same = LastReconciledState(semantic_sha256=current, public_commit="1" * 40)
        old = LastReconciledState(semantic_sha256="b" * 64, public_commit="1" * 40)
        self.assertEqual(
            classify_reconciliation(current, same, PublicImpact.REQUIRED).status,
            ReconciliationStatus.NONE,
        )
        self.assertEqual(
            classify_reconciliation(current, old, PublicImpact.NONE).status,
            ReconciliationStatus.NONE,
        )
        required = classify_reconciliation(current, old, PublicImpact.REQUIRED)
        self.assertEqual(required.status, ReconciliationStatus.REQUIRED)
        self.assertEqual(
            classify_reconciliation(current, old, PublicImpact.UNKNOWN).status,
            ReconciliationStatus.BLOCKED,
        )
        receipt = VerifiedPublicationReceipt(
            semantic_sha256=current,
            expected_base="1" * 40,
            resulting_commit="2" * 40,
            readback_commit="2" * 40,
            public_payload_sha256="c" * 64,
        )
        self.assertEqual(
            complete_reconciliation(required, receipt).status,
            ReconciliationStatus.COMPLETE,
        )
        with self.assertRaises(ValueError):
            complete_reconciliation(
                required, receipt.model_copy(update={"semantic_sha256": "d" * 64})
            )

    def test_publication_filter_has_positive_and_negative_controls(self):
        safe = b"Derived public implication only.\n"
        self.assertEqual(
            filter_public_payload(safe, private_markers=(FILE_ID, PARENT_ID)), safe
        )
        bad = [
            f"private id {FILE_ID}".encode(),
            f"private parent {PARENT_ID}".encode(),
            b'{"refresh_token":"sentinel-secret"}',
            b"ya29.sentinel-access-token",
            b"AIzaSentinelApiKey012345678901234",
        ]
        for payload in bad:
            with self.subTest(payload=payload):
                with self.assertRaises(PublicationBoundaryError):
                    filter_public_payload(payload, private_markers=(FILE_ID, PARENT_ID))

    def test_atomic_publication_readback_and_remote_race_guard(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "derived.json"
            atomic_replace_public(path, b"one\n")
            self.assertEqual(path.read_bytes(), b"one\n")
            atomic_replace_public(path, b"two\n")
            self.assertEqual(path.read_bytes(), b"two\n")
        verify_remote_base("a" * 40, "a" * 40)
        with self.assertRaises(RemoteRaceError):
            verify_remote_base("a" * 40, "b" * 40)

    def test_last_reconciled_checkpoint_is_private_minimal_and_read_back(self):
        state = LastReconciledState(semantic_sha256="a" * 64, public_commit="b" * 40)
        with tempfile.TemporaryDirectory() as td:
            private = Path(td) / ".private-drive-cache" / "last.json"
            write_last_reconciled_state(private, state)
            self.assertEqual(read_last_reconciled_state(private), state)
            raw = private.read_text()
            self.assertNotIn(FILE_ID, raw)
            self.assertNotIn(PARENT_ID, raw)
            with self.assertRaises(ValueError):
                write_last_reconciled_state(Path(td) / "tracked.json", state)


class TransportTests(unittest.TestCase):
    def test_transport_uses_readonly_scope_exact_fields_and_base_preview(self):
        class Request:
            def __init__(self, result):
                self.result = result

            def execute(self, num_retries=0):
                self.num_retries = num_retries
                return self.result

        class Files:
            def __init__(self):
                self.kw = None

            def get(self, **kw):
                self.kw = kw
                return Request(metadata())

        class Drive:
            def __init__(self):
                self._files = Files()

            def files(self):
                return self._files

        class Documents:
            def __init__(self):
                self.kw = None

            def get(self, **kw):
                self.kw = kw
                return Request(document())

        class Docs:
            def __init__(self):
                self._documents = Documents()

            def documents(self):
                return self._documents

        drive, docs = Drive(), Docs()
        transport = GoogleDriveDocsTransport(drive, docs)
        transport.get_metadata(FILE_ID)
        transport.get_document(FILE_ID)
        self.assertEqual(
            DRIVE_READONLY_SCOPE, "https://www.googleapis.com/auth/drive.readonly"
        )
        self.assertEqual(
            drive._files.kw,
            {"fileId": FILE_ID, "fields": DRIVE_FIELDS, "supportsAllDrives": True},
        )
        self.assertEqual(
            docs._documents.kw,
            {
                "documentId": FILE_ID,
                "fields": DOCS_FIELDS,
                "includeTabsContent": True,
                "suggestionsViewMode": "PREVIEW_WITHOUT_SUGGESTIONS",
            },
        )
