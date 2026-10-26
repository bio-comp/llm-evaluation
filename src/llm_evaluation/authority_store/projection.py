from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import AuthorityMismatchError, AuthorityObjectIdentity, ClosedRecord


PROJECTION_SCHEMA = "llm-evaluation-google-doc-text-v1"


class UnsupportedDocumentStructure(ValueError):
    pass


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SectionStyle(WireModel):
    sectionType: str


class SectionBreak(WireModel):
    sectionStyle: SectionStyle


class TextStyle(WireModel):
    link: dict[str, Any] | None = None


class TextRun(WireModel):
    content: str
    textStyle: TextStyle = Field(default_factory=TextStyle)


class ParagraphElement(WireModel):
    startIndex: int | None = None
    endIndex: int | None = None
    textRun: TextRun | None = None


class ParagraphStyle(WireModel):
    namedStyleType: str


class Bullet(WireModel):
    listId: str
    nestingLevel: int | None = None


class Paragraph(WireModel):
    elements: list[ParagraphElement] = Field(default_factory=list)
    paragraphStyle: ParagraphStyle
    bullet: Bullet | None = None


class StructuralElement(WireModel):
    startIndex: int | None = None
    endIndex: int | None = None
    paragraph: Paragraph | None = None
    sectionBreak: SectionBreak | None = None
    table: dict[str, Any] | None = None
    tableOfContents: dict[str, Any] | None = None

    @model_validator(mode="after")
    def one_kind(self) -> "StructuralElement":
        kinds = (self.paragraph, self.sectionBreak, self.table, self.tableOfContents)
        if sum(value is not None for value in kinds) != 1:
            raise ValueError(
                "structural element must expose exactly one requested kind"
            )
        return self


class Body(WireModel):
    content: list[StructuralElement]


class DocumentTab(WireModel):
    body: Body
    headers: dict[str, Any] = Field(default_factory=dict)
    footers: dict[str, Any] = Field(default_factory=dict)
    footnotes: dict[str, Any] = Field(default_factory=dict)
    inlineObjects: dict[str, Any] = Field(default_factory=dict)
    positionedObjects: dict[str, Any] = Field(default_factory=dict)
    namedRanges: dict[str, Any] = Field(default_factory=dict)


class TabProperties(WireModel):
    tabId: str
    title: str
    parentTabId: str | None = None
    index: int


class ChildTab(WireModel):
    tabProperties: TabProperties


class Tab(WireModel):
    tabProperties: TabProperties
    childTabs: list[ChildTab] = Field(default_factory=list)
    documentTab: DocumentTab


class NativeDocumentWire(WireModel):
    documentId: str
    title: str
    suggestionsViewMode: str
    tabs: list[Tab]


class SemanticProjection(ClosedRecord):
    projection_schema: str
    text: str
    sha256: str


def _digest(text: str) -> str:
    semantic = json.dumps(
        {"schema": PROJECTION_SCHEMA, "text": text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(semantic).hexdigest()


def project_native_document(
    raw: dict[str, Any], identity: AuthorityObjectIdentity
) -> SemanticProjection:
    try:
        doc = NativeDocumentWire.model_validate(raw)
    except Exception as exc:
        raise UnsupportedDocumentStructure(
            "native Docs response does not match the supported schema"
        ) from exc
    if doc.documentId != identity.file_id or doc.title != identity.expected_name:
        raise AuthorityMismatchError(
            "Docs object identity does not match the registered authority"
        )
    if doc.suggestionsViewMode != "PREVIEW_WITHOUT_SUGGESTIONS":
        raise UnsupportedDocumentStructure(
            "semantic read must reject unresolved suggestions"
        )
    if len(doc.tabs) != 1:
        raise UnsupportedDocumentStructure("projection supports exactly one root tab")
    tab = doc.tabs[0]
    if tab.tabProperties.parentTabId is not None or tab.childTabs:
        raise UnsupportedDocumentStructure("nested tabs are unsupported")
    dt = tab.documentTab
    if any(
        (
            dt.headers,
            dt.footers,
            dt.footnotes,
            dt.inlineObjects,
            dt.positionedObjects,
            dt.namedRanges,
        )
    ):
        raise UnsupportedDocumentStructure(
            "out-of-body semantic structures are unsupported"
        )
    content = dt.body.content
    if not content or content[0].sectionBreak is None:
        raise UnsupportedDocumentStructure("expected the native leading section break")
    if content[0].sectionBreak.sectionStyle.sectionType != "CONTINUOUS":
        raise UnsupportedDocumentStructure("unsupported section break type")
    parts: list[str] = []
    for element in content[1:]:
        if element.paragraph is None:
            raise UnsupportedDocumentStructure(
                "only body paragraphs are supported after the leading section break"
            )
        paragraph = element.paragraph
        if paragraph.bullet is not None:
            raise UnsupportedDocumentStructure(
                "list paragraphs require an explicit list projection"
            )
        if paragraph.paragraphStyle.namedStyleType != "NORMAL_TEXT":
            raise UnsupportedDocumentStructure(
                "named paragraph styles are semantically unsupported"
            )
        for child in paragraph.elements:
            if child.textRun is None:
                raise UnsupportedDocumentStructure(
                    "non-text paragraph elements are unsupported"
                )
            if child.textRun.textStyle.link is not None:
                raise UnsupportedDocumentStructure(
                    "link targets are not represented by the text-only projection"
                )
            parts.append(child.textRun.content)
    text = "".join(parts)
    return SemanticProjection(
        projection_schema=PROJECTION_SCHEMA, text=text, sha256=_digest(text)
    )
