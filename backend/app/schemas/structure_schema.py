"""Source-linked structure added to whole-document review results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.ai_schema import DocumentReviewResult, ReviewEvidence, ReviewSource


class SectionSpan(ReviewSource):
    """An inclusive range in the original chunk's splitlines(keepends=True)."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class DocumentSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    parent_id: str | None
    title: str
    title_origin: Literal["source", "generated"]
    level: int = Field(ge=1, le=3)
    kind: Literal["body", "cover", "table_of_contents", "references"]
    heading_evidence: list[ReviewEvidence]
    spans: list[SectionSpan]
    sources: list[ReviewSource]
    summary: str


class DocumentStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules_version: str = "source-sections-1"
    semantic_correctness_verified: bool = False
    sections: list[DocumentSection]
    warnings: list[str]


class StructuredDocumentReviewResult(DocumentReviewResult):
    structure: DocumentStructure
