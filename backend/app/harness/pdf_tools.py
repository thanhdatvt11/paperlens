"""PDF tools scoped to one server-selected document version."""

from dataclasses import dataclass, field

import pymupdf as fitz
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import DocumentVersion
from app.services.vision_pdf_service import read_pdf


@dataclass(frozen=True)
class PDFSource:
    """Server-owned identity of the selected PDF."""

    document_id: int
    version_id: int
    storage_key: str
    sha256: str
    page_count: int


@dataclass
class ToolResult:
    """Structured data and optional images for the model."""

    data: dict[str, object]
    images: list[bytes] = field(default_factory=list)


class EmptyArguments(BaseModel):
    """Arguments for tools that accept no parameters."""

    model_config = ConfigDict(extra="forbid")


class ViewPageArguments(BaseModel):
    """Arguments accepted from the agent."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1, strict=True)


def load_pdf_source(
    db: Session,
    document_id: int,
    version_id: int,
) -> PDFSource:
    """Find a version only when it belongs to the document."""
    statement = select(DocumentVersion).where(
        DocumentVersion.id == version_id,
        DocumentVersion.document_id == document_id,
    )

    version = db.scalar(statement)

    if version is None:
        raise LookupError(
            "Version does not belong to this document."
        )

    return PDFSource(
        document_id=version.document_id,
        version_id=version.id,
        storage_key=version.storage_key,
        sha256=version.sha256,
        page_count=version.page_count,
    )


class PDFTools:
    """Keep one validated PDF open during a harness run."""

    def __init__(self, source: PDFSource) -> None:
        self.source = source
        self._pdf: fitz.Document | None = None

    def __enter__(self) -> "PDFTools":
        self._pdf = read_pdf(
            self.source.storage_key,
            self.source.sha256,
            self.source.page_count,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def _document(self) -> fitz.Document:
        if self._pdf is None:
            raise RuntimeError(
                "Use PDFTools inside a with block."
            )

        return self._pdf

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolResult:
        """Dispatch only explicitly registered PDF tools."""
        if tool_name == "pdf_info":
            EmptyArguments.model_validate(arguments)
            return self.pdf_info()

        if tool_name == "view_pdf_page":
            parsed = ViewPageArguments.model_validate(
                arguments
            )
            return self.view_pdf_page(parsed.page)

        raise PermissionError(
            f"PDF tool is not registered: {tool_name}"
        )

    def pdf_info(self) -> ToolResult:
        """Return source identity and page count."""
        pdf = self._document()

        return ToolResult(
            data={
                "document_id": self.source.document_id,
                "version_id": self.source.version_id,
                "page_count": len(pdf),
            }
        )

    def view_pdf_page(self, page: int) -> ToolResult:
        """Render one PDF page without OCR."""
        pdf = self._document()

        if not 1 <= page <= len(pdf):
            raise ValueError(
                f"page must be between 1 and {len(pdf)}"
            )

        pdf_page = pdf[page - 1]
        longest_side = max(
            pdf_page.rect.width,
            pdf_page.rect.height,
        )

        if longest_side <= 0:
            raise ValueError("Invalid PDF page dimensions.")

        scale = min(2.0, 1400 / longest_side)

        pixmap = pdf_page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )

        return ToolResult(
            data={
                "document_id": self.source.document_id,
                "version_id": self.source.version_id,
                "page_number": page,
                "width": pixmap.width,
                "height": pixmap.height,
            },
            images=[pixmap.tobytes("png")],
        )