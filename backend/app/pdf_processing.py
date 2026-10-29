"""PDF storage and text extraction helpers."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pymupdf as fitz

import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
import logging

from markitdown import MarkItDown

LOGGER = logging.getLogger(__name__)



from fastapi import UploadFile

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PAGE_COUNT = 80
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage" / "documents"


class PdfProcessingError(ValueError):
    """Indicate that an uploaded file cannot be processed."""


@dataclass(frozen=True)
class ExtractedChunk:
    """Text and source page for one chunk."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ProcessedPdf:
    """Result of storing and extracting an uploaded PDF."""

    storage_key: str
    sha256: str
    page_count: int
    original_filename: str
    chunks: list[ExtractedChunk]
    warning_pages: list[int]


def process_upload(upload: UploadFile) -> ProcessedPdf:
    """Store an upload, validate it, and extract page-local chunks."""
    original_filename = upload.filename or ""
    if Path(original_filename).suffix.lower() != ".pdf":
        raise PdfProcessingError("Only PDF files are supported.")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    storage_key = f"{uuid4()}.pdf"
    target = STORAGE_DIR / storage_key
    digest = sha256()
    total_size = 0

    try:
        with target.open("xb") as output:
            while True:
                block = upload.file.read(1024 * 1024)
                if not block:
                    break
                total_size += len(block)
                if total_size > MAX_FILE_SIZE:
                    raise PdfProcessingError(
                        "PDF file must not exceed 50 MiB."
                    )
                digest.update(block)
                output.write(block)

        if total_size == 0:
            raise PdfProcessingError("The uploaded PDF is empty.")

        return _extract_pdf(
            target=target,
            storage_key=storage_key,
            original_filename=original_filename,
            digest=digest.hexdigest(),
        )
    except PdfProcessingError:
        target.unlink(missing_ok=True)
        raise
    except (fitz.FileDataError, RuntimeError) as exc:
        target.unlink(missing_ok=True)
        raise PdfProcessingError(
            "The uploaded file is not a valid PDF."
        ) from exc
    except Exception:
        target.unlink(missing_ok=True)
        raise


def remove_storage_file(storage_key: str) -> None:
    """Remove a server-owned file after a failed database write."""
    path = STORAGE_DIR / storage_key
    if path.parent == STORAGE_DIR:
        path.unlink(missing_ok=True)


def _ocr_page(page: fitz.Page, page_number: int) -> str:
    """Read a page image using the installed Tesseract executable."""
    executable = (
        os.environ.get("TESSERACT_CMD")
        or shutil.which("tesseract")
    )
    language = os.environ.get("PAPERLENS_OCR_LANGUAGE", "eng")

    if not executable:
        raise PdfProcessingError(
            "OCR is required. Set TESSERACT_CMD to tesseract.exe."
        )

    with TemporaryDirectory(prefix="paperlens_ocr_") as directory:
        image_path = Path(directory) / "page.png"

        try:
            pixmap = page.get_pixmap(dpi=200, alpha=False)
            pixmap.save(str(image_path))

            result = subprocess.run(
                [
                    executable,
                    str(image_path),
                    "stdout",
                    "-l",
                    language,
                    "--psm",
                    "3",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PdfProcessingError(
                f"OCR timed out on page {page_number}."
            ) from exc
        except OSError as exc:
            raise PdfProcessingError(
                "Cannot run Tesseract. Check TESSERACT_CMD."
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise PdfProcessingError(
                f"Cannot render page {page_number} for OCR."
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip()[-500:]
            raise PdfProcessingError(
                f"OCR failed on page {page_number}: {detail}"
            )

        return result.stdout.strip()



def _markitdown_page(
    pdf: fitz.Document,
    page_index: int,
    converter: MarkItDown,
) -> str:
    """Convert one PDF page while preserving its source page mapping."""
    with TemporaryDirectory(prefix="paperlens_md_") as directory:
        page_path = Path(directory) / "page.pdf"

        with fitz.open() as single_page:
            single_page.insert_pdf(
                pdf,
                from_page=page_index,
                to_page=page_index,
            )
            single_page.save(str(page_path))

        result = converter.convert_local(str(page_path))

        # Support the result attribute used by installed versions.
        for attribute in ("text_content", "markdown"):
            content = getattr(result, attribute, None)
            if isinstance(content, str) and content.strip():
                return content.strip()

        return ""


def _text_character_count(text: str) -> int:
    """Count letters and numbers without Markdown formatting characters."""
    return sum(character.isalnum() for character in text)





def _extract_pdf(
    target: Path,
    storage_key: str,
    original_filename: str,
    digest: str,
) -> ProcessedPdf:
    """Extract text from a validated PDF without opening a DB transaction."""
    try:
        pdf = fitz.open(stream=target.read_bytes(), filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise PdfProcessingError(
            "The uploaded file is not a valid PDF."
        ) from exc

    with pdf:
        if pdf.needs_pass:
            raise PdfProcessingError(
                "Password-protected PDFs are not supported."
            )
        page_count = len(pdf)
        if page_count == 0:
            raise PdfProcessingError("The PDF does not contain any pages.")
        if page_count > MAX_PAGE_COUNT:
            raise PdfProcessingError(
                f"PDF files must not exceed {MAX_PAGE_COUNT} pages."
            )

        converter = MarkItDown(enable_plugins=False)

        chunks: list[ExtractedChunk] = []
        warning_pages: list[int] = []

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            # Native text is a fallback and helps identify sparse pages.
            try:
                native_text = page.get_text("text").strip()
            except (RuntimeError, ValueError):
                native_text = ""

            try:
                text = _markitdown_page(
                    pdf,
                    page_index,
                    converter,
                )
                method = "markitdown"
            except Exception as exc:
                LOGGER.warning(
                    "MarkItDown failed on page %s: %s",
                    page_number,
                    type(exc).__name__,
                )
                text = ""
                method = "native_fallback"

            if not text:
                text = native_text
                method = "native_fallback"

            # Heuristic: images with very little native text may be scans.
            sparse_image_page = (
                _text_character_count(native_text) < 40
                and bool(page.get_images())
            )
            needs_ocr = not text.strip() or sparse_image_page

            if needs_ocr:
                ocr_text = _ocr_page(page, page_number)

                # Retain existing text when OCR produces less content.
                if (
                    _text_character_count(ocr_text)
                    > _text_character_count(text)
                ):
                    text = ocr_text
                    method = "ocr"

                # A sparse image page remains uncertain.
                if method != "ocr":
                    warning_pages.append(page_number)

            if not text.strip():
                if page_number not in warning_pages:
                    warning_pages.append(page_number)
                continue

            LOGGER.info(
                "pdf_page_extracted page=%s method=%s characters=%s",
                page_number,
                method,
                len(text),
            )

            chunks.extend(
                ExtractedChunk(page_number, chunk)
                for chunk in _split_page_text(text)
            )

        if not chunks:
            raise PdfProcessingError(
                "No readable text was found after extraction and OCR."
            )

    return ProcessedPdf(
        storage_key=storage_key,
        sha256=digest,
        page_count=page_count,
        original_filename=original_filename,
        chunks=chunks,
        warning_pages=warning_pages,
    )


def _split_page_text(text: str) -> list[str]:
    """Split one page into overlapping character-sized chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        proposed_end = min(start + CHUNK_SIZE, len(text))
        end = proposed_end
        if proposed_end < len(text):
            boundary = max(
                text.rfind("\n\n", start + 1, proposed_end),
                text.rfind("\n", start + 1, proposed_end),
                text.rfind(". ", start + 1, proposed_end),
                text.rfind("? ", start + 1, proposed_end),
                text.rfind("! ", start + 1, proposed_end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + (
                    2
                    if text[boundary : boundary + 2]
                    in ("\n\n", ". ", "? ", "! ")
                    else 1
                )
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks
