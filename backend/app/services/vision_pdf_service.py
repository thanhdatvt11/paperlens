"""Store/validate PDFs for vision without invoking text extraction or OCR."""

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pymupdf as fitz

from app.pdf_processing import (
    MAX_FILE_SIZE, MAX_PAGE_COUNT, STORAGE_DIR, PdfProcessingError, ProcessedPdf,
)


def pdf_path(storage_key):
    root = STORAGE_DIR.resolve()
    path = (root / storage_key).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PdfProcessingError("PDF storage path is invalid") from exc
    return path


def read_pdf(storage_key, expected_sha, expected_pages=None):
    data = pdf_path(storage_key).read_bytes()
    if len(data) > MAX_FILE_SIZE or sha256(data).hexdigest() != expected_sha:
        raise PdfProcessingError("PDF content differs from the queued source")
    pdf = fitz.open(stream=data, filetype="pdf")
    if pdf.needs_pass or not 1 <= len(pdf) <= MAX_PAGE_COUNT:
        pdf.close()
        raise PdfProcessingError("PDF is password protected or has invalid page count")
    if expected_pages is not None and len(pdf) != expected_pages:
        pdf.close()
        raise PdfProcessingError("PDF page count differs from the queued source")
    return pdf


def store_upload(upload):
    name = upload.filename or ""
    if Path(name).suffix.lower() != ".pdf":
        raise PdfProcessingError("Only PDF files are supported")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{uuid4()}.pdf"
    target = pdf_path(key)
    digest, total = sha256(), 0
    try:
        with target.open("xb") as file:
            while data := upload.file.read(1024 * 1024):
                total += len(data)
                if total > MAX_FILE_SIZE:
                    raise PdfProcessingError("PDF file must not exceed 50 MiB")
                digest.update(data)
                file.write(data)
        if total == 0:
            raise PdfProcessingError("Uploaded PDF is empty")
        with read_pdf(key, digest.hexdigest()) as pdf:
            count = len(pdf)
        return ProcessedPdf(key, digest.hexdigest(), count, name, [], [])
    except Exception:
        target.unlink(missing_ok=True)
        raise

