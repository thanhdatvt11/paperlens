r"""Inspect native PDF text and supplementary OCR without changing PaperLens.

Requires only PyMuPDF and a Tesseract executable. No model calls or DB writes.
Run from PaperLens/backend, for example:
  python pdf_layout_probe.py --version-id 2 --document-id 5 --pages 4 \
      --tesseract D:\\Download\\tesseract.exe

Alternatively use --pdf PATH without importing the PaperLens application.
Output coordinates are PDF points. Position sorting is NOT semantic order.
OCR confidence and overlap thresholds are heuristics, not correctness proofs.
"""

import argparse
import csv
import io
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz


@dataclass(frozen=True)
class OcrConfig:
    """Settings for a bounded, sequential OCR experiment."""

    executable: str
    language: str = "eng"
    dpi: int = 200
    min_confidence: float = 75.0
    native_overlap: float = 0.35
    timeout_seconds: int = 120
    tessdata: str | None = None


def _native_lines(page: fitz.Page) -> list[dict[str, Any]]:
    """Keep each native text line and its original geometry separately."""
    result = []
    data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                result.append(
                    {
                        "text": text,
                        "bbox": list(line["bbox"]),
                        "source": "native",
                        "confidence": None,
                        "needs_review": False,
                    }
                )
    return result


def _overlap_ratio(box: list[float], native: list[Any]) -> float:
    """Estimate coverage of an OCR word by native word rectangles."""
    rectangle = fitz.Rect(box)
    area = rectangle.get_area()
    if area <= 0:
        return 0.0
    covered = sum((rectangle & fitz.Rect(item[:4])).get_area() for item in native)
    return min(1.0, covered / area)


def _read_tsv(image_path: Path, config: OcrConfig) -> list[dict[str, str]]:
    """Call the executable directly; never use a shell or download anything."""
    command = [config.executable, str(image_path), "stdout"]
    if config.tessdata:
        command.extend(["--tessdata-dir", config.tessdata])
    command.extend(["-l", config.language, "--psm", "11", "tsv"])
    response = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.timeout_seconds,
        check=False,
    )
    if response.returncode:
        raise RuntimeError(f"Tesseract failed: {response.stderr.strip()}")
    reader = csv.DictReader(io.StringIO(response.stdout), delimiter="\t")
    required = {
        "level", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf", "text",
    }
    if not required.issubset(set(reader.fieldnames or [])):
        raise RuntimeError("Tesseract did not return a valid TSV header")
    return list(reader)


def _word_box(row: dict[str, str], scale: float) -> list[float]:
    """Convert raster coordinates back into PDF page coordinates."""
    left, top = float(row["left"]), float(row["top"])
    width, height = float(row["width"]), float(row["height"])
    return [left * scale, top * scale,
            (left + width) * scale, (top + height) * scale]


def _flush_words(words: list[dict[str, Any]], output: list[Any]) -> None:
    """Group only consecutive accepted words, preserving gaps in provenance."""
    if not words:
        return
    output.append(
        {
            "text": " ".join(word["text"] for word in words),
            "bbox": [
                min(word["bbox"][0] for word in words),
                min(word["bbox"][1] for word in words),
                max(word["bbox"][2] for word in words),
                max(word["bbox"][3] for word in words),
            ],
            "source": "ocr",
            "confidence": round(min(word["confidence"] for word in words), 2),
            "needs_review": True,
            "words": list(words),
        }
    )
    words.clear()


def _supplementary_ocr(
    rows: list[dict[str, str]],
    native_words: list[Any],
    config: OcrConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Retain missing, confident OCR words; audit every rejected candidate."""
    accepted, rejected = [], []
    current: list[dict[str, Any]] = []
    previous_key = None
    previous_number = 0
    total_words = 0
    scale = 72.0 / config.dpi

    for row in rows:
        if row["level"] != "5" or not row["text"].strip():
            continue
        total_words += 1
        key = (row["block_num"], row["par_num"], row["line_num"])
        number = int(row["word_num"])
        if key != previous_key or number != previous_number + 1:
            _flush_words(current, accepted)
        previous_key, previous_number = key, number
        box = _word_box(row, scale)
        confidence = float(row["conf"])
        overlap = _overlap_ratio(box, native_words)
        word = {"text": row["text"], "bbox": box, "confidence": confidence}

        if overlap >= config.native_overlap:
            reason = "overlaps_native_text"
        elif confidence < config.min_confidence:
            reason = "low_ocr_confidence"
        elif not any(character.isalnum() for character in row["text"]):
            reason = "punctuation_only_candidate"
        else:
            current.append(word)
            continue

        _flush_words(current, accepted)
        rejected.append({**word, "reason": reason, "native_overlap": overlap})

    _flush_words(current, accepted)
    return accepted, rejected, total_words


def extract_page_layout(page: fitz.Page, config: OcrConfig) -> dict[str, Any]:
    """Combine native lines and OCR candidates without inferring relations."""
    if page.rotation:
        raise ValueError(
            "This probe does not support rotated pages yet; "
            "refusing to merge different coordinate systems."
        )
    if not 72 <= config.dpi <= 300:
        raise ValueError("OCR dpi must be between 72 and 300")
    if not 0 <= config.min_confidence <= 100:
        raise ValueError("OCR confidence must be between 0 and 100")
    if not 0 < config.native_overlap <= 1:
        raise ValueError("Native overlap must be greater than 0 and at most 1")
    if config.timeout_seconds <= 0:
        raise ValueError("OCR timeout must be positive")

    native = _native_lines(page)
    native_words = page.get_text("words", sort=False)
    with tempfile.TemporaryDirectory(prefix="paperlens_ocr_") as work:
        image_path = Path(work) / "page.png"
        page.get_pixmap(dpi=config.dpi, alpha=False).save(image_path)
        rows = _read_tsv(image_path, config)

    supplementary, rejected, total_words = _supplementary_ocr(
        rows, native_words, config
    )
    blocks = sorted(native + supplementary,
                    key=lambda block: (block["bbox"][1], block["bbox"][0]))
    warnings = [
        "Position sorting is not semantic reading order.",
        "All OCR additions require visual review, even with high confidence.",
        "Overlap filtering is heuristic; rejected candidates are retained.",
        "No title-description, column, arrow, or table relationships inferred.",
        "Bounding boxes and source labels must not be treated as PDF prose.",
    ]
    if any(row["reason"] == "low_ocr_confidence" for row in rejected):
        warnings.append("Some OCR candidates were rejected for low confidence.")
    return {
        "page_number": page.number + 1,
        "page_size": {"width": page.rect.width, "height": page.rect.height},
        "coordinate_system": "PDF points, origin at top-left",
        "ordering": "position_sorted_not_semantic",
        "settings": {"dpi": config.dpi,
                     "min_confidence": config.min_confidence,
                     "native_overlap": config.native_overlap,
                     "language": config.language},
        "counts": {"native_lines": len(native),
                   "supplementary_ocr_segments": len(supplementary),
                   "ocr_words_seen": total_words,
                   "ocr_words_rejected": len(rejected)},
        "blocks": blocks,
        "rejected_ocr_words": rejected,
        "warnings": warnings,
    }


def _save_overlay(page: fitz.Page, layout: dict[str, Any], path: Path) -> None:
    """Write a visual audit: blue native boxes, orange OCR additions."""
    with fitz.open() as overlay:
        overlay.insert_pdf(page.parent, from_page=page.number,
                           to_page=page.number)
        target = overlay[0]
        for block in layout["blocks"]:
            color = (0.0, 0.35, 1.0) if block["source"] == "native" else (1, 0.4, 0)
            target.draw_rect(fitz.Rect(block["bbox"]), color=color, width=1)
        target.get_pixmap(dpi=100, alpha=False).save(path)


def _resolve_pdf(args: argparse.Namespace) -> Path:
    """Resolve an explicit file or read an existing PaperLens version."""
    if args.pdf is not None:
        return args.pdf.resolve(strict=True)

    # Lazy imports keep this tool usable without FastAPI or SQLAlchemy.
    from app.core.database import SESSION_FACTORY  # pylint: disable=import-outside-toplevel
    from app.models.document import DocumentVersion  # pylint: disable=import-outside-toplevel

    with SESSION_FACTORY() as database:
        version = database.get(DocumentVersion, args.version_id)
        if version is None:
            raise ValueError("Document version was not found")
        if args.document_id is not None and version.document_id != args.document_id:
            raise ValueError("Version does not belong to the specified document")
        storage_key = version.storage_key
    if Path(storage_key).name != storage_key:
        raise ValueError("Expected a server-owned filename as storage_key")
    paths = [path for path in Path.cwd().rglob(storage_key) if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected one stored PDF, found {len(paths)}")
    return paths[0]


def main() -> None:
    """Run a standalone, bounded experiment; never enqueue review jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path)
    source.add_argument("--version-id", type=int)
    parser.add_argument("--document-id", type=int)
    parser.add_argument("--pages", default="4", help="1-based comma-separated pages")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--tessdata", type=Path)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--min-confidence", type=float, default=75)
    parser.add_argument("--output", type=Path, default=Path("layout_probe_output"))
    args = parser.parse_args()
    executable = shutil.which(args.tesseract)
    if not executable:
        parser.error("Tesseract was not found; pass its full path with --tesseract")
    executable = str(Path(executable).resolve())
    tessdata = args.tessdata
    adjacent = Path(executable).parent / "tessdata"
    if tessdata is None and adjacent.is_dir():
        tessdata = adjacent
    config = OcrConfig(executable=executable, language=args.language,
                       dpi=args.dpi, min_confidence=args.min_confidence,
                       tessdata=str(tessdata) if tessdata else None)
    pdf_path = _resolve_pdf(args)
    if pdf_path.stat().st_size > 10 * 1024 * 1024:
        parser.error("Probe input must not exceed 10 MiB")
    pages = sorted({int(number.strip()) for number in args.pages.split(",")})
    with fitz.open(pdf_path) as document:
        if document.needs_pass:
            parser.error("Password-protected PDFs are not supported")
        if not 1 <= len(document) <= 80:
            parser.error("Probe input must contain 1 to 80 pages")
        if not pages or min(pages) < 1 or max(pages) > len(document):
            parser.error("Requested page number is outside this PDF")
        args.output.mkdir(parents=True, exist_ok=True)
        for number in pages:
            page = document[number - 1]
            layout = extract_page_layout(page, config)
            json_path = args.output / f"page_{number:03d}.json"
            image_path = args.output / f"page_{number:03d}_overlay.png"
            json_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            _save_overlay(page, layout, image_path)
            print(f"Page {number}: {layout['counts']}")
            for block in layout["blocks"]:
                if block["source"] == "ocr":
                    print(f"  OCR candidate: {block['text']!r}")
            print(f"  JSON: {json_path.resolve()}")
            print(f"  Overlay: {image_path.resolve()}")


if __name__ == "__main__":
    main()
