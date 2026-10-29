"""Route PDF pages by native-text, orientation and layout evidence.

Standalone experiment, not a PaperLens upload replacement. Keep this beside
pdf_layout_probe.py and pdf_region_probe.py. These helpers are required.
No mapping by page number, model calls, database writes, or source-PDF edits.
Routing thresholds are heuristics. Complex/uncertain pages require review.
"""

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pymupdf as fitz

from pdf_layout_probe import (
    OcrConfig,
    _native_lines,
    _resolve_pdf,
    _save_overlay,
    extract_page_layout,
)
from pdf_region_probe import group_regions

RULES_VERSION = "page-router-experiment-2"


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def analyze_page(page: fitz.Page) -> dict[str, Any]:
    """Collect routing signals; do not equate them with design defects."""
    native = _native_lines(page)
    width, height = page.cropbox.width, page.cropbox.height
    if width <= 0 or height <= 0:
        raise ValueError("Invalid page dimensions")
    orientations = []
    data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip():
                continue
            dx, dy = line.get("dir", (1.0, 0.0))
            orientations.append((math.degrees(math.atan2(dy, dx)), len(text)))
    total_chars = sum(weight for _, weight in orientations)
    upright_chars = sum(weight for angle, weight in orientations if abs(angle) <= 3)
    upright_fraction = upright_chars / total_chars if total_chars else None
    has_non_horizontal_text = any(abs(angle) > 3 for angle, _ in orientations)

    # Ignore only the top/bottom margins for layout classification. Their
    # source text/images remain in extraction; exclusion is explicitly reported.
    body = [line for line in native
            if 0.05 * height <= (line["bbox"][1] + line["bbox"][3]) / 2
            <= 0.92 * height]
    line_height = median(line["bbox"][3] - line["bbox"][1]
                         for line in body) if body else 12
    rows: list[dict[str, Any]] = []
    for line in sorted(body, key=lambda item: item["bbox"][1]):
        center_y = (line["bbox"][1] + line["bbox"][3]) / 2
        matching = next((row for row in rows
                         if abs(center_y - row["y"]) <= 0.4 * line_height), None)
        if matching is None:
            rows.append({"y": center_y, "lines": [line]})
        else:
            matching["lines"].append(line)
    parallel_rows = 0
    for row in rows:
        ordered = sorted(row["lines"], key=lambda item: item["bbox"][0])
        if any(right["bbox"][0] - left["bbox"][2]
               > max(0.025 * width, line_height)
               for left, right in zip(ordered, ordered[1:])):
            parallel_rows += 1

    image_candidates = []
    seen_images = set()
    repeated_image_rectangles = 0
    footer_images_not_classified = 0
    for image in page.get_image_info():
        rectangle = fitz.Rect(image["bbox"])
        rectangle_key = tuple(round(value, 1) for value in rectangle)
        if rectangle_key in seen_images:
            repeated_image_rectangles += 1
            continue
        seen_images.add(rectangle_key)
        if rectangle.get_area() < 0.001 * width * height:
            continue
        if rectangle.y0 >= 0.92 * height:
            footer_images_not_classified += 1
            continue
        if not any((rectangle & fitz.Rect(line["bbox"])).get_area() > 0
                   for line in native):
            image_candidates.append(list(rectangle))
    # An image without native text might be a photo/icon rather than missing
    # text. This is an OCR candidate, not a claim that the PDF is defective.
    limited_text = total_chars < 40
    orientation_uncertain = bool(page.rotation or has_non_horizontal_text)
    if orientation_uncertain:
        route = "orientation_review_required"
    elif limited_text:
        route = "ocr_layout"
    elif image_candidates:
        route = "hybrid_layout"
    elif parallel_rows:
        route = "native_layout"
    else:
        route = "native_simple_candidate"
    layout_strategy = (
        "orientation_review_required" if orientation_uncertain else
        "multiple_regions_candidate" if parallel_rows else
        "single_flow_candidate" if native else "unknown_until_ocr"
    )
    extraction_strategy = (
        "blocked_until_orientation_review" if orientation_uncertain else
        "ocr_required" if limited_text else
        "native_plus_ocr_candidates" if image_candidates else "native_text"
    )
    return {
        "page_number": page.number + 1,
        "route": route,
        "decision_is_heuristic": True,
        "strategies": {
            "text_extraction": extraction_strategy,
            "layout_handling": layout_strategy,
            "image_text_orientation": "not_assessed",
            "semantic_relationships": "not_verified",
        },
        "signals": {
            "native_characters": total_chars,
            "native_lines": len(native),
            "page_rotation_metadata": page.rotation,
            "upright_native_character_fraction": upright_fraction,
            "has_non_horizontal_native_text": has_non_horizontal_text,
            "parallel_text_rows": parallel_rows,
            "image_regions_without_native_text": image_candidates,
            "footer_images_excluded_from_routing": footer_images_not_classified,
            "repeated_image_rectangles_removed": repeated_image_rectangles,
        },
        "warnings": [
            "These are extraction signals, not findings about the PDF design.",
            "A simple candidate can still contain missed relationships or text.",
            "Native direction says nothing about the orientation of image text.",
            "Image skew, perspective, arrows and table structure are not resolved.",
            "No automatic quality score or semantic correctness is asserted.",
        ],
    }


def _native_layout(page: fitz.Page) -> dict[str, Any]:
    """Preserve all native lines, including margins, without running OCR."""
    blocks = sorted(_native_lines(page),
                    key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return {
        "page_number": page.number + 1,
        "page_size": {"width": page.rect.width, "height": page.rect.height},
        "coordinate_system": "PDF points, origin at top-left",
        "ordering": "position_sorted_not_semantic",
        "counts": {"native_lines": len(blocks), "supplementary_ocr_segments": 0},
        "blocks": blocks,
        "rejected_ocr_words": [],
        "warnings": ["OCR was not run. Image-only content may still be absent.",
                     "All native lines are retained; reading order is heuristic."],
    }


def extract_routed_page(
    page: fitz.Page, config: OcrConfig | None, output: Path,
) -> dict[str, Any]:
    """Use the selected path, preserve provenance and enforce quality gates."""
    report = analyze_page(page)
    number = page.number + 1
    stem = f"page_{number:03d}"
    route = report["route"]
    if route == "orientation_review_required":
        # The existing OCR helper does not support rotated coordinate systems.
        # Never rotate an original file or merge incompatible coordinates.
        page.get_pixmap(dpi=100, alpha=False).save(output / f"{stem}_orientation.png")
        report["status"] = "MANUAL_REVIEW_REQUIRED"
        report["review_gate"] = "blocked_orientation_or_skew"
        report["warnings"].append(
            "No deskew/rotation was applied; orientation handling is not implemented."
        )
        return report

    use_ocr = route in {"ocr_layout", "hybrid_layout"}
    if use_ocr:
        if config is None:
            raise ValueError("This page requires OCR; supply --tesseract")
        layout = extract_page_layout(page, config)
    else:
        layout = _native_layout(page)
    layout["routing"] = report
    _json_write(output / f"{stem}_layout.json", layout)
    _save_overlay(page, layout, output / f"{stem}_overlay.png")
    if not layout["blocks"]:
        report["status"] = "MANUAL_REVIEW_REQUIRED"
        report["review_gate"] = "blocked_no_reliable_text"
        return report

    regions = group_regions(layout)
    _json_write(output / f"{stem}_regions.json", regions)
    report["extraction_counts"] = layout["counts"]
    report["region_counts"] = regions["counts"]
    report["candidate_pairs"] = len(regions["relation_candidates"])
    # OCR and layout are independent. Image-only footer text does not make a
    # page a diagram; no OCR text found also does not prove an image is irrelevant.
    ocr_blocks = [block for block in layout["blocks"] if block["source"] == "ocr"]
    body_ocr = [block for block in ocr_blocks
                if block["bbox"][1] < 0.92 * page.rect.height]
    report["ocr_observation"] = {
        "body_or_title_segments_added": len(body_ocr),
        "footer_segments_added": len(ocr_blocks) - len(body_ocr),
        "image_content_proven_irrelevant": False,
        "ocr_additions_visually_verified": False,
    }
    report["text_view_candidate"] = (
        report["strategies"]["layout_handling"] == "single_flow_candidate"
        and not body_ocr
    )
    report["status"] = "EXTRACTED_NOT_SEMANTICALLY_VERIFIED"
    if route == "native_simple_candidate":
        # A candidate text representation only, never overwrite stored chunks.
        text = "\n".join(block["text"].rstrip() for block in layout["blocks"])
        (output / f"{stem}_text.txt").write_text(text, encoding="utf-8")
        report["review_gate"] = "check_simple_text_before_integration"
    else:
        report["review_gate"] = "check_ocr_and_layout_before_integration"
    return report


def main() -> None:
    """Analyze many pages, or extract selected pages sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", type=Path)
    source.add_argument("--version-id", type=int)
    parser.add_argument("--document-id", type=int)
    parser.add_argument("--pages", default="all", help="all or comma-separated page numbers")
    parser.add_argument("--analyze-only", action="store_true", help="Do not run OCR")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--output", type=Path, default=Path("auto_extract_output"))
    args = parser.parse_args()
    source_path = _resolve_pdf(args)
    if source_path.stat().st_size > 10 * 1024 * 1024:
        parser.error("PDF must not exceed 10 MiB")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new/empty --output directory to preserve old experiments")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    executable = shutil.which(args.tesseract)
    config = None
    if executable:
        executable = str(Path(executable).resolve())
        tessdata = Path(executable).parent / "tessdata"
        config = OcrConfig(executable=executable, dpi=args.dpi,
                           language=args.language,
                           tessdata=str(tessdata) if tessdata.is_dir() else None)
    reports = []
    with fitz.open(source_path) as pdf:
        if pdf.needs_pass or not 1 <= len(pdf) <= 80:
            parser.error("Expected an unencrypted PDF with 1 to 80 pages")
        selected = (list(range(1, len(pdf) + 1)) if args.pages == "all"
                    else sorted({int(value.strip()) for value in args.pages.split(",")}))
        if not selected or min(selected) < 1 or max(selected) > len(pdf):
            parser.error("Invalid page selection")
        args.output.mkdir(parents=True, exist_ok=True)
        for number in selected:
            page = pdf[number - 1]
            try:
                report = (analyze_page(page) if args.analyze_only else
                          extract_routed_page(page, config, args.output))
                if args.analyze_only:
                    report["status"] = "ANALYZED_ONLY"
            except Exception as exc:  # Record failure; never silently skip a page.
                report = {"page_number": number, "route": "failed",
                          "status": "FAILED", "error": str(exc)}
            reports.append(report)
            _json_write(args.output / f"page_{number:03d}_route.json", report)
            print(f"Page {number}: {report['route']} / {report['status']}", flush=True)

    manifest = {
        "rules_version": RULES_VERSION,
        "source_sha256": source_hash,
        "source_filename": source_path.name,
        "analyze_only": args.analyze_only,
        "route_counts": dict(Counter(report["route"] for report in reports)),
        "pages": reports,
        "automatic_semantic_review_approved": False,
        "note": "Text extraction and layout handling are separate decisions; "
                "the legacy route name alone is not a layout classification.",
    }
    _json_write(args.output / "routing_report.json", manifest)
    print("Report:", (args.output / "routing_report.json").resolve())
    print("No API/database/source-PDF changes or model calls were made.")
    print("Routing is heuristic; do not equate extracted text with verified structure.")
    if any(report["status"] == "FAILED" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
