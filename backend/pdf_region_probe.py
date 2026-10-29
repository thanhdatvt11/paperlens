"""Group a layout-probe JSON into conservative geometric text regions.

Standard-library only. No PDF changes, model calls, or database access.
The output retains every original block exactly once. Candidate links are
geometric hints, never verified title-description or semantic relationships.
"""

import argparse
import html
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any


def _height(item: dict[str, Any]) -> float:
    return item["bbox"][3] - item["bbox"][1]


def _center_x(item: dict[str, Any]) -> float:
    return (item["bbox"][0] + item["bbox"][2]) / 2


def _numeric(item: dict[str, Any]) -> bool:
    return bool(re.fullmatch(r"\d{1,3}[.)]?", item["text"].strip()))


def _validate_input(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject malformed geometry instead of guessing or dropping blocks."""
    if not isinstance(layout.get("page_number"), int):
        raise ValueError("Missing integer page_number")
    size = layout.get("page_size", {})
    if any(
        not isinstance(size.get(key), (int, float))
        or not math.isfinite(size[key])
        or size[key] <= 0
        for key in ("width", "height")
    ):
        raise ValueError("Missing valid page dimensions")
    blocks = layout.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Expected a nonempty list of layout blocks")
    if len(blocks) > 5000:
        raise ValueError("Probe supports at most 5000 blocks per page")
    result = []
    for number, block in enumerate(blocks, start=1):
        box = block.get("bbox")
        if (
            not isinstance(block.get("text"), str)
            or not block["text"].strip()
            or not isinstance(box, list)
            or len(box) != 4
            or not all(
                isinstance(value, (float, int)) and math.isfinite(value)
                for value in box
            )
            or box[2] <= box[0]
            or box[3] <= box[1]
            or block.get("source") not in {"native", "ocr"}
        ):
            raise ValueError(f"Malformed source block {number}")
        result.append({"block_id": f"B{number:04d}", "original": block,
                       "text": block["text"], "bbox": box,
                       "source": block["source"]})
    return result


def _join_cost(upper: dict[str, Any], lower: dict[str, Any]) -> float | None:
    """Identify nearby, similarly sized lines with strong alignment."""
    if upper["source"] != lower["source"] or _numeric(upper) or _numeric(lower):
        return None
    a, b = upper["bbox"], lower["bbox"]
    small, large = sorted((_height(upper), _height(lower)))
    if small / large < 0.7 or b[1] - a[1] < 0.6 * small:
        return None
    gap = b[1] - a[3]
    if gap < -0.15 * small or gap > 1.4 * large:
        return None
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    if overlap / min(a[2] - a[0], b[2] - b[0]) < 0.55:
        return None
    center_distance = abs(_center_x(upper) - _center_x(lower))
    left_distance = abs(a[0] - b[0])
    if center_distance > 0.75 * large and left_distance > 0.3 * large:
        return None
    alignment = min(center_distance, left_distance) / large
    return max(0, gap) / large + alignment


def _unique_best(options: list[tuple[float, int]]) -> int | None:
    """Keep ambiguous alternatives separate when scores are similar."""
    options.sort()
    if not options or (len(options) > 1 and options[1][0] - options[0][0] < 0.25):
        return None
    return options[0][1]


def _make_region(
    members: list[dict[str, Any]], number: int,
) -> dict[str, Any]:
    """Preserve membership independently of the derived display text."""
    return {
        "region_id": f"R{number:03d}",
        "text": "\n".join(item["text"].rstrip() for item in members),
        "bbox": [min(item["bbox"][0] for item in members),
                 min(item["bbox"][1] for item in members),
                 max(item["bbox"][2] for item in members),
                 max(item["bbox"][3] for item in members)],
        "block_ids": [item["block_id"] for item in members],
        "sources": sorted({item["source"] for item in members}),
        "line_height": median(_height(item) for item in members),
        "needs_review": any(
            item["source"] == "ocr" or item["original"].get("needs_review", False)
            for item in members
        ),
        "grouping": "geometric_heuristic",
    }


def _candidate_links(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find mutual-nearest vertically aligned regions; infer no semantics."""
    options: dict[int, list[tuple[float, int]]] = {
        index: [] for index in range(len(regions))
    }
    for i, upper in enumerate(regions):
        if _numeric(upper):
            continue
        for j in range(i + 1, len(regions)):
            lower = regions[j]
            if _numeric(lower) or upper["sources"] != lower["sources"]:
                continue
            gap = lower["bbox"][1] - upper["bbox"][3]
            height = max(upper["line_height"], lower["line_height"])
            distance = abs(_center_x(upper) - _center_x(lower))
            if not 0 <= gap <= 3.5 * height or distance > height:
                continue
            cost = gap / height + distance / height
            options[i].append((cost, j))
            options[j].append((cost, i))
    nearest = {index: _unique_best(items) for index, items in options.items()}
    return [
        {
            "region_ids": [regions[index]["region_id"], regions[other]["region_id"]],
            "basis": "mutual_nearest_vertical_alignment",
            "status": "candidate_only",
            "semantic_relationship_verified": False,
            "requires_visual_check": True,
        }
        for index, other in nearest.items()
        if other is not None and index < other and nearest[other] == index
    ]


def group_regions(layout: dict[str, Any]) -> dict[str, Any]:
    """Group line chains while maintaining one-to-one source coverage."""
    blocks = _validate_input(layout)
    successors = {index: [] for index in range(len(blocks))}
    predecessors = {index: [] for index in range(len(blocks))}
    for i, upper in enumerate(blocks):
        for j, lower in enumerate(blocks):
            if i == j:
                continue
            cost = _join_cost(upper, lower)
            if cost is not None:
                successors[i].append((cost, j))
                predecessors[j].append((cost, i))
    next_best = {index: _unique_best(items) for index, items in successors.items()}
    previous_best = {index: _unique_best(items) for index, items in predecessors.items()}
    links = {i: j for i, j in next_best.items()
             if j is not None and previous_best[j] == i}
    targets = set(links.values())
    chains = []
    visited = set()
    for start in sorted(
        set(range(len(blocks))) - targets,
        key=lambda index: (blocks[index]["bbox"][1], blocks[index]["bbox"][0]),
    ):
        members = []
        current = start
        while current not in visited:
            visited.add(current)
            members.append(blocks[current])
            if current not in links:
                break
            current = links[current]
        chains.append(members)
    if len(visited) != len(blocks):
        raise RuntimeError("Grouping did not account for every source block")
    regions = [_make_region(members, number)
               for number, members in enumerate(chains, start=1)]
    assigned = [block_id for region in regions for block_id in region["block_ids"]]
    if len(assigned) != len(set(assigned)) or len(assigned) != len(blocks):
        raise RuntimeError("Grouping duplicated or lost a source block")
    return {
        "page_number": layout["page_number"],
        "page_size": layout["page_size"],
        "coordinate_system": layout.get("coordinate_system"),
        "counts": {"source_blocks": len(blocks), "regions": len(regions)},
        "coverage": {"all_blocks_assigned_once": True,
                     "semantic_coverage_verified": False},
        "regions": regions,
        "relation_candidates": _candidate_links(regions),
        "source_blocks": {item["block_id"]: item["original"] for item in blocks},
        "source_extraction": {
            "settings": layout.get("settings"),
            "warnings": layout.get("warnings", []),
            "rejected_ocr_words": layout.get("rejected_ocr_words", []),
        },
        "warnings": [
            "Line grouping and region links are geometric heuristics.",
            "A candidate link does not establish a title-description relation.",
            "Original source blocks, including footer and OCR text, are retained.",
            "Derived region text is not an original PDF quote; resolve block_ids.",
            "No arrows, table cells, numbered sequences, or column semantics parsed.",
            "Do not feed the entire JSON into a 4096-token model context.",
        ],
    }


def _write_html(result: dict[str, Any], destination: Path) -> None:
    """Write a self-contained region table for manual checking."""
    rows = []
    for region in result["regions"]:
        rows.append(
            "<tr><td>" + region["region_id"] + "</td><td><pre>"
            + html.escape(region["text"]) + "</pre></td><td>"
            + html.escape(", ".join(region["block_ids"])) + "</td><td>"
            + html.escape(", ".join(region["sources"])) + "</td><td>"
            + ", ".join(f"{value:.1f}" for value in region["bbox"])
            + "</td></tr>"
        )
    pairs = []
    lookup = {region["region_id"]: region for region in result["regions"]}
    for pair in result["relation_candidates"]:
        left, right = pair["region_ids"]
        pairs.append("<li><b>" + left + " / " + right + "</b><pre>"
                     + html.escape(lookup[left]["text"]) + "\n---\n"
                     + html.escape(lookup[right]["text"]) + "</pre></li>")
    document = """<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Kiểm tra gom vùng PDF</title><style>
body{font:16px system-ui;margin:32px;color:#172033;background:#f6f8fc}
table{border-collapse:collapse;background:white;width:100%}
td,th{border:1px solid #cbd5e1;padding:10px;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;font:inherit;margin:0}li{margin:16px 0}
.note{padding:16px;background:#fff2cd;border-radius:8px}
</style><h1>Kiểm tra gom vùng — trang PAGE</h1>
<p class="note">Các cặp bên dưới chỉ gần nhau theo hình học, chưa xác nhận
quan hệ tên–mô tả. Vùng OCR vẫn cần đối chiếu ảnh gốc.
Đây không phải bản dựng lại bố cục PDF.</p>
<h2>Cặp vùng cần đối chiếu</h2><ul>PAIRS</ul>
<h2>Tất cả vùng và nguồn</h2><table><thead><tr>
<th>ID</th><th>Nội dung</th><th>Dòng nguồn</th><th>Loại nguồn</th><th>Bbox</th>
</tr></thead><tbody>ROWS</tbody></table></html>"""
    substitutions = {"PAGE": str(result["page_number"]),
                     "PAIRS": "".join(pairs), "ROWS": "".join(rows)}
    document = re.sub(r"\b(PAGE|PAIRS|ROWS)\b",
                      lambda match: substitutions[match.group(0)], document)
    destination.write_text(document, encoding="utf-8")


def main() -> None:
    """Read one layout JSON, write separate region JSON and HTML files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("region_probe_output"))
    args = parser.parse_args()
    if args.input.stat().st_size > 20 * 1024 * 1024:
        parser.error("Layout JSON must not exceed 20 MiB")
    result = group_regions(json.loads(args.input.read_text(encoding="utf-8-sig")))
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"page_{result['page_number']:03d}_regions"
    json_path = args.output / f"{stem}.json"
    html_path = args.output / f"{stem}.html"
    if args.input.resolve() in {json_path.resolve(), html_path.resolve()}:
        parser.error("Output must not overwrite the input file")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    _write_html(result, html_path)
    print(f"Page {result['page_number']}: {result['counts']}")
    print(f"Geometric candidate pairs: {len(result['relation_candidates'])}")
    print("Every source block is assigned exactly once; semantics unverified.")
    print(f"JSON: {json_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")


if __name__ == "__main__":
    main()
