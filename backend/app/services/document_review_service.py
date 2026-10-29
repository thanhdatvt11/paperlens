"""Review all chunks, then discover and summarize source-backed sections."""
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
from typing import Any, Callable

from app.adapters.ollama_adapter import OllamaAdapter
from app.models.document import ReviewRunChunk, ReviewRunStage
from app.repositories import review_run_repository
from app.schemas.ai_schema import (
    ChunkReviewPreviewResponse, DocumentFinding, DocumentReviewContent, ReviewEvidence,
)
from app.schemas.structure_schema import StructuredDocumentReviewResult
from app.services import ai_service, document_structure_service

LOGGER = logging.getLogger(__name__)

def _finding_id(finding: DocumentFinding) -> str:
    """Build a stable reference for a complete validated finding."""
    serialized = json.dumps(
        finding.model_dump(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    return f"F{digest[:16]}"


@dataclass(frozen=True)
class SnapshotContext:
    """A durable chunk snapshot copied out of the ORM session."""

    snapshot_id: int
    chunk_id: int
    chunk_index: int
    page_number: int
    text: str


def load_snapshot_contexts(
    session_factory: Callable[..., Any],
    run_id: int,
) -> list[SnapshotContext]:
    """Read snapshots in a short transaction before model calls."""
    with session_factory() as db:
        with db.begin():
            snapshots = review_run_repository.list_run_chunks(db, run_id)
            return [_snapshot_context(snapshot) for snapshot in snapshots]


def _snapshot_context(snapshot: ReviewRunChunk) -> SnapshotContext:
    """Copy a snapshot ORM row into plain immutable data."""
    return SnapshotContext(
        snapshot_id=snapshot.id,
        chunk_id=snapshot.source_chunk_id,
        chunk_index=snapshot.chunk_index,
        page_number=snapshot.page_number,
        text=snapshot.text,
    )


def _load_cached_results(
    session_factory: Callable[..., Any],
    run_id: int,
    document_id: int,
    version_id: int,
) -> dict[int, dict[str, Any]]:
    """Validate saved results before reusing them for synthesis."""
    with session_factory() as db:
        with db.begin():
            snapshots = review_run_repository.list_run_chunks(db, run_id)
            rows = [
                (
                    snapshot.source_chunk_id,
                    snapshot.page_number,
                    snapshot.text,
                    snapshot.status,
                    snapshot.result_json,
                )
                for snapshot in snapshots
            ]

    cached: dict[int, dict[str, Any]] = {}
    for chunk_id, page_number, text, status, result_json in rows:
        if status != "COMPLETED":
            continue
        if not result_json:
            raise RuntimeError(
                f"Saved result is missing for chunk_id={chunk_id}"
            )

        result = ChunkReviewPreviewResponse.model_validate_json(result_json)
        if (
            result.document_id != document_id
            or result.version_id != version_id
            or result.chunk_id != chunk_id
            or result.page_number != page_number
        ):
            raise RuntimeError(
                f"Saved result metadata differs for chunk_id={chunk_id}"
            )

        normalized_text = ai_service.normalize_for_quote(text)
        for finding in result.findings:
            quote = ai_service.normalize_for_quote(finding.quote)
            if not quote or quote not in normalized_text:
                raise RuntimeError(
                    "Saved result has invalid evidence for "
                    f"chunk_id={chunk_id}"
                )

        cached[chunk_id] = result.model_dump()

    return cached


def review_document_run(
    session_factory: Callable[..., Any],
    run_id: int,
    document_id: int,
    version_id: int,
    instruction: str,
    adapter: OllamaAdapter,
) -> dict[str, Any]:
    """Reuse saved chunk reviews, then synthesize the document."""
    contexts = load_snapshot_contexts(session_factory, run_id)
    if not contexts:
        raise RuntimeError("document review has no chunk snapshots")

    cached_results = _load_cached_results(
        session_factory=session_factory,
        run_id=run_id,
        document_id=document_id,
        version_id=version_id,
    )

    chunk_results: list[dict[str, Any]] = []

    for context in contexts:
        cached = cached_results.get(context.chunk_id)
        if cached is not None:
            chunk_results.append(cached)
            LOGGER.info(
                "document_review_reuse_chunk run_id=%s chunk_id=%s",
                run_id,
                context.chunk_id,
            )
            continue

        chunk_context = ai_service.ChunkReviewContext(
            document_id=document_id,
            version_id=version_id,
            chunk_id=context.chunk_id,
            page_number=context.page_number,
            text=context.text,
        )

        try:
            result = ai_service.review_chunk_context(
                context=chunk_context,
                adapter=adapter,
                instruction=instruction,
            )
        except Exception as exc:
            with session_factory() as db:
                with db.begin():
                    review_run_repository.fail_chunk(
                        db=db,
                        run_id=run_id,
                        snapshot_id=context.snapshot_id,
                        error_message=str(exc),
                    )
            raise RuntimeError(
                "chunk review failed for "
                f"chunk_id={context.chunk_id}, "
                f"page={context.page_number}"
            ) from exc

        with session_factory() as db:
            with db.begin():
                review_run_repository.complete_chunk(
                    db=db,
                    run_id=run_id,
                    snapshot_id=context.snapshot_id,
                    result_json=json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                )

        chunk_results.append(result)

    with session_factory() as db:
        with db.begin():
            review_run_repository.set_stage(
                db,
                run_id,
                ReviewRunStage.SYNTHESIZING,
            )

    try:
        summary, model, structure = document_structure_service.analyze_and_summarize(
            chunks=[document_structure_service.StructureChunk(
                chunk_id=context.chunk_id, chunk_index=context.chunk_index,
                page_number=context.page_number, text=context.text,
            ) for context in contexts],
            adapter=adapter,
            run_id=run_id,
        )
    except Exception as exc:
        raise RuntimeError(f"document structure/summary failed: {exc}") from exc

    findings = _saved_findings(contexts, chunk_results)
    # Key points expose actual section titles, not model-invented citations.
    key_points = [
        {"point": section.title,
         "sources": [{"chunk_id": evidence.chunk_id,
                      "page_number": evidence.page_number}
                     for evidence in section.heading_evidence]}
        for section in structure.sections
        if section.kind == "body" and section.parent_id is None
        and section.title_origin == "source" and section.heading_evidence
    ]
    result = StructuredDocumentReviewResult.model_validate({
        "summary": summary, "key_points": key_points, "findings": findings,
        "document_id": document_id, "version_id": version_id, "model": model,
        "structure": structure.model_dump(),
    })
    return result.model_dump()


def _deduplicate_findings(
    content: DocumentReviewContent,
) -> DocumentReviewContent:
    """Remove only exact duplicate findings without inventing content."""
    unique: list[DocumentFinding] = []
    seen: set[str] = set()
    for finding in content.findings:
        key = json.dumps(
            finding.model_dump(), ensure_ascii=False, sort_keys=True
        )
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return content.model_copy(update={"findings": unique})


def _saved_findings(
    contexts: list[SnapshotContext], results: list[dict[str, Any]],
) -> list[DocumentFinding]:
    """Keep complete original findings; never ask synthesis to rewrite quotes."""
    registry: dict[str, DocumentFinding] = {}
    for context, result in zip(contexts, results, strict=True):
        for item in result['findings']:
            finding = DocumentFinding(
                severity=item['severity'], comment=item['comment'],
                suggestion=item['suggestion'],
                evidence=[ReviewEvidence(
                    chunk_id=context.chunk_id, page_number=context.page_number,
                    quote=item['quote'],
                )],
            )
            registry.setdefault(_finding_id(finding), finding)
    # This is a server ordering rule, not a model importance judgment.
    ordered = sorted(registry.values(), key=lambda f: 0 if f.severity == 'major' else 1)
    return ordered[:10]
