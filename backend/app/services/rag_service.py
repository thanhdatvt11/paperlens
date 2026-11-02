"""Read original chunks in a short transaction before local model calls."""

from sqlalchemy import select

from app.adapters.ollama_adapter import OllamaAdapter
from app.core.config import get_settings
from app.models.document import DocumentChunk, DocumentVersion
from app.services import rag_core


class RagNotFoundError(LookupError):
    """The requested version does not belong to this document."""


class RagNoTextError(ValueError):
    """Vision-only versions need original text extraction before text RAG."""


class RagOutputError(ValueError):
    """Provider output is incomplete, malformed or has invalid citations."""


def retrieve(db, document_id, version_id, question, top_k):
    with db.begin():
        version = db.scalar(select(DocumentVersion).where(
            DocumentVersion.id == version_id,
            DocumentVersion.document_id == document_id,
        ))
        if version is None:
            raise RagNotFoundError("Version does not belong to this document")
        rows = db.scalars(select(DocumentChunk).where(
            DocumentChunk.document_version_id == version_id
        ).order_by(DocumentChunk.chunk_index, DocumentChunk.id)).all()
        chunks = [dict(chunk_id=row.id, page_number=row.page_number,
                       chunk_index=row.chunk_index, text=row.text)
                  for row in rows if row.text.strip()]
    if not chunks:
        raise RagNoTextError(
            "Version has no text chunks. Extract original PDF text/OCR first; "
            "vision summaries/findings are not original source text."
        )
    return rag_core.search(chunks, question, top_k)


def answer(db, document_id, version_id, question, top_k, adapter=None):
    retrieved = retrieve(db, document_id, version_id, question, top_k)
    sources = rag_core.fit_sources(question, retrieved)
    metadata = dict(document_id=document_id, version_id=version_id,
                    retrieval_method="sqlite_fts5_bm25_keywords",
                    question=question, requested_top_k=top_k,
                    retrieved_count=len(retrieved), context_source_count=len(sources),
                    semantic_correctness_verified=False,
                    warnings=["Keyword retrieval may miss synonyms or cross-language matches.",
                              "Quotes refer to extracted text, not visually verified PDF text.",
                              "Valid citations do not prove that claims follow from evidence."])
    if not sources:
        return dict(metadata, status="insufficient_evidence", claims=[], sources=[],
                    answer="Chưa tìm được đoạn nguồn phù hợp trong giới hạn ngữ cảnh.",
                    model=None, citations_valid=True)
    adapter = adapter or OllamaAdapter(get_settings())
    provider = adapter.chat(
        rag_core.prompt_for(question, sources), system_prompt=rag_core.SYSTEM,
        response_format=rag_core.FORMAT, num_predict=768, temperature=0,
    )
    if provider.get("done") is False or provider.get("done_reason") in {"length", "max_tokens"}:
        raise RagOutputError("RAG output was truncated; no partial answer accepted")
    try:
        parsed = rag_core.parse_answer(provider.get("content"), sources)
    except (ValueError, TypeError) as exc:
        raise RagOutputError(str(exc)) from exc
    used = {claim["source_id"] for claim in parsed["claims"]}
    return dict(metadata, **parsed,
                answer="\n".join(f"{c['text']} [{c['source_id']}]" for c in parsed["claims"])
                or "Các đoạn tìm được chưa đủ bằng chứng để trả lời câu hỏi.",
                sources=[source for source in sources if source["source_id"] in used],
                model=provider.get("model"), citations_valid=True)
