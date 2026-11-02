"""Version-scoped keyword RAG: inspect retrieval before asking Qwen3."""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.adapters.ollama_adapter import OllamaAdapterError
from app.core.database import get_db
from app.services import rag_service

router = APIRouter(prefix="/api/v1/documents", tags=["RAG"])


class RagRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5, strict=True)


def invoke(function, db, document_id, version_id, payload):
    try:
        return function(db, document_id, version_id, payload.question, payload.top_k)
    except rag_service.RagNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except rag_service.RagNoTextError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (rag_service.RagOutputError, OllamaAdapterError) as exc:
        raise HTTPException(502, str(exc)) from exc
    except sqlite3.OperationalError as exc:
        raise HTTPException(503, "SQLite FTS5 retrieval unavailable; check server logs/runtime") from exc


@router.post("/{document_id}/versions/{version_id}/rag/search")
def search(payload: RagRequest, document_id: int = Path(ge=1),
           version_id: int = Path(ge=1), db: Session = Depends(get_db)):
    """Return source chunks without calling Ollama. BM25 is not confidence."""
    sources = invoke(rag_service.retrieve, db, document_id, version_id, payload)
    return {"document_id": document_id, "version_id": version_id,
            "retrieval_method": "sqlite_fts5_bm25_keywords", "sources": sources}


@router.post("/{document_id}/versions/{version_id}/rag/ask")
def ask(payload: RagRequest, document_id: int = Path(ge=1),
        version_id: int = Path(ge=1), db: Session = Depends(get_db)):
    """One synchronous local model call; existing review worker is not used."""
    return invoke(rag_service.answer, db, document_id, version_id, payload)
