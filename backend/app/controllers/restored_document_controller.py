"""Restore version upload, chunk browsing, Ollama access and queued review routes."""
from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy.orm import Session

from app.adapters.ollama_adapter import (
    OllamaAdapter, OllamaConnectionError, OllamaTimeoutError,
    OllamaModelUnavailableError, OllamaAdapterError,
)
from app.core.config import get_settings
from app.core.database import get_db
from app.pdf_processing import PdfProcessingError
from app.schemas.ai_schema import (
    AIModelsResponse, AITestRequest, AITestResponse,
    ChunkReviewPreviewRequest, DocumentReviewRequest,
    ReviewRunAcceptedResponse, ReviewRunStatusResponse,
)
from app.schemas.restored_version_schema import VersionResponse, UploadVersionResponse, ChunkResponse
from app.services import ai_service, document_service, restored_review_queue_service as queue

router = APIRouter(prefix='/api/v1')


def get_adapter() -> OllamaAdapter:
    return OllamaAdapter(get_settings())


def _provider_error(exc: OllamaAdapterError) -> HTTPException:
    if isinstance(exc, OllamaTimeoutError):
        return HTTPException(504, str(exc) or 'Ollama request timed out.')
    if isinstance(exc, (OllamaConnectionError, OllamaModelUnavailableError)):
        return HTTPException(503, str(exc) or 'Ollama or its configured model is unavailable.')
    return HTTPException(502, str(exc) or 'Ollama returned an invalid or unsuccessful response.')


@router.get('/ai/models', response_model=AIModelsResponse, tags=['AI'])
def list_models(adapter: OllamaAdapter = Depends(get_adapter)):
    try:
        return {'models': ai_service.list_models(adapter)}
    except OllamaAdapterError as exc:
        raise _provider_error(exc) from exc


@router.post('/ai/test', response_model=AITestResponse, tags=['AI'])
def test_chat(payload: AITestRequest, adapter: OllamaAdapter = Depends(get_adapter)):
    try:
        response = ai_service.test_chat(adapter, payload.message)
        return {'model': response['model'], 'content': response['content']}
    except OllamaAdapterError as exc:
        raise _provider_error(exc) from exc


@router.post('/documents/{document_id}/versions', response_model=UploadVersionResponse, status_code=201, tags=['Versions'])
def upload_version(document_id: int = Path(ge=1), upload: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        version, processed = document_service.create_version(db, document_id, upload)
        return UploadVersionResponse(
            **VersionResponse.model_validate(version).model_dump(),
            warning_pages=processed.warning_pages, chunk_count=len(processed.chunks),
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PdfProcessingError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get('/documents/{document_id}/versions', response_model=list[VersionResponse], tags=['Versions'])
def list_versions(document_id: int = Path(ge=1), db: Session = Depends(get_db)):
    try:
        return document_service.list_versions(db, document_id)
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get('/documents/{document_id}/versions/{version_id}/chunks', response_model=list[ChunkResponse], tags=['Versions'])
def list_chunks(document_id: int = Path(ge=1), version_id: int = Path(ge=1), offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    try:
        return document_service.list_chunks(db, document_id, version_id, offset, limit)
    except (document_service.DocumentNotFoundError, document_service.VersionNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post('/documents/{document_id}/versions/{version_id}/review', response_model=ReviewRunAcceptedResponse, status_code=202, tags=['AI review'])
def enqueue_document_review(payload: DocumentReviewRequest, document_id: int = Path(ge=1), version_id: int = Path(ge=1), db: Session = Depends(get_db)):
    try:
        return queue.enqueue_review(db, document_id, version_id, payload.instruction)
    except queue.ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except queue.NoReviewChunksError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post('/documents/{document_id}/versions/{version_id}/chunks/{chunk_id}/review-preview', response_model=ReviewRunAcceptedResponse, status_code=202, tags=['AI review'])
def enqueue_chunk_review(payload: ChunkReviewPreviewRequest, document_id: int = Path(ge=1), version_id: int = Path(ge=1), chunk_id: int = Path(ge=1), db: Session = Depends(get_db)):
    try:
        return queue.enqueue_review(db, document_id, version_id, payload.instruction, chunk_id)
    except queue.ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except queue.NoReviewChunksError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get('/review-runs/{run_id}', response_model=ReviewRunStatusResponse, tags=['AI review'])
def get_review_run(run_id: int = Path(ge=1), db: Session = Depends(get_db)):
    try:
        return queue.get_review_run(db, run_id)
    except queue.ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
