"""Vision upload/review/resume endpoints, using the existing review worker."""

from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.pdf_processing import PdfProcessingError
from app.schemas.ai_schema import ReviewRunAcceptedResponse
from app.schemas.restored_version_schema import UploadVersionResponse, VersionResponse
from app.schemas.vision_schema import VisionReviewRequest
from app.services import vision_queue_service as queue
from app.services.restored_review_queue_service import ReviewTargetNotFoundError

router = APIRouter(prefix="/api/v1", tags=["Vision review"])


@router.post("/documents/{document_id}/vision-versions", response_model=UploadVersionResponse, status_code=201)
def upload_vision_version(document_id: int = Path(ge=1), upload: UploadFile = File(...), db: Session = Depends(get_db)):
    """Store a PDF without text extraction/OCR; it can be reviewed by vision."""
    try:
        version, processed = queue.create_version(db, document_id, upload)
        return UploadVersionResponse(**VersionResponse.model_validate(version).model_dump(),
                                     warning_pages=processed.warning_pages, chunk_count=0)
    except ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (PdfProcessingError, RuntimeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/documents/{document_id}/versions/{version_id}/vision-review", response_model=ReviewRunAcceptedResponse, status_code=202)
def enqueue_vision_review(payload: VisionReviewRequest, document_id: int = Path(ge=1), version_id: int = Path(ge=1), db: Session = Depends(get_db)):
    """pages omitted means all PDF pages; page positions are server-owned."""
    try:
        return queue.enqueue(db, document_id, version_id, payload)
    except ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (PdfProcessingError, ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/review-runs/{run_id}/resume", response_model=ReviewRunAcceptedResponse, status_code=202)
def resume_vision_review(run_id: int = Path(ge=1), db: Session = Depends(get_db)):
    """Requeue a FAILED vision run while retaining successful page steps."""
    try:
        return queue.resume(db, run_id)
    except ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/review-runs/{run_id}/pages")
def get_page_results(run_id: int = Path(ge=1), db: Session = Depends(get_db)):
    """Read saved page reviews/checks, including after a partial failure."""
    try:
        return queue.page_results(db, run_id)
    except ReviewTargetNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

