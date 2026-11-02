"""PDF upload and version listing for the vision harness."""

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Path,
    UploadFile,
)
from hashlib import sha256
from fastapi import Response

from app.pdf_processing import MAX_FILE_SIZE, PdfProcessingError
from app.repositories import version_repository
from app.services import vision_pdf_service

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.pdf_processing import PdfProcessingError
from app.schemas.restored_version_schema import (
    UploadVersionResponse,
    VersionResponse,
)
from app.services import document_service, vision_queue_service
from app.services.restored_review_queue_service import (
    ReviewTargetNotFoundError,
)


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Versions"],
)


@router.post(
    "/{document_id}/versions",
    response_model=UploadVersionResponse,
    status_code=201,
)
def upload_version(
    document_id: int = Path(ge=1),
    upload: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Store a PDF for page-image review without OCR or chunking."""
    try:
        version, processed = vision_queue_service.create_version(
            db,
            document_id,
            upload,
        )

        return UploadVersionResponse(
            **VersionResponse.model_validate(
                version
            ).model_dump(),
            warning_pages=processed.warning_pages,
            chunk_count=0,
        )

    except ReviewTargetNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (
        PdfProcessingError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@router.get(
    "/{document_id}/versions",
    response_model=list[VersionResponse],
)
def list_versions(
    document_id: int = Path(ge=1),
    db: Session = Depends(get_db),
):
    """List versions belonging to a document."""
    try:
        return document_service.list_versions(
            db=db,
            document_id=document_id,
        )

    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
        
@router.get(
    "/{document_id}/versions/{version_id}/file",
    response_class=Response,
)
def get_version_file(
    document_id: int = Path(ge=1),
    version_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> Response:
    """Read the original PDF for one document/version pair."""
    version = version_repository.get_version_for_document(
        db=db,
        document_id=document_id,
        version_id=version_id,
    )

    if version is None:
        raise HTTPException(
            status_code=404,
            detail="Version not found for this document.",
        )

    try:
        path = vision_pdf_service.pdf_path(
            version.storage_key
        )

        if path.stat().st_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the size limit.",
            )

        data = path.read_bytes()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Stored PDF file is missing.",
        ) from exc

    except PdfProcessingError as exc:
        raise HTTPException(
            status_code=409,
            detail="Stored PDF path is invalid.",
        ) from exc

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the size limit.",
        )

    if sha256(data).hexdigest() != version.sha256:
        raise HTTPException(
            status_code=409,
            detail="PDF differs from its stored source hash.",
        )

    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="version-{version_id}.pdf"'
            ),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )