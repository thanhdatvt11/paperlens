"""HTTP endpoints for documents."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Response,
    status,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document_schema import (
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
)
from app.services import document_service

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    payload: DocumentCreate,
    db: Annotated[Session, Depends(get_db)],
) -> Document:
    """Create document metadata."""
    return document_service.create_document(
        db=db,
        payload=payload,
    )


@router.get(
    "",
    response_model=list[DocumentResponse],
)
def list_documents(
    db: Annotated[Session, Depends(get_db)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[Document]:
    """Return a page of documents."""
    return document_service.list_documents(
        db=db,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: Annotated[int, Path(ge=1)],
    db: Annotated[Session, Depends(get_db)],
) -> Document:
    """Return one document by ID."""
    try:
        return document_service.get_document(
            db=db,
            document_id=document_id,
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc


@router.put(
    "/{document_id}",
    response_model=DocumentResponse,
)
def update_document(
    document_id: Annotated[int, Path(ge=1)],
    payload: DocumentUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> Document:
    """Replace a document's editable fields."""
    try:
        return document_service.update_document(
            db=db,
            document_id=document_id,
            payload=payload,
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: Annotated[int, Path(ge=1)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Delete a document."""
    try:
        document_service.delete_document(
            db=db,
            document_id=document_id,
        )
    except document_service.DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
