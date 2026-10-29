"""Business operations for documents, restored from supplied source."""
from sqlalchemy.orm import Session
from fastapi import UploadFile
from app.models.document import Document, DocumentVersion
from app.pdf_processing import (
    PdfProcessingError, ProcessedPdf, process_upload, remove_storage_file,
)
from app.repositories import document_repository, version_repository
from app.schemas.document_schema import DocumentCreate, DocumentUpdate


class DocumentNotFoundError(LookupError):
    """Requested document does not exist."""


class DocumentHasVersionsError(RuntimeError):
    """Document cannot be deleted while versions exist."""


class VersionNotFoundError(LookupError):
    """Version does not belong to the requested document."""


def create_document(db: Session, payload: DocumentCreate) -> Document:
    with db.begin():
        document = document_repository.create_document(
            db=db, title=payload.title, description=payload.description,
        )
    return document


def list_documents(db: Session, offset: int, limit: int) -> list[Document]:
    return document_repository.list_documents(db=db, offset=offset, limit=limit)


def get_document(db: Session, document_id: int) -> Document:
    document = document_repository.get_document(db=db, document_id=document_id)
    if document is None:
        raise DocumentNotFoundError(f'Document {document_id} does not exist.')
    return document


def update_document(db: Session, document_id: int, payload: DocumentUpdate) -> Document:
    with db.begin():
        document = document_repository.get_document(db=db, document_id=document_id)
        if document is None:
            raise DocumentNotFoundError(f'Document {document_id} does not exist.')
        document = document_repository.update_document(
            db=db, document=document, title=payload.title, description=payload.description,
        )
    return document


def delete_document(db: Session, document_id: int) -> None:
    with db.begin():
        document = document_repository.get_document(db=db, document_id=document_id)
        if document is None:
            raise DocumentNotFoundError(f'Document {document_id} does not exist.')
        if version_repository.has_versions(db=db, document_id=document_id):
            raise DocumentHasVersionsError(f'Document {document_id} has uploaded versions.')
        document_repository.delete_document(db=db, document=document)


def create_version(db: Session, document_id: int, upload: UploadFile) -> tuple[DocumentVersion, ProcessedPdf]:
    document = document_repository.get_document(db=db, document_id=document_id)
    if document is None:
        raise DocumentNotFoundError(f'Document {document_id} does not exist.')
    db.rollback()
    processed = process_upload(upload)
    try:
        with db.begin():
            version = version_repository.create_version(
                db=db, document_id=document_id,
                original_filename=processed.original_filename,
                storage_key=processed.storage_key, sha256=processed.sha256,
                page_count=processed.page_count, chunks=processed.chunks,
            )
    except Exception:
        remove_storage_file(processed.storage_key)
        raise
    return version, processed


def list_versions(db: Session, document_id: int) -> list[DocumentVersion]:
    get_document(db=db, document_id=document_id)
    return version_repository.list_versions(db=db, document_id=document_id)


def list_chunks(db: Session, document_id: int, version_id: int, offset: int, limit: int):
    get_document(db=db, document_id=document_id)
    version = db.get(DocumentVersion, version_id)
    if version is None or version.document_id != document_id:
        raise VersionNotFoundError(f'Version {version_id} does not belong to document {document_id}.')
    return version_repository.list_chunks(db=db, version_id=version_id, offset=offset, limit=limit)
