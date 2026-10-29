"""Database operations for document versions and chunks."""
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models.document import DocumentChunk, DocumentVersion
from app.pdf_processing import ExtractedChunk


def create_version(
    db: Session, document_id: int, original_filename: str,
    storage_key: str, sha256: str, page_count: int,
    chunks: list[ExtractedChunk],
) -> DocumentVersion:
    version = DocumentVersion(
        document_id=document_id, original_filename=original_filename,
        storage_key=storage_key, sha256=sha256, page_count=page_count,
    )
    db.add(version)
    db.flush()
    db.add_all(
        DocumentChunk(
            document_version_id=version.id, chunk_index=index,
            page_number=chunk.page_number, text=chunk.text,
        ) for index, chunk in enumerate(chunks)
    )
    db.flush()
    db.refresh(version)
    return version


def list_versions(db: Session, document_id: int) -> list[DocumentVersion]:
    statement = select(DocumentVersion).where(
        DocumentVersion.document_id == document_id
    ).order_by(DocumentVersion.id.desc())
    return list(db.scalars(statement).all())


def get_version_for_document(db: Session, document_id: int, version_id: int) -> DocumentVersion | None:
    return db.scalar(select(DocumentVersion).where(
        DocumentVersion.id == version_id,
        DocumentVersion.document_id == document_id,
    ))


def count_chunks(db: Session, version_id: int) -> int:
    return int(db.scalar(select(func.count()).where(
        DocumentChunk.document_version_id == version_id
    )) or 0)


def list_chunks(db: Session, version_id: int, offset: int, limit: int) -> list[DocumentChunk]:
    statement = select(DocumentChunk).where(
        DocumentChunk.document_version_id == version_id
    ).order_by(DocumentChunk.chunk_index, DocumentChunk.id).offset(offset).limit(limit)
    return list(db.scalars(statement).all())


def list_text_chunks(db: Session, version_id: int) -> list[DocumentChunk]:
    statement = select(DocumentChunk).where(
        DocumentChunk.document_version_id == version_id
    ).order_by(DocumentChunk.chunk_index, DocumentChunk.id)
    return [chunk for chunk in db.scalars(statement).all() if chunk.text.strip()]


def get_chunk_for_document_version(
    db: Session, document_id: int, version_id: int, chunk_id: int,
) -> DocumentChunk | None:
    statement = select(DocumentChunk).join(
        DocumentVersion, DocumentVersion.id == DocumentChunk.document_version_id,
    ).where(
        DocumentVersion.document_id == document_id,
        DocumentVersion.id == version_id,
        DocumentChunk.id == chunk_id,
    )
    return db.scalar(statement)


def has_versions(db: Session, document_id: int) -> bool:
    statement = select(DocumentVersion.id).where(
        DocumentVersion.document_id == document_id
    ).limit(1)
    return db.scalar(statement) is not None
