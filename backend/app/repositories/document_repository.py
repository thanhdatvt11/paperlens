"""Database operations for documents."""

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.document import Document


def create_document(
    db: Session,
    title: str,
    description: str | None,
) -> Document:
    """Insert and load a document within the current transaction."""
    document = Document(
        title=title,
        description=description,
    )

    db.add(document)
    db.flush()
    db.refresh(document)

    return document

def list_documents(
    db: Session,
    offset: int,
    limit: int,
) -> list[Document]:
    """Return documents ordered by newest ID first."""
    statement = (
        select(Document)
        .order_by(Document.id.desc())
        .offset(offset)
        .limit(limit)
    )

    return list(db.scalars(statement).all())


def get_document(
    db: Session,
    document_id: int,
) -> Document | None:
    """Return a document by primary key, or None."""
    return db.get(Document, document_id)