"""Document/version/chunk and durable review models restored from supplied source."""
from datetime import datetime
from enum import Enum
from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Document(Base):
    __tablename__ = 'documents'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class DocumentVersion(Base):
    __tablename__ = 'document_versions'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey('documents.id'), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)


class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    __table_args__ = (UniqueConstraint('document_version_id', 'chunk_index', name='uq_document_chunks_version_index'),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_version_id: Mapped[int] = mapped_column(ForeignKey('document_versions.id'), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    page_number: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)


class ReviewRunStatus(str, Enum):
    QUEUED = 'QUEUED'
    RUNNING = 'RUNNING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'


class ReviewJobType(str, Enum):
    CHUNK_PREVIEW = 'CHUNK_PREVIEW'
    DOCUMENT_REVIEW = 'DOCUMENT_REVIEW'


class ReviewRunStage(str, Enum):
    QUEUED = 'QUEUED'
    READING_CHUNKS = 'READING_CHUNKS'
    SYNTHESIZING = 'SYNTHESIZING'
    FINISHED = 'FINISHED'


class ReviewRun(Base):
    __tablename__ = 'review_runs'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey('documents.id'), nullable=False, index=True)
    version_id: Mapped[int] = mapped_column(ForeignKey('document_versions.id'), nullable=False, index=True)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey('document_chunks.id'), nullable=True, index=True)
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    job_type: Mapped[str] = mapped_column(String(30), nullable=False, default=ReviewJobType.CHUNK_PREVIEW.value)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=ReviewRunStatus.QUEUED.value, index=True)
    stage: Mapped[str] = mapped_column(String(30), nullable=False, default=ReviewRunStage.QUEUED.value)
    total_chunks: Mapped[int] = mapped_column(nullable=False, default=1)
    processed_chunks: Mapped[int] = mapped_column(nullable=False, default=0)
    timeout_seconds: Mapped[int] = mapped_column(nullable=False, default=180)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ReviewRunChunkStatus(str, Enum):
    QUEUED = 'QUEUED'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'


class ReviewRunChunk(Base):
    __tablename__ = 'review_run_chunks'
    __table_args__ = (UniqueConstraint('review_run_id', 'chunk_index', name='uq_review_run_chunks_run_index'),)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    review_run_id: Mapped[int] = mapped_column(ForeignKey('review_runs.id'), nullable=False, index=True)
    source_chunk_id: Mapped[int] = mapped_column(nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    page_number: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=ReviewRunChunkStatus.QUEUED.value)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
