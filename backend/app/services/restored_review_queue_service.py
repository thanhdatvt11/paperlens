"""Durable enqueue/status operations rebuilt around the existing repository."""
import json
from sqlalchemy.orm import Session
from app.models.document import ReviewJobType
from app.repositories import review_run_repository, version_repository
from app.schemas.ai_schema import ReviewRunStatusResponse


class ReviewTargetNotFoundError(LookupError):
    pass


class NoReviewChunksError(ValueError):
    pass


def enqueue_review(
    db: Session, document_id: int, version_id: int,
    instruction: str, chunk_id: int | None = None,
) -> dict:
    with db.begin():
        version = version_repository.get_version_for_document(db, document_id, version_id)
        if version is None:
            raise ReviewTargetNotFoundError('Document/version relationship was not found.')
        if chunk_id is None:
            chunks = version_repository.list_text_chunks(db, version_id)
            job_type, timeout = ReviewJobType.DOCUMENT_REVIEW, 7200
        else:
            chunk = version_repository.get_chunk_for_document_version(db, document_id, version_id, chunk_id)
            if chunk is None:
                raise ReviewTargetNotFoundError('Chunk does not belong to this document/version.')
            chunks = [chunk] if chunk.text.strip() else []
            job_type, timeout = ReviewJobType.CHUNK_PREVIEW, 600
        if not chunks:
            raise NoReviewChunksError('No text-bearing chunks are available for review.')
        snapshots = [review_run_repository.ChunkSnapshotInput(
            source_chunk_id=chunk.id, chunk_index=chunk.chunk_index,
            page_number=chunk.page_number, text=chunk.text,
        ) for chunk in chunks]
        run = review_run_repository.create_run(
            db=db, document_id=document_id, version_id=version_id,
            instruction=instruction, timeout_seconds=timeout,
            job_type=job_type, chunks=snapshots,
        )
        run_id = run.id
    return {'run_id': run_id, 'status': 'QUEUED'}


def get_review_run(db: Session, run_id: int) -> ReviewRunStatusResponse:
    run = review_run_repository.get_run(db, run_id)
    if run is None:
        raise ReviewTargetNotFoundError('Review run was not found.')
    result = json.loads(run.result_json) if run.result_json else None
    return ReviewRunStatusResponse(
        run_id=run.id, job_type=run.job_type, document_id=run.document_id,
        version_id=run.version_id, status=run.status, stage=run.stage,
        total_chunks=run.total_chunks, processed_chunks=run.processed_chunks,
        created_at=run.created_at, started_at=run.started_at,
        finished_at=run.finished_at, result=result, error=run.error_message,
    )
