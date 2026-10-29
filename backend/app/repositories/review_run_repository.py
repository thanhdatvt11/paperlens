"""Database operations for durable review jobs and snapshots."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.document import (
    ReviewJobType,
    ReviewRun,
    ReviewRunChunk,
    ReviewRunChunkStatus,
    ReviewRunStage,
    ReviewRunStatus,
)


@dataclass(frozen=True)
class ChunkSnapshotInput:
    """Chunk data copied into a durable review run."""

    source_chunk_id: int
    chunk_index: int
    page_number: int
    text: str


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def create_run(
    db: Session,
    document_id: int,
    version_id: int,
    instruction: str,
    timeout_seconds: int,
    job_type: ReviewJobType,
    chunks: list[ChunkSnapshotInput],
) -> ReviewRun:
    """Insert a queued run and its immutable chunk snapshots."""
    run = ReviewRun(
        document_id=document_id,
        version_id=version_id,
        chunk_id=(chunks[0].source_chunk_id if len(chunks) == 1 else None),
        instruction=instruction,
        job_type=job_type.value,
        status=ReviewRunStatus.QUEUED.value,
        stage=ReviewRunStage.QUEUED.value,
        total_chunks=len(chunks),
        processed_chunks=0,
        timeout_seconds=timeout_seconds,
    )
    db.add(run)
    db.flush()
    db.add_all(
        ReviewRunChunk(
            review_run_id=run.id,
            source_chunk_id=chunk.source_chunk_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            text=chunk.text,
            status=ReviewRunChunkStatus.QUEUED.value,
        )
        for chunk in chunks
    )
    db.flush()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: int) -> ReviewRun | None:
    """Find one review run by its database-assigned ID."""
    return db.get(ReviewRun, run_id)


def list_run_chunks(db: Session, run_id: int) -> list[ReviewRunChunk]:
    """Return snapshots in their original chunk order."""
    statement = (
        select(ReviewRunChunk)
        .where(ReviewRunChunk.review_run_id == run_id)
        .order_by(ReviewRunChunk.chunk_index, ReviewRunChunk.id)
    )
    return list(db.scalars(statement).all())


def get_run_chunk(
    db: Session,
    run_id: int,
    snapshot_id: int,
) -> ReviewRunChunk | None:
    """Return one snapshot belonging to a run."""
    statement = select(ReviewRunChunk).where(
        ReviewRunChunk.review_run_id == run_id,
        ReviewRunChunk.id == snapshot_id,
    )
    return db.scalar(statement)


def recover_running_runs(db: Session, error_message: str) -> int:
    """Fail jobs left running by a worker process that stopped."""
    statement = (
        update(ReviewRun)
        .where(ReviewRun.status == ReviewRunStatus.RUNNING.value)
        .values(
            status=ReviewRunStatus.FAILED.value,
            finished_at=datetime.utcnow(),
            error_message=error_message,
        )
    )
    result = db.execute(statement)
    return int(result.rowcount or 0)


def claim_next_queued(db: Session) -> ReviewRun | None:
    """Atomically claim the oldest queued run."""
    candidate = db.scalars(
        select(ReviewRun)
        .where(ReviewRun.status == ReviewRunStatus.QUEUED.value)
        .order_by(ReviewRun.id)
        .limit(1)
    ).first()
    if candidate is None:
        return None

    statement = (
        update(ReviewRun)
        .where(
            ReviewRun.id == candidate.id,
            ReviewRun.status == ReviewRunStatus.QUEUED.value,
        )
        .values(
            status=ReviewRunStatus.RUNNING.value,
            stage=ReviewRunStage.READING_CHUNKS.value,
            started_at=datetime.utcnow(),
            error_message=None,
        )
    )
    result = db.execute(statement)
    if result.rowcount != 1:
        return None
    db.refresh(candidate)
    return candidate


def set_stage(db: Session, run_id: int, stage: ReviewRunStage) -> None:
    """Persist the current processing stage."""
    db.execute(
        update(ReviewRun)
        .where(ReviewRun.id == run_id, ReviewRun.status == "RUNNING")
        .values(stage=stage.value)
    )


def complete_chunk(
    db: Session,
    run_id: int,
    snapshot_id: int,
    result_json: str,
) -> None:
    """Persist one validated chunk result and advance progress atomically."""
    snapshot_update = (
        update(ReviewRunChunk)
        .where(
            ReviewRunChunk.id == snapshot_id,
            ReviewRunChunk.review_run_id == run_id,
            ReviewRunChunk.status == ReviewRunChunkStatus.QUEUED.value,
        )
        .values(
            status=ReviewRunChunkStatus.COMPLETED.value,
            result_json=result_json,
            error_message=None,
        )
    )
    result = db.execute(snapshot_update)
    if result.rowcount != 1:
        raise RuntimeError("review chunk progress update was not applied")
    db.execute(
        update(ReviewRun)
        .where(ReviewRun.id == run_id)
        .values(processed_chunks=ReviewRun.processed_chunks + 1)
    )


def fail_chunk(
    db: Session,
    run_id: int,
    snapshot_id: int,
    error_message: str,
) -> None:
    """Persist the failed state of one chunk without advancing progress."""
    db.execute(
        update(ReviewRunChunk)
        .where(
            ReviewRunChunk.id == snapshot_id,
            ReviewRunChunk.review_run_id == run_id,
            ReviewRunChunk.status == ReviewRunChunkStatus.QUEUED.value,
        )
        .values(
            status=ReviewRunChunkStatus.FAILED.value,
            error_message=error_message,
        )
    )


def complete_run(db: Session, run_id: int, result_json: str) -> None:
    """Store a successful final result for a claimed run."""
    db.execute(
        update(ReviewRun)
        .where(
            ReviewRun.id == run_id,
            ReviewRun.status == ReviewRunStatus.RUNNING.value,
        )
        .values(
            status=ReviewRunStatus.COMPLETED.value,
            stage=ReviewRunStage.FINISHED.value,
            finished_at=datetime.utcnow(),
            result_json=result_json,
            error_message=None,
        )
    )


def fail_run(db: Session, run_id: int, error_message: str) -> None:
    """Store a failure for a claimed run."""
    db.execute(
        update(ReviewRun)
        .where(
            ReviewRun.id == run_id,
            ReviewRun.status == ReviewRunStatus.RUNNING.value,
        )
        .values(
            status=ReviewRunStatus.FAILED.value,
            finished_at=datetime.utcnow(),
            error_message=error_message,
        )
    )

def clone_run_for_synthesis(db: Session, source_run_id: int) -> ReviewRun:
    """Queue a new run using complete, saved chunk reviews."""
    source = get_run(db, source_run_id)
    if source is None:
        raise ValueError("Source run does not exist")

    if source.job_type != ReviewJobType.DOCUMENT_REVIEW.value:
        raise ValueError("Source must be a whole-document review")

    if source.status != ReviewRunStatus.FAILED.value:
        raise ValueError("Source run must be FAILED")

    snapshots = list_run_chunks(db, source_run_id)
    if (
        not snapshots
        or len(snapshots) != source.total_chunks
        or source.processed_chunks != source.total_chunks
    ):
        raise ValueError("Source run has incomplete chunk progress")

    if any(
        snapshot.status != ReviewRunChunkStatus.COMPLETED.value
        or not snapshot.result_json
        for snapshot in snapshots
    ):
        raise ValueError("Some saved chunk results are missing")

    new_run = create_run(
        db=db,
        document_id=source.document_id,
        version_id=source.version_id,
        instruction=source.instruction,
        timeout_seconds=source.timeout_seconds,
        job_type=ReviewJobType.DOCUMENT_REVIEW,
        chunks=[
            ChunkSnapshotInput(
                source_chunk_id=snapshot.source_chunk_id,
                chunk_index=snapshot.chunk_index,
                page_number=snapshot.page_number,
                text=snapshot.text,
            )
            for snapshot in snapshots
        ],
    )

    new_snapshots = list_run_chunks(db, new_run.id)
    for old_snapshot, new_snapshot in zip(snapshots, new_snapshots):
        complete_chunk(
            db=db,
            run_id=new_run.id,
            snapshot_id=new_snapshot.id,
            result_json=old_snapshot.result_json,
        )

    db.refresh(new_run)
    return new_run