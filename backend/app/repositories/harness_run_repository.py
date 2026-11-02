"""Database operations for harness runs and events."""

from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.models.harness_run import HarnessRun, HarnessRunEvent


def create_run(
    db: Session,
    run: HarnessRun,
) -> HarnessRun:
    """Insert a queued run."""
    db.add(run)
    db.flush()
    db.refresh(run)
    return run


def get_run(
    db: Session,
    run_id: int,
) -> HarnessRun | None:
    """Find a run by ID."""
    return db.get(HarnessRun, run_id)


def claim_run(
    db: Session,
    run_id: int,
) -> HarnessRun:
    """Claim a queued run exactly once."""
    result = db.execute(
        update(HarnessRun)
        .where(
            HarnessRun.id == run_id,
            HarnessRun.status == "QUEUED",
        )
        .values(
            status="RUNNING",
            started_at=datetime.utcnow(),
            error_message=None,
        )
    )

    if result.rowcount != 1:
        raise RuntimeError(
            "Run does not exist or is not QUEUED."
        )

    run = db.get(HarnessRun, run_id)
    db.refresh(run)
    return run


def append_event(
    db: Session,
    run_id: int,
    event: dict,
    payload_json: str,
) -> None:
    """Store an event only while its run is active."""
    increment = 1 if event["event"] == "model_call" else 0

    result = db.execute(
        update(HarnessRun)
        .where(
            HarnessRun.id == run_id,
            HarnessRun.status == "RUNNING",
        )
        .values(
            step_count=HarnessRun.step_count + increment,
        )
    )

    if result.rowcount != 1:
        raise RuntimeError("Run is not RUNNING.")

    db.add(
        HarnessRunEvent(
            run_id=run_id,
            sequence=event["sequence"],
            event=event["event"],
            payload_json=payload_json,
        )
    )
    db.flush()


def next_sequence(
    db: Session,
    run_id: int,
) -> int:
    """Return the next event sequence number."""
    statement = select(
        func.max(HarnessRunEvent.sequence)
    ).where(
        HarnessRunEvent.run_id == run_id
    )

    return int(db.scalar(statement) or 0) + 1


def finish_run(
    db: Session,
    run_id: int,
    *,
    result_json: str | None = None,
    error_message: str | None = None,
) -> None:
    """Finish an active run with a result or an error."""
    if (result_json is None) == (error_message is None):
        raise ValueError(
            "Provide either result_json or error_message."
        )

    status = (
        "COMPLETED"
        if result_json is not None
        else "FAILED"
    )

    result = db.execute(
        update(HarnessRun)
        .where(
            HarnessRun.id == run_id,
            HarnessRun.status == "RUNNING",
        )
        .values(
            status=status,
            finished_at=datetime.utcnow(),
            result_json=result_json,
            error_message=error_message,
        )
    )

    if result.rowcount != 1:
        raise RuntimeError("Run is not RUNNING.")


def list_events(
    db: Session,
    run_id: int,
) -> list[HarnessRunEvent]:
    """Read events in execution order."""
    statement = (
        select(HarnessRunEvent)
        .where(HarnessRunEvent.run_id == run_id)
        .order_by(HarnessRunEvent.sequence)
    )

    return list(db.scalars(statement).all())

def next_queued_run_id(
    db: Session,
) -> int | None:
    """Find the oldest queued run; execution claims it atomically."""
    statement = (
        select(HarnessRun.id)
        .where(HarnessRun.status == "QUEUED")
        .order_by(HarnessRun.id)
        .limit(1)
    )

    return db.scalar(statement)

def append_export_event(
    db: Session,
    run_id: int,
    event_name: str,
    data: dict,
) -> None:
    """Record an export decision after a run has completed."""
    import json

    allowed_events = {
        "tool_permission_checked",
        "report_export_approved",
        "report_export_rejected",
    }

    if event_name not in allowed_events:
        raise ValueError("Invalid export event.")

    # Acquire a write lock before allocating the next sequence.
    result = db.execute(
        update(HarnessRun)
        .where(
            HarnessRun.id == run_id,
            HarnessRun.status == "COMPLETED",
        )
        .values(
            step_count=HarnessRun.step_count,
        )
    )

    if result.rowcount != 1:
        raise RuntimeError("Run is not COMPLETED.")

    sequence = next_sequence(
        db=db,
        run_id=run_id,
    )

    db.add(
        HarnessRunEvent(
            run_id=run_id,
            sequence=sequence,
            event=event_name,
            payload_json=json.dumps(
                data,
                ensure_ascii=False,
            ),
        )
    )

    db.flush()