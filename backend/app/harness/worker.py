"""Harness queue operations used by the existing review worker."""

import json
import logging

from sqlalchemy import select

from app.core.database import SESSION_FACTORY
from app.models.harness_run import HarnessRun
from app.repositories import harness_run_repository as repository
from app.services import harness_run_service


LOGGER = logging.getLogger(__name__)


def log_event(event) -> None:
    """Show progress without logging image bytes."""
    LOGGER.info(
        "harness_event %s",
        json.dumps(event, ensure_ascii=False),
    )


def process_one(
    session_factory=SESSION_FACTORY,
) -> bool:
    """Process one queued harness run."""
    with session_factory() as db:
        run_id = repository.next_queued_run_id(db)

    if run_id is None:
        return False

    try:
        harness_run_service.execute_run(
            run_id=run_id,
            session_factory=session_factory,
            on_event=log_event,
        )
    except Exception:
        LOGGER.exception(
            "Harness run %s failed",
            run_id,
        )

    return True


def recover_interrupted_runs(
    session_factory=SESSION_FACTORY,
) -> int:
    """Fail runs interrupted by a stopped worker."""
    with session_factory() as db:
        with db.begin():
            runs = list(
                db.scalars(
                    select(HarnessRun).where(
                        HarnessRun.status == "RUNNING"
                    )
                ).all()
            )

            for run in runs:
                message = (
                    "Worker restarted while this "
                    "harness run was RUNNING."
                )

                event = {
                    "sequence": repository.next_sequence(
                        db=db,
                        run_id=run.id,
                    ),
                    "event": "failed",
                    "data": {"error": message},
                }

                repository.append_event(
                    db=db,
                    run_id=run.id,
                    event=event,
                    payload_json=json.dumps(
                        event["data"],
                        ensure_ascii=False,
                    ),
                )

                repository.finish_run(
                    db=db,
                    run_id=run.id,
                    error_message=message,
                )

            count = len(runs)

    return count