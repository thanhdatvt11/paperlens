"""Load bounded execution memory from previous completed runs."""

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.harness_run import HarnessRun


def load_memory(
    db: Session,
    *,
    agent_id: int | None,
    document_id: int,
    version_id: int,
    before_run_id: int,
    source_sha256: str,
    pages: list[int],
) -> list[dict[str, object]]:
    """Find up to three relevant previous requests."""
    if agent_id is None:
        return []

    statement = (
        select(HarnessRun)
        .where(
            HarnessRun.agent_id == agent_id,
            HarnessRun.document_id == document_id,
            HarnessRun.version_id == version_id,
            HarnessRun.status == "COMPLETED",
            HarnessRun.id < before_run_id,
        )
        .order_by(HarnessRun.id.desc())
        .limit(20)
    )

    runs = db.scalars(statement).all()
    memory = []
    requested = set(pages)

    for run in runs:
        try:
            source = json.loads(run.source_json)
            previous_pages = json.loads(run.pages_json)

            if (
                not isinstance(source, dict)
                or not isinstance(previous_pages, list)
            ):
                continue

            if source.get("sha256") != source_sha256:
                continue

            if not requested.intersection(previous_pages):
                continue

        except (ValueError, TypeError):
            continue

        memory.append(
            {
                "run_id": run.id,
                "previous_request": run.instruction[:240],
                "pages": previous_pages,
                "note": (
                    "Previous request history only; "
                    "not verified document knowledge."
                ),
            }
        )

        if len(memory) == 3:
            break

    return memory