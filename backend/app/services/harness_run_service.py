"""Create durable runs and persist harness execution events."""

from dataclasses import asdict
import json
from app.harness.memory import load_memory
from sqlalchemy.orm import Session

from app.core.database import SESSION_FACTORY
from app.harness.config import AgentConfig
from app.harness.loop import run_harness
from app.harness.pdf_tools import PDFSource, load_pdf_source
from app.models.harness_run import HarnessRun
from app.repositories import harness_run_repository as repository
from app.services.harness_agent_service import get_agent_config


def create_run(
    db: Session,
    agent_id: int,
    document_id: int,
    version_id: int,
    instruction: str,
    pages: list[int],
) -> HarnessRun:
    """Validate input and snapshot configuration and PDF identity."""
    instruction = instruction.strip()

    if not 1 <= len(instruction) <= 2000:
        raise ValueError(
            "instruction must contain 1..2000 characters"
        )

    with db.begin():
        config = get_agent_config(
            db=db,
            agent_id=agent_id,
        )

        source = load_pdf_source(
            db=db,
            document_id=document_id,
            version_id=version_id,
        )

        if not pages or any(
            type(page) is not int
            or not 1 <= page <= source.page_count
            for page in pages
        ):
            raise ValueError(
                "pages must contain valid PDF page positions"
            )
        minimum_steps = len(set(pages)) + 1

        if minimum_steps > config.max_steps:
            raise ValueError(
                f"Selected pages require at least "
                f"{minimum_steps} model calls, but "
                f"max_steps is {config.max_steps}."
            )

        run = HarnessRun(
            agent_id=agent_id,
            document_id=document_id,
            version_id=version_id,
            instruction=instruction,
            pages_json=json.dumps(sorted(set(pages))),
            agent_config_json=config.model_dump_json(),
            source_json=json.dumps(
                asdict(source),
                ensure_ascii=False,
            ),
            status="QUEUED",
            step_count=0,
        )

        run = repository.create_run(
            db=db,
            run=run,
        )

    return run


def execute_run(
    run_id: int,
    *,
    session_factory=SESSION_FACTORY,
    adapter=None,
    on_event=None,
) -> dict:
    """Execute a claimed run outside database transactions."""
    with session_factory() as db:
        with db.begin():
            run = repository.claim_run(
                db=db,
                run_id=run_id,
            )

            job = {
                "instruction": run.instruction,
                "pages_json": run.pages_json,
                "agent_config_json": run.agent_config_json,
                "source_json": run.source_json,
                "agent_id": run.agent_id,
                "instruction": run.instruction,
                "pages_json": run.pages_json,
                "agent_config_json": run.agent_config_json,
                "source_json": run.source_json,
            }

    def persist_event(event):
        with session_factory() as db:
            with db.begin():
                repository.append_event(
                    db=db,
                    run_id=run_id,
                    event=event,
                    payload_json=json.dumps(
                        event["data"],
                        ensure_ascii=False,
                    ),
                )

        if on_event is not None:
            on_event(event)

    try:
        config = AgentConfig.model_validate_json(
            job["agent_config_json"]
        )

        source = PDFSource(
            **json.loads(job["source_json"])
        )
        pages = json.loads(job["pages_json"])
        memory_context = []

        if config.memory_enabled:
            with session_factory() as db:
                memory_context = load_memory(
                    db=db,
                    agent_id=job["agent_id"],
                    document_id=source.document_id,
                    version_id=source.version_id,
                    before_run_id=run_id,
                    source_sha256=source.sha256,
                    pages=pages,
                )

        result = run_harness(
            config=config,
            source=source,
            instruction=job["instruction"],
            pages=pages,
            memory_context=memory_context,
            adapter=adapter,
            on_event=persist_event,
        )

        result["run_id"] = run_id

        with session_factory() as db:
            with db.begin():
                repository.finish_run(
                    db=db,
                    run_id=run_id,
                    result_json=json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                )

        return result

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: {exc}"
        )[:1000]

        with session_factory() as db:
            with db.begin():
                failure_event = {
                    "sequence": repository.next_sequence(
                        db=db,
                        run_id=run_id,
                    ),
                    "event": "failed",
                    "data": {"error": error},
                }

                repository.append_event(
                    db=db,
                    run_id=run_id,
                    event=failure_event,
                    payload_json=json.dumps(
                        failure_event["data"],
                        ensure_ascii=False,
                    ),
                )

                repository.finish_run(
                    db=db,
                    run_id=run_id,
                    error_message=error,
                )

        raise RuntimeError(
            f"Harness run {run_id} failed: {error}"
        ) from exc
        
class HarnessRunNotFoundError(LookupError):
    """The requested harness run does not exist."""


def get_run(
    db: Session,
    run_id: int,
) -> HarnessRun:
    """Retrieve one run or raise a not-found error."""
    run = repository.get_run(
        db=db,
        run_id=run_id,
    )

    if run is None:
        raise HarnessRunNotFoundError(
            f"Harness run {run_id} does not exist."
        )

    return run


def list_run_events(
    db: Session,
    run_id: int,
):
    """Read events only when the parent run exists."""
    get_run(
        db=db,
        run_id=run_id,
    )

    return repository.list_events(
        db=db,
        run_id=run_id,
    )