"""Read persisted harness runs and their execution events."""

import json

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from fastapi import Response
from fastapi.responses import JSONResponse

from app.schemas.harness_run_schema import (
    ReportExportDecision,
)
from app.services import harness_report_service
from app.core.database import get_db
from app.harness.config import AgentConfig
from app.schemas.harness_run_schema import (
    HarnessEventResponse,
    HarnessRunResponse,
)
from app.services import harness_run_service as service
from app.schemas.harness_run_schema import (
    HarnessEventResponse,
    HarnessRunAcceptedResponse,
    HarnessRunCreate,
    HarnessRunResponse,
)

router = APIRouter(
    prefix="/api/v1/harness-runs",
    tags=["Harness runs"],
)


@router.get(
    "/{run_id}",
    response_model=HarnessRunResponse,
)
def get_run(
    run_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> HarnessRunResponse:
    """Read the current state and saved result of a run."""
    try:
        run = service.get_run(
            db=db,
            run_id=run_id,
        )
    except service.HarnessRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return HarnessRunResponse(
        run_id=run.id,
        agent_id=run.agent_id,
        document_id=run.document_id,
        version_id=run.version_id,
        status=run.status,
        instruction=run.instruction,
        pages=json.loads(run.pages_json),
        config=AgentConfig.model_validate_json(
            run.agent_config_json
        ),
        step_count=run.step_count,
        result=(
            json.loads(run.result_json)
            if run.result_json is not None
            else None
        ),
        error=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@router.get(
    "/{run_id}/events",
    response_model=list[HarnessEventResponse],
)
def get_events(
    run_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> list[HarnessEventResponse]:
    """Read events in their original execution order."""
    try:
        events = service.list_run_events(
            db=db,
            run_id=run_id,
        )
    except service.HarnessRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return [
        HarnessEventResponse(
            run_id=event.run_id,
            sequence=event.sequence,
            event=event.event,
            data=json.loads(event.payload_json),
            created_at=event.created_at,
        )
        for event in events
    ]
    
@router.post(
    "",
    response_model=HarnessRunAcceptedResponse,
    status_code=202,
)
def create_run(
    payload: HarnessRunCreate,
    db: Session = Depends(get_db),
) -> HarnessRunAcceptedResponse:
    """Enqueue a run without calling the model."""
    try:
        run = service.create_run(
            db=db,
            agent_id=payload.agent_id,
            document_id=payload.document_id,
            version_id=payload.version_id,
            instruction=payload.instruction,
            pages=payload.pages,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return HarnessRunAcceptedResponse(
        run_id=run.id,
    )
    
def _call_report_service(function, *args):
    """Translate report service errors into HTTP responses."""
    try:
        return function(*args)

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.get("/{run_id}/report-export")
def preview_report_export(
    run_id: int = Path(ge=1),
    db: Session = Depends(get_db),
):
    """Read the exact report preview before approval."""
    return _call_report_service(
        harness_report_service.prepare_export,
        db,
        run_id,
    )


@router.post(
    "/{run_id}/report-export",
    response_class=Response,
)
def decide_report_export(
    payload: ReportExportDecision,
    run_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> Response:
    """Approve a download or reject report export."""
    result = _call_report_service(
        harness_report_service.decide_export,
        db,
        run_id,
        payload.decision,
        payload.content_sha256,
    )

    if result["status"] == "REJECTED":
        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "REJECTED",
            }
        )

    filename = result["filename"]

    return Response(
        content=result["content"],
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )