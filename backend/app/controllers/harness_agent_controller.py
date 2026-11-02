"""REST endpoints for harness agent configurations."""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Response,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.harness.config import AgentConfig
from app.models.harness_agent import HarnessAgent
from app.schemas.harness_agent_schema import HarnessAgentResponse
from app.services import harness_agent_service as service


router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Harness agents"],
)


def _to_response(
    agent: HarnessAgent,
) -> HarnessAgentResponse:
    """Convert a database row into an API response."""
    return HarnessAgentResponse(
        id=agent.id,
        name=agent.name,
        config=AgentConfig.model_validate_json(
            agent.config_json
        ),
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.post(
    "",
    response_model=HarnessAgentResponse,
    status_code=201,
)
def create_agent(
    payload: AgentConfig,
    db: Session = Depends(get_db),
) -> HarnessAgentResponse:
    """Create an agent configuration."""
    agent = service.create_agent(
        db=db,
        config=payload,
    )

    return _to_response(agent)


@router.get(
    "",
    response_model=list[HarnessAgentResponse],
)
def list_agents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[HarnessAgentResponse]:
    """List agent configurations."""
    agents = service.list_agents(
        db=db,
        offset=offset,
        limit=limit,
    )

    return [_to_response(agent) for agent in agents]


@router.get(
    "/{agent_id}",
    response_model=HarnessAgentResponse,
)
def get_agent(
    agent_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> HarnessAgentResponse:
    """Get one agent configuration."""
    try:
        agent = service.get_agent(
            db=db,
            agent_id=agent_id,
        )
    except service.AgentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return _to_response(agent)


@router.put(
    "/{agent_id}",
    response_model=HarnessAgentResponse,
)
def update_agent(
    payload: AgentConfig,
    agent_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> HarnessAgentResponse:
    """Replace an agent configuration."""
    try:
        agent = service.update_agent(
            db=db,
            agent_id=agent_id,
            config=payload,
        )
    except service.AgentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return _to_response(agent)


@router.delete(
    "/{agent_id}",
    status_code=204,
    response_class=Response,
)
def delete_agent(
    agent_id: int = Path(ge=1),
    db: Session = Depends(get_db),
) -> Response:
    """Delete an agent configuration."""
    try:
        service.delete_agent(
            db=db,
            agent_id=agent_id,
        )
    except service.AgentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return Response(status_code=204)