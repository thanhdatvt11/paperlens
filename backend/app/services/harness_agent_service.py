"""Business operations for harness agent configurations."""

from sqlalchemy.orm import Session

from app.harness.config import AgentConfig
from app.models.harness_agent import HarnessAgent
from app.repositories import harness_agent_repository as repository


class AgentNotFoundError(LookupError):
    """The requested agent does not exist."""


def create_agent(
    db: Session,
    config: AgentConfig,
) -> HarnessAgent:
    """Create a validated agent configuration."""
    with db.begin():
        agent = repository.create_agent(
            db=db,
            name=config.name,
            config_json=config.model_dump_json(),
        )

    return agent


def list_agents(
    db: Session,
    offset: int = 0,
    limit: int = 20,
) -> list[HarnessAgent]:
    """Retrieve an ordered page of agents."""
    return repository.list_agents(
        db=db,
        offset=offset,
        limit=limit,
    )


def get_agent(
    db: Session,
    agent_id: int,
) -> HarnessAgent:
    """Retrieve an agent or raise a not-found error."""
    agent = repository.get_agent(
        db=db,
        agent_id=agent_id,
    )

    if agent is None:
        raise AgentNotFoundError(
            f"Agent {agent_id} does not exist."
        )

    return agent


def update_agent(
    db: Session,
    agent_id: int,
    config: AgentConfig,
) -> HarnessAgent:
    """Replace the full configuration in one transaction."""
    with db.begin():
        agent = get_agent(
            db=db,
            agent_id=agent_id,
        )

        agent = repository.update_agent(
            db=db,
            agent=agent,
            name=config.name,
            config_json=config.model_dump_json(),
        )

    return agent


def delete_agent(
    db: Session,
    agent_id: int,
) -> None:
    """Delete an agent configuration in one transaction."""
    with db.begin():
        agent = get_agent(
            db=db,
            agent_id=agent_id,
        )

        repository.delete_agent(
            db=db,
            agent=agent,
        )


def get_agent_config(
    db: Session,
    agent_id: int,
) -> AgentConfig:
    """Load and validate stored configuration for the runtime."""
    agent = get_agent(
        db=db,
        agent_id=agent_id,
    )

    return AgentConfig.model_validate_json(
        agent.config_json
    )