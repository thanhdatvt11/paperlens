"""Database operations for harness agents."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.harness_agent import HarnessAgent


def create_agent(
    db: Session,
    name: str,
    config_json: str,
) -> HarnessAgent:
    """Insert an agent in the current transaction."""
    agent = HarnessAgent(
        name=name,
        config_json=config_json,
    )

    db.add(agent)
    db.flush()
    db.refresh(agent)

    return agent


def list_agents(
    db: Session,
    offset: int = 0,
    limit: int = 20,
) -> list[HarnessAgent]:
    """Return agents from newest to oldest."""
    statement = (
        select(HarnessAgent)
        .order_by(HarnessAgent.id.desc())
        .offset(offset)
        .limit(limit)
    )

    return list(db.scalars(statement).all())


def get_agent(
    db: Session,
    agent_id: int,
) -> HarnessAgent | None:
    """Find an agent by ID."""
    return db.get(HarnessAgent, agent_id)


def update_agent(
    db: Session,
    agent: HarnessAgent,
    name: str,
    config_json: str,
) -> HarnessAgent:
    """Replace an agent configuration."""
    agent.name = name
    agent.config_json = config_json

    db.flush()
    db.refresh(agent)

    return agent


def delete_agent(
    db: Session,
    agent: HarnessAgent,
) -> None:
    """Delete an agent in the current transaction."""
    db.delete(agent)
    db.flush()