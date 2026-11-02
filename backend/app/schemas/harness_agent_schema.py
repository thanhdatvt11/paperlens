"""Response schema for harness agents."""

from datetime import datetime

from pydantic import BaseModel

from app.harness.config import AgentConfig


class HarnessAgentResponse(BaseModel):
    """Agent data returned by the API."""

    id: int
    name: str
    config: AgentConfig
    created_at: datetime
    updated_at: datetime