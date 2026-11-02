"""Response schemas for persisted harness executions."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.harness.config import AgentConfig


class HarnessRunResponse(BaseModel):
    """State, input configuration and result of one run."""

    run_id: int
    agent_id: int | None

    document_id: int
    version_id: int

    status: Literal[
        "QUEUED",
        "RUNNING",
        "COMPLETED",
        "FAILED",
    ]

    instruction: str
    pages: list[int]
    config: AgentConfig

    step_count: int

    result: dict[str, object] | None
    error: str | None

    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    semantic_correctness_verified: Literal[False] = False


class HarnessEventResponse(BaseModel):
    """One stored execution event."""

    run_id: int
    sequence: int
    event: str
    data: dict[str, object]
    created_at: datetime
    
class HarnessRunCreate(BaseModel):
    """Request to enqueue a harness execution."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    agent_id: int = Field(ge=1, strict=True)
    document_id: int = Field(ge=1, strict=True)
    version_id: int = Field(ge=1, strict=True)

    instruction: str = Field(
        min_length=1,
        max_length=2000,
    )

    pages: list[
        Annotated[int, Field(ge=1, strict=True)]
    ] = Field(
        min_length=1,
        max_length=11,
    )


class HarnessRunAcceptedResponse(BaseModel):
    """Acknowledgement after durable enqueue."""

    run_id: int
    status: Literal["QUEUED"] = "QUEUED"
    
class ReportExportDecision(BaseModel):
    """Operator decision for an exact report preview."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]

    content_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )